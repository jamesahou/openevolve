#!/usr/bin/env python3
import argparse
import builtins
import sys
from pathlib import Path
from typing import Callable
import types
from IPython import embed
import os
from collections import deque
import ast

if sys.version_info >= (3, 12):
    from sys import monitoring

# --- Builtins storage ---
builtins._dbg_storage = dict(
    fns_to_evolve=[],
    fns_to_run=[],
    run_codeobjs=set(),
    evolve_codeobjs=set(),
    call_graph={},
    fn_locs={},
)

def evolve(func: Callable) -> Callable:
    builtins._dbg_storage['fns_to_evolve'].append(func)
    if hasattr(func, "__code__"):
        builtins._dbg_storage['evolve_codeobjs'].add(func.__code__)
    return func

def run(func: Callable) -> Callable:
    builtins._dbg_storage['fns_to_run'].append(func)
    if hasattr(func, "__code__"):
        builtins._dbg_storage['run_codeobjs'].add(func.__code__)
    return func

builtins._dbg_storage['evolve'] = evolve
builtins._dbg_storage['run'] = run

def _is_in_base_dir(base_dir: Path, file_path_str: str) -> bool:
    """
    Return True if 'file_path_str' is inside or is the same as 'base_dir'.
    """
    try:
        if file_path_str.startswith("<"):
            return False
        
        file_path = (base_dir / file_path_str).resolve()
        file_path.relative_to(base_dir)
        return True
    except ValueError:
        return False

class Extractor:
    def __init__(self):
        self.script_path: Path = None
        self.base_dir: Path = None
        self.watch_stack_depth = 0

    def find_path(self, start, end):
        queue = deque([(start, [start])])
        visited = set()

        while queue:
            current, path = queue.popleft()
            visited.add(current)

            if current == end:
                return path

            for neighbor in builtins._dbg_storage['call_graph'].get(current, []):
                if neighbor not in visited:
                    queue.append((neighbor, path + [neighbor]))

        return None
    
    def extract_function_code(self, file_path, qualname):
        """
        Extract the code and header for the function/method with the given qualname.
        """
        with open(file_path, 'r') as f:
            source = f.read()

        tree = ast.parse(source)

        class QualnameVisitor(ast.NodeVisitor):
            def __init__(self, target_qualname):
                self.target_qualname = target_qualname
                self.result = None
                self.stack = []

            def visit_ClassDef(self, node):
                self.stack.append(node.name)
                self.generic_visit(node)
                self.stack.pop()

            def visit_FunctionDef(self, node):
                self.stack.append(node.name)
                current_qualname = ".".join(self.stack)
                if current_qualname == self.target_qualname and self.result is None:
                    class_name = self.stack[-2] if len(self.stack) > 1 else None
                    self.result = {
                        "class_name": class_name,
                        "function_node": node
                    }
                self.generic_visit(node)
                self.stack.pop()

        visitor = QualnameVisitor(qualname)
        visitor.visit(tree)

        if visitor.result:
            node = visitor.result["function_node"]
            start_line = node.lineno - 1
            end_line = node.end_lineno
            lines = source.splitlines()[start_line:end_line]
            if not lines:
                return "", visitor.result["class_name"], ""
            lines[0] = lines[0].lstrip()
            # Extract the full header (all lines up to and including the line with the colon ending the header)
            header_lines = []
            found_colon = False
            for line in lines:
                header_lines.append(line)
                if line.rstrip().endswith(":"):
                    found_colon = True
                    break
            header = "\n".join(header_lines) if found_colon else header_lines[0]

            # Dedent function body if inside a class
            class_name = visitor.result["class_name"]
            if class_name:
                # Remove one indentation level (4 spaces or a tab) from all lines except the header
                dedented_lines = [header_lines[0]]
                for line in lines[len(header_lines):]:
                    if line.startswith("    "):
                        dedented_lines.append(line[4:])
                    elif line.startswith("\t"):
                        dedented_lines.append(line[1:])
                    else:
                        dedented_lines.append(line)
                code = "\n".join(dedented_lines)
            else:
                code = "\n".join(lines)

            return code, class_name, header

        return None, None, None

    def trace(self, frame: types.FrameType, event: str, arg):
        """Equivalent logic for Python < 3.12 using sys.settrace."""
        if event not in ("call", "return"):
            return self.trace

        code = frame.f_code
        if event == "call":
            caller_frame = frame.f_back
            if caller_frame is None:
                return self.trace

            caller_code = caller_frame.f_code
            caller_file = caller_code.co_filename
            callee_file = code.co_filename

            # Always store function locations if they're in our base directory
            if _is_in_base_dir(self.base_dir, caller_file):
                caller_name = caller_code.co_name
                qualname = getattr(caller_code, "co_qualname", caller_name)
                builtins._dbg_storage['fn_locs'][qualname] = (caller_file, caller_code.co_firstlineno, qualname)

            if _is_in_base_dir(self.base_dir, callee_file):
                callee_name = code.co_name
                qualname = getattr(code, "co_qualname", callee_name)
                builtins._dbg_storage['fn_locs'][qualname] = (callee_file, code.co_firstlineno, qualname)

            # Only process calls where either caller or callee is in our base directory
            if not (_is_in_base_dir(self.base_dir, caller_file) or
                    _is_in_base_dir(self.base_dir, callee_file)):
                return self.trace

            # Always record the call in our graph if both functions are in our directory
            if _is_in_base_dir(self.base_dir, caller_file) and _is_in_base_dir(self.base_dir, callee_file):
                caller_qualname = getattr(caller_code, "co_qualname", caller_code.co_name)
                callee_qualname = getattr(code, "co_qualname", code.co_name)
                builtins._dbg_storage['call_graph'].setdefault(caller_qualname, set()).add(callee_qualname)

        elif event == "return":
            if code in builtins._dbg_storage['run_codeobjs'] and self.watch_stack_depth > 0:
                self.watch_stack_depth -= 1

        return self.trace

    def run(self, file: Path, script_args: list, depth: int=-1):
        assert(depth >= -1 and isinstance(depth, int))
        self.script_path = file.resolve()
        self.base_dir = self.script_path.parent
        print("base_dir", self.base_dir)
        sys.argv.pop(0)
        sys.argv.extend(script_args)
        original_argv = sys.argv.copy()

        try:
            # Clear builtins data
            builtins._dbg_storage['fns_to_evolve'].clear()
            builtins._dbg_storage['fns_to_run'].clear()
            builtins._dbg_storage['run_codeobjs'].clear()
            builtins._dbg_storage['evolve_codeobjs'].clear()
            builtins._dbg_storage['call_graph'].clear()

            # Compile user script
            code_text = file.read_text()
            compiled = compile(code_text, filename=file.name, mode="exec")
            
            sys.settrace(self.trace)
            module_globals = {"__name__": "__main__"}
            exec(compiled, module_globals)

        finally:
            sys.settrace(None)
            sys.argv = original_argv

        opt_qualnames = {getattr(fn.__code__, "co_qualname", fn.__name__) for fn in builtins._dbg_storage['fns_to_evolve']}
        watch_qualnames = {getattr(fn.__code__, "co_qualname", fn.__name__) for fn in builtins._dbg_storage['fns_to_run']}
        print("Evolving functions: ", opt_qualnames)
        print("Running functions: ", watch_qualnames)

        # Only get the first valid path
        path = None
        for opt_qualname in opt_qualnames:
            for watch_qualname in watch_qualnames:
                candidate_path = self.find_path(watch_qualname, opt_qualname)
                if candidate_path:
                    print(f"Path from {watch_qualname} to {opt_qualname}: {candidate_path}")
                    path = candidate_path
                    break
            if path:
                break

        if not path:
            print("No path found.")
            return "", None, {}

        func_class = dict()
        func_header = dict()
        spec_code = []
        spec_structured = {}
        # Extract the code in the order of the path
        for qualname in path[-depth:]:
            file_path, line_no, _ = builtins._dbg_storage["fn_locs"][qualname]
            code, class_name, header = self.extract_function_code(self.base_dir / file_path, qualname)
            location_comment = f"# Location: {file_path}:{line_no}"
            if class_name:
                location_comment += f" | Class: {class_name}"
            spec_code.append(f"{location_comment}\n{code}")
            spec_structured[qualname] = code

            func_class[qualname] = class_name
            func_header[qualname] = header
        
        # Write the code to a file
        spec_code_str = "\n\n".join(spec_code)
        spec_code_str = f"# Minimal code path from {watch_qualnames} to {opt_qualnames}\n\n{spec_code_str}"
        spec_code_path = str(self.script_path)[:-3] + "_spec.py"

        with open(spec_code_path, 'w') as f:
            f.write(spec_code_str)

        # Get location of functions on path, now also include header
        loc_dict = {
            qualname: {
                "file_path": file_path,
                "line_no": line_no,
                "class": func_class.get(qualname, None),
                "header": func_header.get(qualname, None),
                "qualname": qualname
            }
            for qualname in path[-depth:]
            for (file_path, line_no, _) in [builtins._dbg_storage["fn_locs"][qualname]]
        }

        return spec_structured, path[-depth:], loc_dict

def extract_code(eval_file: Path, args: list, depth=-1) -> str:
    extractor = Extractor()
    spec_structured, path, loc_dict = extractor.run(eval_file, args, depth=depth)
    return spec_structured, path, loc_dict

def add_decorators(loc_dict, decorator="@funsearch.hotswap"):
    """
    Edit the original file to add the specified decorator to the functions.
    Import funsearch at the top of the file.
    """
    
    for fn_name, loc in loc_dict.items():
        file_path = Path(loc["file_path"])
        if not file_path.exists():
            continue
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        # Add the decorator above the function definition
        line_no = loc["line_no"] - 1  # Convert to 0-based index
        
        if loc["class"]:
            lines[line_no] = f"    {decorator}\n" + lines[line_no]
        else:
            lines[line_no] = f"{decorator}\n" + lines[line_no]
        
        # lines.insert(0, "import funsearch\n\n") 
        # import funsearch under __future__ imports, otherwise at the top of the file
        future_import_index = -1
        for i, line in enumerate(lines):
            if line.startswith("from __future__ import") or line.startswith("import __future__"):
                future_import_index = i
                break

        if future_import_index != -1:
            lines.insert(future_import_index + 1, "import funsearch\n")
        else:
            lines.insert(0, "import funsearch\n\n")
        

        with open(file_path, 'w') as f:
            f.writelines(lines)
        
        print(f"Added decorator to {file_path} at line {line_no + 1}")

def remove_decorators(loc_dict, decorator="@funsearch.hotswap"):
    """
    Edit the original file to remove the specified decorator from the functions.
    """
    
    for fn_name, loc in loc_dict.items():
        file_path = Path(loc["file_path"])
        if not file_path.exists():
            continue
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        decorator_lines = []
        for i, line in enumerate(lines):
            if decorator in line.strip():
                decorator_lines.append(i)

        for line_no in decorator_lines:
            # Remove the decorator line
            if loc["class"]:
                lines[line_no] = "\n"
            else:
                lines[line_no] = "\n"
        
        with open(file_path, 'w') as f:
            f.writelines(lines)

        print(f"Removed decorator from {file_path} at line {line_no + 1}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", help="file to debug")
    parser.add_argument("--args", nargs="*", help="arguments to pass to file")
    args = parser.parse_args()

    spec_structured, path, loc_dict = extract_code(Path(args.file), args.args)

    print(spec_structured)

    add_decorators(loc_dict)
    remove_decorators(loc_dict)
