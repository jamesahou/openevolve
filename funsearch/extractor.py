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
    
    def extract_function_code(self, file_path, function_name):
        with open(file_path, 'r') as f:
            source = f.read()

        tree = ast.parse(source)

        class FunctionClassVisitor(ast.NodeVisitor):
            def __init__(self, target_name):
                self.target_name = target_name
                self.result = None

            def visit_ClassDef(self, node):
                for item in node.body:
                    if isinstance(item, ast.FunctionDef) and item.name == self.target_name:
                        self.result = {
                            "class_name": node.name,
                            "function_node": item
                        }
                self.generic_visit(node)

            def visit_FunctionDef(self, node):
                if node.name == self.target_name and self.result is None:
                    self.result = {
                        "class_name": None,
                        "function_node": node
                    }

        visitor = FunctionClassVisitor(function_name)
        visitor.visit(tree)

        if visitor.result:
            node = visitor.result["function_node"]
            start_line = node.lineno - 1
            # The header ends at the line before the first statement in the body
            if node.body:
                header_end_line = node.body[0].lineno - 1
            else:
                header_end_line = node.end_lineno - 1  # fallback
            end_line = node.end_lineno
            lines = source.splitlines()[start_line:end_line]
            if not lines:
                return "", visitor.result["class_name"], ""
            # Extract the header lines (may be multiline)
            header_lines = source.splitlines()[start_line:header_end_line]
            header = "\n".join([line.lstrip() for line in header_lines])
            lines[0] = lines[0].lstrip()
            return '\n'.join(lines), visitor.result["class_name"], header

        return None, None, None

    def _legacy_trace(self, frame: types.FrameType, event: str, arg):
        """Equivalent logic for Python < 3.12 using sys.settrace."""
        if event not in ("call", "return"):
            return self._legacy_trace

        code = frame.f_code
        if event == "call":
            caller_frame = frame.f_back
            if caller_frame is None:
                return self._legacy_trace

            caller_code = caller_frame.f_code
            caller_file = caller_code.co_filename
            callee_file = code.co_filename
            
            # Always store function locations if they're in our base directory
            if _is_in_base_dir(self.base_dir, caller_file):
                caller_name = caller_code.co_name
                builtins._dbg_storage['fn_locs'][caller_name] = (caller_file, caller_code.co_firstlineno)
            
            if _is_in_base_dir(self.base_dir, callee_file):
                callee_name = code.co_name
                builtins._dbg_storage['fn_locs'][callee_name] = (callee_file, code.co_firstlineno)

            # Only process calls where either caller or callee is in our base directory
            if not (_is_in_base_dir(self.base_dir, caller_file) or
                    _is_in_base_dir(self.base_dir, callee_file)):
                return self._legacy_trace

            # Always record the call in our graph if both functions are in our directory
            if _is_in_base_dir(self.base_dir, caller_file) and _is_in_base_dir(self.base_dir, callee_file):
                caller_name = caller_code.co_name
                callee_name = code.co_name
                builtins._dbg_storage['call_graph'].setdefault(caller_name, set()).add(callee_name)

            # Update watch depth if needed
            if caller_code in builtins._dbg_storage['run_codeobjs'] or code in builtins._dbg_storage['run_codeobjs']:
                self.watch_stack_depth += 1

            # Stop tracing if we hit an optimized function
            if code in builtins._dbg_storage['evolve_codeobjs']:
                return None

        elif event == "return":
            if code in builtins._dbg_storage['run_codeobjs'] and self.watch_stack_depth > 0:
                self.watch_stack_depth -= 1

        return self._legacy_trace

    def run(self, file: Path, script_args: list):
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
            
            sys.settrace(self._legacy_trace)
            exec(compiled, {"__name__": "__main__"})

        finally:
            sys.settrace(None)
            sys.argv = original_argv

        all_fns = set()
        opt_names = {fn.__name__ for fn in builtins._dbg_storage['fns_to_evolve']}
        watch_names = {fn.__name__ for fn in builtins._dbg_storage['fns_to_run']}
        print("Evolving functions: ", opt_names)
        print("Running functions: ", watch_names)
        for opt_fn in builtins._dbg_storage['fns_to_evolve']:
            for watch_fn in builtins._dbg_storage['fns_to_run']:
                path = self.find_path(watch_fn.__name__, opt_fn.__name__)
                if path:
                    print(f"Path from {watch_fn.__name__} to {opt_fn.__name__}: {path}")
                    all_fns.update(path)

        func_class = dict()
        func_header = dict()
        spec_code = []
        # Extract the code from the path
        for fn_name in all_fns:
            file_path, line_no = builtins._dbg_storage["fn_locs"][fn_name]
            code, class_name, header = self.extract_function_code(self.base_dir / file_path, fn_name)
            if fn_name in opt_names:
                code = code.replace("def ", f"@funsearch.evolve\ndef ")
            elif fn_name in watch_names:
                code = code.replace("def ", f"@funsearch.run\ndef ")
            location_comment = f"# Location: {file_path}:{line_no}"
            if class_name:
                location_comment += f" | Class: {class_name}"
            spec_code.append(f"{location_comment}\n{code}")
            func_class[fn_name] = class_name
            func_header[fn_name] = header
        
        # Write the code to a file
        spec_code_str = "\n\n".join(spec_code)
        spec_code_str = f"# Minimal code path from {watch_names} to {opt_names}\n\n{spec_code_str}"
        spec_code_path = str(self.script_path)[:-3] + "_spec.py"

        with open(spec_code_path, 'w') as f:
            f.write(spec_code_str)

        # Get location of functions on path, now also include header
        loc_dict = {
            fn_name: {"file_path": file_path, "line_no": line_no, "class": func_class.get(fn_name, None), "header": func_header.get(fn_name, None)}
            for fn_name, (file_path, line_no) in builtins._dbg_storage["fn_locs"].items()
            if fn_name in all_fns
        }

        return spec_code_str, path, loc_dict

def extract_code(eval_file: Path, args: list) -> str:
    extractor = Extractor()
    spec_code_str, path, loc_dict = extractor.run(eval_file, args)
    return spec_code_str, path, loc_dict

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

    spec_code_str, path, loc_dict = extract_code(Path(args.file), args.args)

    print(loc_dict)

    add_decorators(loc_dict)
    remove_decorators(loc_dict)
