import ast
import os
from collections import defaultdict

class CallGraphBuilder(ast.NodeVisitor):
    def __init__(self, filename, module_name):
        self.filename = filename
        self.module_name = module_name
        self.current_function = None
        self.call_graph = defaultdict(set)
        self.defined_functions = set()

    def visit_FunctionDef(self, node):
        func_name = f"{self.module_name}.{node.name}"
        self.defined_functions.add(func_name)
        prev_function = self.current_function
        self.current_function = func_name
        self.generic_visit(node)
        self.current_function = prev_function

    def visit_AsyncFunctionDef(self, node):
        self.visit_FunctionDef(node)

    def visit_Call(self, node):
        if self.current_function:
            if isinstance(node.func, ast.Attribute):
                called = node.func.attr
            elif isinstance(node.func, ast.Name):
                called = node.func.id
            else:
                called = None
            if called:
                self.call_graph[self.current_function].add(called)
        self.generic_visit(node)

def get_py_files(root_dir):
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.py'):
                yield os.path.join(dirpath, filename)

def module_name_from_path(root_dir, path):
    rel = os.path.relpath(path, root_dir)
    no_ext = os.path.splitext(rel)[0]
    return no_ext.replace(os.sep, ".")

def build_full_call_graph(root_dir):
    call_graph = defaultdict(set)
    function_defs = set()
    for pyfile in get_py_files(root_dir):
        with open(pyfile, "r", encoding="utf-8") as f:
            try:
                tree = ast.parse(f.read(), filename=pyfile)
            except Exception:
                continue
        module_name = module_name_from_path(root_dir, pyfile)
        builder = CallGraphBuilder(pyfile, module_name)
        builder.visit(tree)
        for k, v in builder.call_graph.items():
            call_graph[k].update(v)
        function_defs.update(builder.defined_functions)
    return call_graph, function_defs

def build_dag(call_graph, function_defs):
    dag = defaultdict(set)
    for caller, callees in call_graph.items():
        for callee in callees:
            # Only include edges to known functions in the codebase
            matches = [f for f in function_defs if f.endswith(f".{callee}")]
            for match in matches:
                dag[caller].add(match)
    return dag

def print_dag(dag):
    for caller, callees in dag.items():
        for callee in callees:
            print(f"{caller} -> {callee}")

if __name__ == "__main__":
    # Set this to the root of your codebase
    root_dir = "/Users/ryanrudes/GitHub/OpenEvolve/astropy/astropy" # os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    call_graph, function_defs = build_full_call_graph(root_dir)
    dag = build_dag(call_graph, function_defs)
    print_dag(dag)