from openevolve.constants import HOTSWAP_ENVVAR, WORKSPACE_ROOT, CONTAINER_IMPS_PATH, LIBRARY_NAME
from openevolve.custom_types import AbsPath

from types import FunctionType

import inspect
import ast
import os
from typing import Iterator, Tuple
from pathlib import Path

class NoImplementationSpecified(Exception):
    pass

def get_relative_path(func: FunctionType, root: AbsPath) -> str:
    """Get the file path of the provided function relative to the project root."""
    absolute_path = inspect.getfile(func)
    relative_path = str(Path(absolute_path).relative_to(root))
    return relative_path

def get_implementation(func: FunctionType) -> str:
    """Get the implementation of the function as a string."""
    filepath = get_relative_path(func, WORKSPACE_ROOT)
    qualname = func.__qualname__
    implementation_id = os.environ.get(HOTSWAP_ENVVAR)

    if implementation_id == "-1" or implementation_id is None:
        raise NoImplementationSpecified(
            "Environment variable 'OPENEVOLE_HOTSWAP_IMP' is not set or is set to '-1'. "
            "Using the original function definition instead."
        )
        
    imp_name = qualname + " " + implementation_id
    imp_path = os.path.join(CONTAINER_IMPS_PATH, filepath, imp_name)

    try:
        return open(imp_path, 'r').read() 
    except FileNotFoundError:
        raise NoImplementationSpecified(
            f"No implementation found for function '{func.__name__}'. "
            "Using the original definition."
        )

def hotswap(func: FunctionType):
    implementation = get_implementation(func)

    def wrapper(*args, **kwargs):
        func_code = f"def _dynamic_func(self):\n"
        for line in implementation.splitlines():
            func_code += "    " + line + "\n"
        local_vars = {}
        exec(func_code, func.__globals__, local_vars)
        return local_vars["_dynamic_func"](*args, **kwargs)

    wrapper.__name__ = func.__name__
    wrapper.__qualname__ = func.__qualname__
    wrapper.__doc__ = func.__doc__

    return wrapper

QUALIFIED_CONSTRUCTS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)

def qualwalk(tree: ast.Module) -> Iterator[tuple[ast.AST, str]]:
    """Yield (node, qualname) pairs for all nodes, like ast.walk but with qualified names for functions/classes."""
    def visit(node, qualname=""):
        # Build qualname for functions and classes
        if isinstance(node, QUALIFIED_CONSTRUCTS):
            child_qualname = qualname + '.' + node.name
            output_qualname = child_qualname[1:]
        else:
            child_qualname = qualname
            output_qualname = None

        yield node, output_qualname

        for child in ast.iter_child_nodes(node):
            yield from visit(child, child_qualname)

    yield from visit(tree)

def apply_decorator(
    tree: ast.Module,
    qualname: str,
    decorator: str = "openevolve.hotswap"
):
    """Apply a decorator to a function defined in a specific file."""
    for node, node_qualname in qualwalk(tree):
        #print(node_qualname, qualname)
        if isinstance(node, ast.FunctionDef) and node_qualname == qualname:
            node.decorator_list.insert(0, ast.Name(id=decorator, ctx=ast.Load()))

def remove_decorator(
    tree: ast.Module,
    qualname: str,
    decorator: str = "openevolve.hotswap",
):
    """Remove a specific decorator from a function in the AST."""
    for node, node_qualname in qualwalk(tree):
        if isinstance(node, ast.FunctionDef) and node_qualname == qualname:
            node.decorator_list = [
                d for d in node.decorator_list if not isinstance(d, ast.Name) or d.id != decorator
            ]

def does_import_openevolve(tree: ast.Module) -> bool:
    """Check if an absolute import statement for openevolve exists as a child of the root of the AST."""
    for node in tree.body:
        if isinstance(node, ast.Import) and len(node.names) == 1 and node.names[0] == LIBRARY_NAME:
            return True
    return False

def import_openevolve(tree: ast.Module):
    """Add an absolute import statement for openevolve to the AST."""

    if does_import_openevolve(tree):
        return
        
    # Find the last __future__ import
    insert_index = 0
    for idx, node in enumerate(tree.body):
        if isinstance(node, ast.ImportFrom) and node.module == "__future__":
            insert_index = idx + 1

    # Create the import node
    import_node = ast.Import(names=[ast.alias(name=LIBRARY_NAME, asname=None)])
    tree.body.insert(insert_index, import_node)

def unimport_openevolve(tree: ast.Module):
    """Remove the absolute import statement of openevolve from the top level of the AST."""
    for node in tree.body:
        if isinstance(node, ast.Import) and len(node.names) == 1 and node.names[0] == LIBRARY_NAME:
            tree.body.remove(node)
            break