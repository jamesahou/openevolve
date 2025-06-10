from openevolve.constants import HOTSWAP_ENVVAR, WORKSPACE_ROOT, CONTAINER_IMPS_PATH, LIBRARY_NAME
from openevolve.custom_types import AbsPath

from types import FunctionType

import functools
import inspect
import sys
import ast
import os
from typing import Iterator, Tuple
from pathlib import Path

class NoImplementationSpecified(Exception):
    pass

def get_relative_path(func: FunctionType, root: AbsPath) -> str:
    """Get the file path of the provided function relative to the project root."""
    absolute_path = inspect.getfile(func)
    print(absolute_path)
    relative_path = str(Path(absolute_path).relative_to(root))
    return relative_path

def get_implementation_path(func: FunctionType) -> str:
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
    return imp_path

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

'''
import os
import functools
import threading

def load_from_file(func_name, path, signature):
    """
    Read the file at 'path', exec it in its own namespace,
    and return the function object named func_name.
    """
    namespace = {}
    with open(path, 'r') as f:
        code = f.read()
    
    func_code = f"def {func_name}{signature}:\n"
    for line in code.splitlines():
        func_code += "    " + line + "\n"

    exec(compile(func_code, path, 'exec'), namespace)
    try:
        impl = namespace[func_name]
    except KeyError:
        raise ImportError(f"No function named {func_name!r} in {path}")
    return impl

def hotswap(func: FunctionType):
    """
    env_var: name of ENV var to read
    mapping: dict from ENV-value -> filesystem path
    default_key: mapping key to use if env_var unset
    """
    cache = {}
    lock = threading.Lock()

    #def decorator(original_func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        key = os.getenv(HOTSWAP_ENVVAR, "-1")
        try:
            path = get_implementation_path(func)
            unique_id = path + ' ' + func.__qualname__
        except KeyError:
            raise RuntimeError(f"No implementation for {HOTSWAP_ENVVAR}={key!r}")
        # load & cache
        if unique_id not in cache:
            with lock:
                if unique_id not in cache:
                    cache[unique_id] = load_from_file(func.__name__, path, signature = inspect.signature(func))
        impl = cache[unique_id]
        return impl(*args, **kwargs)
    return wrapper
    #return decorator
'''

def hotswap(func: FunctionType):
    if hasattr(func, "_openevolve_already_hotswapped"):
        return func

    implementation = get_implementation(func)

    if "super()" in implementation:
        return func

    """
    qualname_parts = func.__qualname__.rsplit('.', 1)
    implementation = get_implementation(func)
    
    if len(qualname_parts) > 1:
        class_name = func.__qualname__.rsplit('.', 1)[-2]

        # absolute_path = inspect.getfile(func)
        if "super()" in implementation:
            implementation = implementation.replace("super()", f"super({class_name}, self)")
    """

    header = inspect.signature(func)
    #def wrapper(*args, **kwargs):
    func_code = f"def _dynamic_func{header}:\n"
    for line in implementation.splitlines():
        func_code += "    " + line + "\n"
    local_vars = {}
    exec(func_code, func.__globals__, local_vars)

    dynamic_func = local_vars["_dynamic_func"]
    #dynamic_func = functools.update_wrapper(dynamic_func, func)

    #print(args)
    #print(kwargs)
    #dynamic_func.__class__ = func.__class__
    #return local_vars["_dynamic_func"](*args, **kwargs)
    dynamic_func._openevolve_already_hotswapped = True

    return dynamic_func

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
):
    decorator = ast.Attribute(
        value = ast.Name(id="openevolve", ctx=ast.Load()),
        attr = "hotswap",
        ctx = ast.Load(),
    )
    
    """Apply a decorator to a function defined in a specific file."""
    for node, node_qualname in qualwalk(tree):
        if isinstance(node, ast.FunctionDef) and node_qualname == qualname:
            node.decorator_list.append(decorator)

def remove_decorator(
    tree: ast.Module,
    qualname: str,
):
    """Remove a specific decorator from a function in the AST."""
    for node, node_qualname in qualwalk(tree):
        if isinstance(node, ast.FunctionDef) and node_qualname == qualname:
            new_decorator_list = []

            for decorator in node.decorator_list:
                if isinstance(decorator, ast.Attribute) and decorator.value.id == "openevolve" and decorator.attr == "hotswap":
                    continue  # Skip the decorator we want to remove
                new_decorator_list.append(decorator)

            node.decorator_list = new_decorator_list

def does_import_openevolve(tree: ast.Module) -> bool:
    """Check if an absolute import statement for openevolve exists as a child of the root of the AST."""
    for node in tree.body:
        if isinstance(node, ast.Import) and len(node.names) == 1 and node.names[0].name == LIBRARY_NAME:
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
        if isinstance(node, ast.Import) and len(node.names) == 1 and node.names[0].name == LIBRARY_NAME:
            tree.body.remove(node)
            break