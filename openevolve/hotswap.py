from openevolve.constants import HOTSWAP_ENVVAR, WORKSPACE_ROOT, CONTAINER_IMPS_PATH

from types import FunctionType

import warnings
import inspect
import os

class NoImplementationSpecified(Exception):
    pass

def get_relative_path(func: FunctionType, root: str) -> str:
    """Get the file path of the provided function relative to the project root."""
    absolute_path = inspect.getfile(func)
    relative_path = absolute_path.rsplit(root, 1)[-1].lstrip('/')
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
        return
        
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