from typing import Optional
from types import FunctionType

import warnings
import inspect
import os

from funsearch.constants import HOTSWAP_ENVVAR

IMPLEMENTATIONS_ROOT = "implementations/"
PROJECT_ROOT = "workspace/"


def get_relative_path(func: FunctionType, root: str) -> str:
    """Get the file path of the provided function relative to the project root."""
    absolute_path = inspect.getfile(func)
    relative_path = absolute_path.split(root, 1)[-1].lstrip('/')
    return relative_path


def get_implementation(
    func: FunctionType
) -> Optional[str]:
    """Get the implementation of the function as a string."""
    filepath = get_relative_path(func, PROJECT_ROOT)
    qualname = func.__qualname__
    procname = os.environ.get(HOTSWAP_ENVVAR, "")
    imp_name = f"{qualname} {procname}"
    imp_path = os.path.join(IMPLEMENTATIONS_ROOT, PROJECT_ROOT, filepath, imp_name)

    return open(imp_path, 'r').read() if os.path.exists(imp_path) else None


def evolve(func: FunctionType):
    implementation = get_implementation(func)

    if implementation is None:
        raise ValueError(f"No implementation found for function '{func.__name__}'.")

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

