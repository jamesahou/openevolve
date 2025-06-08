from funsearch.structured_outputs import ProgramImplementation

import dataclasses
import io
import ast
import textwrap
from typing import List, Tuple, Sequence, Any, Dict, Iterable
import re
from enum import Enum
from types import FullName, FuncMeta

@dataclasses.dataclass
class FuncHeader:
    """A parsed Python function header."""

    name: str
    args: List[str]
    return_type: str = ""

    def __str__(self) -> str:
        """Return the function header as a string."""
        args = sorted(self.args, key=lambda x: (x != "self", x))
        args = ", ".join(args)
        if self.return_type:
            return f"def {self.name}({args}) -> {self.return_type}:"
        else:
            return f"def {self.name}({args}):"

    def __hash__(self):
        return hash(str(self))


class Decorator(Enum):
    NONE = ""
    CLASSMETHOD = "classmethod"
    STATICMETHOD = "staticmethod"
    PROPERTY = "property"

@dataclasses.dataclass
class Function:
    """A parsed Python function."""

    name: str
    header: FuncHeader
    body: str
    path: str = ""
    line_no: int = 0
    qualname: str = ""
    decorator: Decorator = Decorator.NONE

    def __str__(self) -> str:
        header_str = str(self.header).strip()
        body = self.body.strip()
        if not body.endswith("\n"):
            body += "\n"
        return f"{header_str}\n{body}"

    def to_str(self, version: int | None = None) -> str:
        """Return the function as a string.
        If *version* is given, the function name in the header is suffixed with
        ``_v<version>`` (e.g. ``def foo()`` → ``def foo_v1()``).  Otherwise this is
        identical to ``str(self)``.
        """
        if version is None:
            return str(self)
        header_copy = FuncHeader(
            name=f"{self.header.name}_v{version}",
            args=list(self.header.args),
            return_type=self.header.return_type,
        )
        body = self.body.strip()
        return f"{header_copy}\n{body}"

    def __setattr__(self, name: str, value: str) -> None:
        # Ensure there aren't leading & trailing new lines in `body`.
        if name == "body":
            value = value.strip("\n")
        super().__setattr__(name, value)


@dataclasses.dataclass(frozen=True)
class Program:
    """A parsed Python program."""

    functions: list[Function]

    def __str__(self) -> str:
        return "\n\n".join(str(f) for f in self.functions)

    def to_str(self, version: int | None = None) -> str:
        """Return the function as a string.
        The function name in the header forall functions is suffixed with
        ``_v<version>`` (e.g. ``def foo()`` becomes ``def foo_v1()``).
        """
        return "\n\n".join(f.to_str(version) for f in self.functions)

    @classmethod
    def from_code(cls, code: str) -> "Program":
        """Parse a Python program from a string."""
        functions = str_to_functions(code)
        return Program(functions=functions)


def _str_to_functions(node: ast.AST, qualname: str = "") -> Iterable[Function]:
    for child in ast.iter_child_nodes(node):
        if isinstance(child, ast.ClassDef):
            class_node: ast.ClassDef = child
            yield from _str_to_functions(child, qualname + '.' + class_node.name)
        elif isinstance(child, ast.FunctionDef):
            function_node: ast.FunctionDef = child

            args = []

            pos_args = function_node.args.args
            pos_defaults = function_node.args.defaults
            pos_default_start = len(pos_args) - len(pos_defaults)

            for i, arg in enumerate(pos_args):
              if i < pos_default_start:
                args.append(arg.arg)
              else:
                default_value = ast.unparse(pos_defaults[i - pos_default_start])
                args.append(f"{arg.arg}={default_value}")

            kwonly_args = function_node.args.kwonlyargs
            kw_defaults = function_node.args.kw_defaults

            for i, arg in enumerate(kwonly_args):
              if i < len(kw_defaults) and kw_defaults[i] is not None:
                default_value = ast.unparse(kw_defaults[i])
                args.append(f"{arg.arg}={default_value}")
              else:
                args.append(arg.arg)

            if function_node.args.vararg:
              args.append(f"*{function_node.args.vararg.arg}")
            if function_node.args.kwarg:
              args.append(f"**{function_node.args.kwarg.arg}")

            header = FuncHeader(
              name=function_node.name,
              args=args,
              return_type=(
                ast.unparse(function_node.returns) if function_node.returns else ""
              ),
            )

            decorators = function_node.decorator_list

            if decorators and isinstance(decorators[0], ast.Name):
                decorator = Decorator(decorators[0].id)
            else:
                decorator = Decorator.NONE

            function = Function(
                name=function_node.name,
                header=header,
                body=ast.unparse(function_node.body).strip(),
                path="",  # Path is not available in this context
                line_no=function_node.lineno,
                qualname=(qualname + '.' + function_node.name)[1:],
                decorator=decorator
            )

            yield function

def str_to_functions(generated_code: str) -> List[Function]:
    """Given a string with code, returns a list of Function objects."""
    tree = ast.parse(generated_code)
    return list(_str_to_functions(tree))

def str_to_function(
    generated_code: str,
) -> Function:
    """Given a string with code, returns a single Function object."""
    functions = str_to_functions(generated_code)
    if len(functions) != 1:
        raise ValueError("Expected exactly one function in the generated code.")
    return functions[0]


def str_to_program(
    generated_code: str,
) -> Program:
    """Given a string with code, returns a Program object."""
    functions = str_to_functions(generated_code)
    return Program(functions=functions)


def structured_output_to_functions(
    structured_output: ProgramImplementation,
) -> dict[FullName, Function]:
    """Given a structured output from the LLM, returns a mapping from function qualnames to Function objects."""
    functions: Dict[FullName, Function] = {}
    for structured_function in structured_output.functions:
        function = str_to_function(structured_function.code)
        function.path = structured_function.filepath
        functions[structured_function.filepath + ' ' + structured_function.qualname] = function
    return functions


def header_from_str(header_str: str) -> FuncHeader:
    """Parse a function header from a string."""
    func = str_to_function(header_str)
    return func.header

def structured_output_to_prog_meta(
    structured_output: ProgramImplementation,
    program_meta: Dict[FullName, FuncMeta],
) -> Program:
    """Given a structured output from the LLM and program metadata, returns a Program object."""

    # Check if the sample contains all the expected keys
    expected_keys = set(program_meta.keys())
    actual_keys = {f.filepath + ' ' + f.qualname for f in structured_output.functions}
    if expected_keys != actual_keys:
        raise ValueError(
            f"Expected keys {expected_keys} but got {actual_keys} in structured output."
        )
    
    functions = structured_output_to_functions(structured_output)
    
    # check if headers are same
    expected_headers = {str(header_from_str(meta.header)) for meta in program_meta.values()}
    actual_headers = {str(func.header) for func in functions.values()}
    if expected_headers != actual_headers:
        raise ValueError(
            f"Expected headers {expected_headers} but got {actual_headers} in structured output."
        )
    
    # Set the metadata for each function
    for func in functions.values():
        meta = program_meta[func.path + ' ' + func.qualname]
        func.line_no = meta.line_no
        func.qualname = meta.qualname

    return Program(functions=list(functions.values()))
  




    

  