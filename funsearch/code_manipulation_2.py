import dataclasses
import io
import ast
import textwrap
from typing import List, Tuple, Sequence, Any, Dict, Iterable
import re
from enum import Enum


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

            header = FuncHeader(
                name=function_node.name,
                args=[arg.arg for arg in function_node.args.args],
                return_type=(
                    ast.unparse(function_node.returns) if function_node.returns else ""
                ),
            )

            decorators = function_node.decorator_list

            if decorators:
                top_decorator: ast.Name = decorators[0]
                decorator = Decorator(top_decorator.id)
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
    structured_output: Dict[str, str],
) -> dict[str, Function]:
    """Given a structured output from the LLM, returns a list of Function objects."""
    functions: Dict[str, Function] = {}
    for func_name, body in structured_output.items():
        functions[func_name] = str_to_function(body)
    return functions


def header_from_str(header_str: str) -> FuncHeader:
    """Parse a function header from a string."""
    header_str = header_str.strip()
    match = re.match(r"def (\w+)\((.*?)\)(?: -> (\w+))?:", header_str)
    if not match:
        raise ValueError(f"Invalid function header: {header_str}")

    name = match.group(1)
    args = [arg.strip() for arg in match.group(2).split(",") if arg.strip()]
    return_type = match.group(3) or ""

    return FuncHeader(name=name, args=args, return_type=return_type)