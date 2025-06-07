import dataclasses
import io
import ast
import textwrap
from typing import List, Tuple, Sequence, Any, Dict
import re

@dataclasses.dataclass
class FuncHeader:
  """A parsed Python function header."""
  
  name: str
  args: List[str]
  return_type: str = ''

  def __str__(self) -> str:
    """Return the function header as a string."""
    args = sorted(self.args, key=lambda x: (x != 'self', x))
    args = ', '.join(args)
    if self.return_type:
      return f'def {self.name}({args}) -> {self.return_type}:'
    else:
      return f'def {self.name}({args}):'
    
  def __hash__(self):
    return hash(str(self))

@dataclasses.dataclass
class Function:
  """A parsed Python function."""

  name: str
  header: FuncHeader
  body: str
  class_name: str = ''
  path: str = ''
  line_no: int = 0
  qual_name: str = ''

  def __str__(self) -> str:
    header_str = str(self.header).strip()
    body = self.body.strip()
    if not body.endswith('\n'):
      body += '\n'
    return f'{header_str}\n{body}'
  
  def to_str(self, version: int | None = None) -> str:
    """Return the function as a string.
    If *version* is given, the function name in the header is suffixed with
    ``_v<version>`` (e.g. ``def foo()`` → ``def foo_v1()``).  Otherwise this is
    identical to ``str(self)``.
    """
    if version is None:
      return str(self)
    header_copy = FuncHeader(
            name=f'{self.header.name}_v{version}',
            args=list(self.header.args),
            return_type=self.header.return_type
        )
    body = self.body.strip()
    return f'{header_copy}\n{body}'

  def __setattr__(self, name: str, value: str) -> None:
    # Ensure there aren't leading & trailing new lines in `body`.
    if name == 'body':
      value = value.strip('\n')
    super().__setattr__(name, value)


@dataclasses.dataclass(frozen=True)
class Program:
  """A parsed Python program."""

  functions: list[Function]

  def __str__(self) -> str:
    return '\n\n'.join(str(f) for f in self.functions)

  def to_str(self, version: int | None = None) -> str:
    """Return the function as a string.
    The function name in the header forall functions is suffixed with
    ``_v<version>`` (e.g. ``def foo()`` becomes ``def foo_v1()``). 
    """
    return '\n\n'.join(f.to_str(version) for f in self.functions)

def str_to_functions(generated_code: str) -> List[Function]:
    """Given a string with code, returns a list of Function objects."""
    tree = ast.parse(generated_code)
    lines = generated_code.splitlines()
    functions: List[Function] = []

    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue

        start_idx = node.lineno - 1
        header_parts: List[str] = []
        for i in range(start_idx, len(lines)):
            part = lines[i].strip()
            header_parts.append(part)
            if part.endswith(":"):
                break
        else:
            raise ValueError(f"Could not locate end of definition for function {node.name!r}")

        header_line = " ".join(header_parts)
        header = header_from_str(header_line)

        body_start = node.body[0].lineno - 1 if node.body else start_idx + len(header_parts)
        end = node.end_lineno
        body_lines = lines[body_start:end]
        body = "\n".join(body_lines).rstrip("\n")

        functions.append(Function(name=node.name, header=header, body=body))

    return functions

def str_to_function(
    generated_code: str,
) -> Function:
  """Given a string with code, returns a single Function object."""
  functions = str_to_functions(generated_code)
  if len(functions) != 1:
    raise ValueError("Expected exactly one function in the generated code.")
  return functions[0]

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
  match = re.match(r'def (\w+)\((.*?)\)(?: -> (\w+))?:', header_str)
  if not match:
    raise ValueError(f"Invalid function header: {header_str}")
  
  name = match.group(1)
  args = [arg.strip() for arg in match.group(2).split(',') if arg.strip()]
  return_type = match.group(3) or ''
  
  return FuncHeader(name=name, args=args, return_type=return_type)  
