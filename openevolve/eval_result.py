from dataclasses import dataclass
from typing import Any

@dataclass
class EvalResult:
    """A class to represent the result of an evaluation."""
    timeout: bool
    success: bool
    output: Any