from dataclasses import dataclass
from typing import Any

@dataclass
class TestCase:
    """
    Represents a test case for the function search.
    
    Attributes:
        args (list[Any]): A list of arguments to be passed to the function.
        kwargs (dict[str, Any]): A dictionary of keyword arguments to be passed to the function.
    """
    args: list[Any]
    kwargs: dict[str, Any]