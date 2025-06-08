from typing import TypeAlias, Any
import dataclasses
from pathlib import Path

# An annotation for full names (abspath + space + qualname)
FullName: TypeAlias = str

@dataclasses.dataclass
class FuncMeta:
    """
    Metadata for a function
    
    """

    file_path: Path
    qualname: str
    line_no: int
    class_name: str | None
    header: str | None

def from_dict(data: dict[str, Any]) -> FuncMeta:
    return FuncMeta(
        file_path=Path(data["file_path"]),
        qualname=data["qualname"],
        line_no=data["line_no"],
        class_name=data.get("class_name", None),
        header=data.get("header", None)
    )
