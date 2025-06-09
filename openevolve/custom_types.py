from dataclasses import dataclass
from typing import TypeAlias, Any
from pathlib import Path

# An annotation for full names (relpath + space + qualname)
FullName: TypeAlias = str

# Path types
AbsPath: TypeAlias = Path          # This is an absolute path on some file system
RelPath: TypeAlias = Path          # This is a relative path inside the project on some file system
HostAbsPath: TypeAlias = Path      # This is an absolute path on the host (e.g. "/home/user/project/.flake8")
HostRelPath: TypeAlias = Path      # This is a relative path inside the project on the host, excluding the workspace root (e.g. "./.flake8")
ContainerAbsPath: TypeAlias = Path # This is an absolute path in the container, starting with "/", e.g. "/workspace/.flake8"
ContainerRelPath: TypeAlias = Path # This is a relative path inside the project in the container, excluding the workspace root (e.g. "./.flake8")

@dataclass
class FuncMeta:
    """Metadata for a function"""
    file_path: RelPath
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