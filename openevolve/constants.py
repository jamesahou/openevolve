"""Project constants for FunSearch."""
from openevolve.custom_types import ContainerAbsPath
from pathlib import Path

HOTSWAP_ENVVAR = "FUNSEARCH_HOTSWAP_IMP"

# Absolute paths in the container file system
WORKSPACE_ROOT: ContainerAbsPath = Path("/workspace")
IMPS_CONTAINER_PATH: ContainerAbsPath = Path("/imps")
LOGS_CONTAINER_PATH: ContainerAbsPath = Path("/logs")
INPUTS_CONTAINER_PATH: ContainerAbsPath = Path("/inputs")
EVAL_CONTAINER_PATH: ContainerAbsPath = WORKSPACE_ROOT / "eval.py"
SETUP_CONTAINER_PATH: ContainerAbsPath = WORKSPACE_ROOT / "setup.py"
MAIN_CONTAINER_PATH: ContainerAbsPath = "/main.py"
CONTAINER_PYTHONPATH: ContainerAbsPath = "/usr/local/bin/python3"

# Container constants
SANDBOX_IMAGE_NAME = "funsearch_image"
SANDBOX_CONTAINER_NAME = "funsearch_sandbox"