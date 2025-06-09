"""Project constants for FunSearch."""
from openevolve.custom_types import ContainerAbsPath
from pathlib import Path

HOTSWAP_ENVVAR = "FUNSEARCH_HOTSWAP_IMP"

# Absolute paths in the container file system
WORKSPACE_ROOT: ContainerAbsPath = Path("/workspace")
CONTAINER_IMPS_PATH: ContainerAbsPath = Path("/imps")
CONTAINER_LOGS_PATH: ContainerAbsPath = Path("/logs")
CONTAINER_INPUTS_PATH: ContainerAbsPath = Path("/inputs")
CONTAINER_OUTPUTS_PATH: ContainerAbsPath = Path("/outputs")
CONTAINER_EVAL_PATH: ContainerAbsPath = Path("/eval.py")
CONTAINER_SETUP_PATH: ContainerAbsPath = Path("/setup.sh")
CONTAINER_MAIN_PATH: ContainerAbsPath = Path("/main.py")
CONTAINER_PYTHONPATH: ContainerAbsPath = Path("/usr/local/bin/python3")

# Container constants
SANDBOX_IMAGE_NAME = "funsearch_image"
SANDBOX_CONTAINER_NAME = "funsearch_sandbox"