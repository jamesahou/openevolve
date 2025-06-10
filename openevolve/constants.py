"""Project constants for OpenEvolve."""
from openevolve.custom_types import ContainerAbsPath
from pathlib import Path

LIBRARY_NAME = "openevolve"

HOTSWAP_ENVVAR = "OPENEVOLE_HOTSWAP_IMP"

# Absolute paths in the container file system
WORKSPACE_ROOT: ContainerAbsPath = Path("/home/workspace")
CONTAINER_IMPS_PATH: ContainerAbsPath = Path("/home/imps")
CONTAINER_LOGS_PATH: ContainerAbsPath = Path("/home/logs")
CONTAINER_INPUTS_PATH: ContainerAbsPath = Path("/home/inputs")
CONTAINER_OUTPUTS_PATH: ContainerAbsPath = Path("/home/outputs")
CONTAINER_EVAL_PATH: ContainerAbsPath = Path("/home/eval.py")
CONTAINER_SETUP_PATH: ContainerAbsPath = Path("/home/setup.sh")
CONTAINER_MAIN_PATH: ContainerAbsPath = Path("/home/main.py")
CONTAINER_OPENEVOLVE_PATH: ContainerAbsPath = Path("/home/openevolve")
CONTAINER_OPENEVOLVE_EDITABLE_PATH: ContainerAbsPath = Path("/home/openevolve_editable")
CONTAINER_PYTHONPATH: ContainerAbsPath = Path("/usr/local/bin/python3")

# Container constants
SANDBOX_IMAGE_NAME = "openevolve_image"
SANDBOX_CONTAINER_NAME = "openevolve_sandbox"