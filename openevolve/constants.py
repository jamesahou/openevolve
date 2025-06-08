"""Project constants for FunSearch."""
from openevolve.custom_types import ContainerAbsPath

HOTSWAP_ENVVAR = "FUNSEARCH_HOTSWAP_IMP"

# Absolute paths in the container file system
WORKSPACE_ROOT: ContainerAbsPath = "/workspace"
IMPS_CONTAINER_PATH: ContainerAbsPath = "/imps"
LOGS_CONTAINER_PATH: ContainerAbsPath = "/logs"
INPUTS_CONTAINER_PATH: ContainerAbsPath = "/inputs"
EVAL_CONTAINER_PATH: ContainerAbsPath = "/eval.py"
MAIN_CONTAINER_PATH: ContainerAbsPath = "/main.py"
CONTAINER_PYTHONPATH: ContainerAbsPath = "/usr/local/bin/python3"

# Container constants
SANDBOX_IMAGE_NAME = "funsearch_image"
SANDBOX_CONTAINER_NAME = "funsearch_sandbox"