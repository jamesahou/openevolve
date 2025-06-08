"""Project constants for FunSearch."""
from funsearch.custom_types import ContainerAbsPath

HOTSWAP_ENVVAR = "FUNSEARCH_HOTSWAP_IMP"

# Absolute paths in the container file system
WORKSPACE_ROOT: ContainerAbsPath = "/workspace"
IMPS_CONTAINER_PATH: ContainerAbsPath = "/imps"
LOGS_CONTAINER_PATH: ContainerAbsPath = "/logs"
EVAL_CONTAINER_PATH: ContainerAbsPath = "/eval.py"
MAIN_CONTAINER_PATH: ContainerAbsPath = "/main.py"

# Container constants
SANDBOX_IMAGE_NAME = "funsearch_image"
SANDBOX_CONTAINER_NAME = "funsearch_sandbox"