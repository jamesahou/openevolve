"""Project constants for FunSearch."""
from funsearch.custom_types import ContainerAbsPath

HOTSWAP_ENVVAR = "FUNSEARCH_HOTSWAP_IMP"

# Absolute paths in the container file system
WORKSPACE_ROOT: ContainerAbsPath = "/workspace"
IMPLEMENTATIONS_CONTAINER_PATH: ContainerAbsPath = "/implementations"
EVAL_CONTAINER_PATH: ContainerAbsPath = "/eval.py"
MAIN_CONTAINER_PATH: ContainerAbsPath = "/main.py"