from openevolve.sandbox import ContainerSandbox, ContainerEngine, DummySandbox
from openevolve.constants import SANDBOX_IMAGE_NAME, SANDBOX_CONTAINER_NAME
from openevolve.custom_types import HostAbsPath, HostRelPath

from pathlib import Path

import tempfile
import pathlib
import pytest
import types
import os

os.environ["PATH"] = "/opt/homebrew/bin/:" + os.environ["PATH"]

@pytest.fixture(autouse=True)
def reset_sandbox_id():
    DummySandbox.sandboxes = 0
    yield
    DummySandbox.sandboxes = 0

def test_builds_image():
    # Get the path to the project root
    project_root: HostAbsPath = Path("/Users/ryanrudes/openevolve/astropy")

    # Get the relative path to the evaluation script
    eval_path: HostRelPath = Path("./eval.py")

    # Get the relative path to the setup.sh file
    setup_path: HostRelPath = Path("./setup.sh")

    ContainerSandbox.build_image(
        project_root=project_root,
        eval_relpath=eval_path,
        setup_relpath=setup_path,
    )

def test_creates_sandbox():
    # # Create temporary directory to store implementations
    # imps_dir = tempfile.TemporaryDirectory()
    # imps_path: HostAbsPath = Path(imps_dir.name)
    raise NotImplementedError("This test is not implemented yet.")