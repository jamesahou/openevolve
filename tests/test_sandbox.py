from openevolve.sandbox import ContainerSandbox, ContainerEngine, DummySandbox
from openevolve.constants import SANDBOX_IMAGE_NAME, SANDBOX_CONTAINER_NAME
from openevolve.custom_types import HostAbsPath, HostRelPath
from openevolve.test_case import TestCase

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

def test_build_image():
    # Get the path to the project root
    project_root: HostAbsPath = Path("/Users/ryanrudes/GitHub/OpenEvolve/astropy")

    # Get the relative path to the evaluation script
    eval_path: HostRelPath = Path("./eval.py")

    # Get the relative path to the setup.sh file
    setup_path: HostRelPath = Path("./setup.sh")

    ContainerSandbox.build_image(
        project_root=project_root,
        eval_relpath=eval_path,
        setup_relpath=setup_path,
    )

def test_create_sandbox():
    imps_root: HostAbsPath = Path("/Users/ryanrudes/GitHub/OpenEvolve/openevolve/imps")

    ContainerSandbox.create_container(imps_root)

def test_start_sandbox():
    ContainerSandbox.start_container()

def test_upload_test_cases():
    dummy_test_cases = [
        TestCase(
            args = [1, 2],
            kwargs= {}
        ),
        TestCase(
            args = [3, 4],
            kwargs= {}
        ),
    ]

    ContainerSandbox.upload_test_cases(dummy_test_cases)

def test_remove_container():
    ContainerSandbox.remove_container()