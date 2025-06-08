from funsearch.test_case import TestCase

from enum import StrEnum

import logging

import ast
import os
import pathlib
import sys
from typing import Any

import cloudpickle

CONTAINER_MAIN = (
    pathlib.Path(__file__).parent / "container" / "container_main.py"
).absolute()

IMAGE_NAME = "funsearch_sandbox"


class DummySandbox:
    """
    Base class for Sandboxes that execute the generated code.

    Note: this base class executes the code but does not offer any sandboxing!!!
    It should be only used in unit testing or debugging, and not with real LLM
    unless the host environment is in some kind of sandbox itself.
    Even in sandboxed host, the executed code could theoretically affect later executions.
    """

    sandboxes = 0

    def __init__(self, **kwargs):
        self.id = self.register_sandbox()

    def register_sandbox(self) -> int:
        """Register this sandbox in the global list of sandboxes."""
        sandbox_id = DummySandbox.sandboxes
        DummySandbox.sandboxes += 1
        return sandbox_id

    def run(
        self,
        code: str,
        function_name: str,
        test_case: TestCase,
        timeout: float = 30.0,
    ) -> tuple[Any, bool]:
        """
        Executes a specified function from dynamically compiled code with given arguments.

        Args:
            code (str): The source code containing the function to execute.
            function_name (str): The name of the function to invoke from the compiled code.
            test_case (TestCase): The test case containing input arguments and keyword arguments.
            timeout (float): The maximum time in seconds to allow for the function execution.

        Returns:
            tuple[Any, bool]: The result of the function execution and a boolean flag indicating success.

        Raises:
            KeyError: If the specified function is not found in the compiled namespace.
            Exception: Propagates any exception raised during function execution.
        """

        # The same "program" seems to be now repeatedly parsed using AST and then compiled.
        # This could probably be simplified quite a bit.
        namespace = DummySandbox.compile(code)
        args = test_case.args
        kwargs = test_case.kwargs
        return namespace[function_name](*args, **kwargs)

    @staticmethod
    def compile(code: str):
        """
        Compiles and executes the given Python code string in a new namespace.

        Args:
            code (str): The Python source code to compile and execute.

        Returns:
            dict: The namespace dictionary containing the variables and functions defined by the executed code.
        """
        namespace = {}

        parsed_code = ast.parse(code)
        compiled_code = compile(parsed_code, filename="<ast>", mode="exec")
        exec(compiled_code, namespace)
        return namespace

### TODO: EVERYTHING BELOW THIS

class ExternalProcessSandbox(DummySandbox):
    """
    Sandbox that executes the code in a separate Python process in the same host.

    Note: This does not provide real safety and should be only used in an environment where the host process is
    in some kind of safe sandbox itself (i.e., a container).
    This kind of sandbox merely makes it more probable that single invalid call does not break the whole
    funsearch algorithm. It might be easier to set up and thus nice environment to tune the prompts and other code.
    """

    def __init__(
        self,
        base_path: pathlib.Path,
        python_path: str = "python",
        timeout: float = 30.0,
    ):
        super().__init__()

        self.output_path = pathlib.Path(base_path) / f"sandbox{self.id}"
        self.python_path = python_path
        self.timeout = timeout
        self.call_count = 0

        self.input_path = self.output_path / "inputs"
        for p in [self.output_path, self.input_path]:
            if not p.exists():
                p.mkdir(parents=True)

    def _exec(
        self,
        call_data_path: pathlib.Path,
        input_path: pathlib.Path,
        error_file_path: pathlib.Path,
    ):
        """Use podman/docker to execute python in a container.
        - The main.py shall execute the LLM generated method from prog.pickle file providing
          input.pickle as the input for the method.
        - main.py writes the output of the method into output.pickle.
        Everything except the /workspace folder will be read-only so that the environment remains good
        for future runs.
        """
        prog_path = call_data_path / "prog.pickle"
        output_file = call_data_path / "output.pickle"
        cmd = (
            f"{self.python_path} {CONTAINER_MAIN} {prog_path} {input_path} {output_file}"
            f"  2> {error_file_path}"
        )
        logging.debug(f"Executing: {cmd}")
        return os.system(cmd)

    def run(
        self,
        program: str,
        function_to_run: str,
        test_input,
        timeout_seconds: int,
    ) -> tuple[Any, bool]:
        call_data_folder = (self.output_path / f"call{self.call_count}").absolute()
        if not call_data_folder.exists():
            call_data_folder.mkdir(exist_ok=True)

        input_hash = hash(test_input)
        input_path = (self.input_path / f"{input_hash}.pickle").absolute()

        if not input_path.exists():
            with open(input_path, "wb") as f:
                cloudpickle.dump(test_input, f)
        try:
            namespace = DummySandbox.compile_code(program)

            prog_file = (call_data_folder / f"prog.pickle").absolute()
            with open(prog_file, "wb+") as f:
                cloudpickle.dump(namespace[function_to_run], f)

            error_file = self.output_path / f"stderr_{self.call_count}.log"

            retcode = self._exec(call_data_folder, input_path, error_file)
            self.call_count += 1

            if retcode != 0:
                self._save_diagnostics(program, call_data_folder)
                return None, False

            output_file = call_data_folder / f"output.pickle"
            with open(output_file, "rb") as f:
                out = cloudpickle.load(f)
                return out, True
        except Exception as e:
            logging.debug(f"Could not execute code: {e}")
        self._save_diagnostics(program, call_data_folder)
        return None, False

    @staticmethod
    def _save_diagnostics(program: str, output_path: pathlib.Path):
        filepath = output_path / "program.py"
        logging.debug(f"Writing program to {filepath}")
        with open(filepath, "w+") as f:
            f.write(program)


class ContainerEngine(StrEnum):
    """Enum for container engines."""
    PODMAN = "podman"
    DOCKER = "docker"

DEFAULT_CONTAINER_ENGINE = ContainerEngine.PODMAN

class ContainerSandbox(ExternalProcessSandbox):
    """
    Basic sandbox that runs unsafe code in Podman or Docker container.
    - the sandbox should be safe against inadvertent bad code by LLM but not against malicious attacks.
    - does not require any other dependencies on the host than Podman/Docker
    - does not support multithreading
    - might provide easier or more lightweight debugging experience than some other fancier sandbox environments
    """
    engine: ContainerEngine = DEFAULT_CONTAINER_ENGINE
    image_built = False

    @classmethod
    def has_engine(cls, engine: ContainerEngine) -> bool:
        """Checks if the specified container engine is available."""
        ret = os.system(f"{engine.value} --version")
        return ret == 0

    @classmethod
    def use_engine(cls, engine: ContainerEngine):
        """Sets the container engine to the specified one."""
        cls.engine = engine

    @classmethod
    def build_image(cls, extra_pip_packages):
        """Builds the container image."""
        version = sys.version.split(" ")[0]
        logging.debug(f"Using Python version: {version}")

        # Select which container engine to use
        logging.debug("Checking for Podman...")

        if not cls.has_engine(ContainerEngine.PODMAN):
            logging.debug("Podman not found, checking for Docker...")

            if cls.has_engine(ContainerEngine.DOCKER):
                logging.debug("Docker found, using it instead of Podman.")
                cls.use_engine(ContainerEngine.DOCKER)
            else:
                raise Exception(
                    "Could not find Podman or Docker. Can not use ContainerSandbox."
                )

        # Build the container image
        dockerfile = pathlib.Path(__file__).parent / "container" / "Dockerfile"
        logging.debug("Building container image")
        extra = ""
        if extra_pip_packages:
            extra = f'--build-arg INSTALL_PACKAGES="{extra_pip_packages}"'

        cmd = (
            f"{cls.engine} build --build-arg PYTHON_VERSION={version} {extra} "
            f"-t {IMAGE_NAME} -f {dockerfile} {CONTAINER_MAIN.parent}"
        )
        logging.debug(f"Executing: {cmd}")
        os.system(cmd)

        # TODO: Add the decorators to the relevant functions in the project directory
        logging.debug("Adding decorators to the project directory.")

        # Complete the image build process
        logging.debug("Container image built successfully.")
        cls.image_built = True

    def __init__(
        self,
        base_path: pathlib.Path,
        python_path: str = "python",
        extra_pip_packages: str = "numpy",
        timeout: float = 30.0,
    ):
        super().__init__(base_path, python_path, timeout)

        if not ContainerSandbox.image_built:
            ContainerSandbox.build_image(extra_pip_packages)

    def _exec(
        self,
        call_data_path: pathlib.Path,
        input_path: pathlib.Path,
        error_file_path: pathlib.Path,
    ):
        """
        Use podman/docker to execute python in a container.
        - The main.py shall execute the LLM generated method from prog.pickle file providing
          input.pickle as the input for the method.
        - main.py writes the output of the method into output.pickle.
        Everything except the /workspace folder will be read-only so that the environment remains good
        for future runs.
        """
        cmd = (
            f"{self.executable} run "
            f"--stop-timeout={self.timeout_secs} "
            f"-v {CONTAINER_MAIN}:/main.py:ro "
            f"-v {call_data_path}:/workspace "
            f"-v {input_path}:/input.pickle:ro "
            f"{IMAGE_NAME}:latest /usr/local/bin/python3 "
            f"/main.py /workspace/prog.pickle /input.pickle /workspace/output.pickle"
            f"  2> {error_file_path}"
        )
        logging.debug(f"Executing: {cmd}")
        return os.system(cmd)