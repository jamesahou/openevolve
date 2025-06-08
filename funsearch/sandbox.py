from funsearch.test_case import TestCase
from funsearch.constants import HOTSWAP_ENVVAR

from enum import StrEnum
from typing import Any

import cloudpickle as pickle
import tempfile
import logging
import pathlib
import shutil
import sys
import ast
import os

CONTAINER_MAIN = "/main.py"

IMAGE_NAME = "funsearch_sandbox"
CONTAINER_NAME = "funsearch_container"

class ContainerEngine(StrEnum):
    """Enum for container engines."""
    PODMAN = "podman"
    DOCKER = "docker"

DEFAULT_CONTAINER_ENGINE = ContainerEngine.DOCKER

class ContainerSandbox:
    """
    Basic sandbox that runs unsafe code in Podman or Docker container.
    - the sandbox should be safe against inadvertent bad code by LLM but not against malicious attacks.
    - does not require any other dependencies on the host than Podman/Docker
    - does not support multithreading
    - might provide easier or more lightweight debugging experience than some other fancier sandbox environments
    """
    engine: ContainerEngine = DEFAULT_CONTAINER_ENGINE
    built = False

    @staticmethod
    def has_engine(engine: ContainerEngine) -> bool:
        """Checks if the specified container engine is available."""
        ret = os.system(f"{engine.value} --version")
        return ret == 0

    @classmethod
    def use_engine(cls, engine: ContainerEngine):
        """Sets the container engine to the specified one."""
        cls.engine = engine

    @classmethod
    @property
    def executable(cls) -> str:
        """Returns the executable command for the container engine."""
        return cls.engine.value

    @classmethod
    def select_engine(cls):
        """Selects the container engine to use."""
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

    @classmethod
    def build_image(cls, workspace_path: pathlib.Path, implementations_path: pathlib.Path, eval_file: pathlib.Path, setup_file: pathlib.Path | None = None):
        """Builds the container image."""
        version = sys.version.split(" ")[0]
        logging.debug(f"Using Python version: {version}")

        # Select which container engine to use
        cls.select_engine()

        # Define the path to the Dockerfile
        dockerfile = pathlib.Path(__file__).parent / "container" / "Dockerfile"
        logging.debug("Building container image")

        # Add the setup file as a build argument if provided
        extra = ""

        if setup_file is not None:
            extra = f'--build-arg SETUP_FILE="{setup_file}"'

        # Set the build context to the root directory of the OpenEvolve project
        build_context = dockerfile.parent.parent.parent

        # Prepare the command to build the container image
        cmd = (
            # Use the container engine to build the image
            f"{cls.executable} build "
            # Set the build argument for the Python version
            f"--build-arg PYTHON_VERSION={version} "
            # Set the build argument for the workspace root
            f"--build-arg WORKSPACE_ROOT={workspace_path} "
            # Set the build argument for the evaluation entry point
            f"--build-arg EVAL_FILE={eval_file} "
            # Tag the image with the name
            f"-t {IMAGE_NAME} "
            # Use the Dockerfile from the container directory
            f"-f {dockerfile} {build_context} "
            # Add any extra build arguments, such as the setup file
            f"{extra}"
        )

        # Execute the command to build the image
        logging.debug(f"Executing: {cmd}")
        os.system(cmd)

        # Create the container
        logging.debug("Creating container from the built image...")
        
        cmd = (
            f"{cls.executable} create "
            # Set the container to run in interactive mode
            f"-i "
            # Set the container name
            f"--name {CONTAINER_NAME} "
            # Mount the implementations directory from the host to /implementations in the container
            f"--mount type=bind,source={implementations_path},target=/implementations,readonly "
            # Use the built image
            f"{IMAGE_NAME}:latest"
        )

        logging.debug(f"Executing: {cmd}")
        os.system(cmd)

        # Complete the image build process
        logging.debug("Container image built successfully.")
        cls.built = True

    @classmethod
    def upload_test_cases(cls, test_cases: list[TestCase]):
        """
        Uploads test cases to the container sandbox.

        Args:
            test_cases (list[TestCase]): List of test cases to upload.
        """
        # Create a temporary directory on the host file system
        # to store the test cases.
        temp_dir = tempfile.mkdtemp()

        try:
            # Copy the test case files to the temporary directory
            for i, test_case in enumerate(test_cases):
                # Create a unique filename for each test case
                test_case_file = os.path.join(temp_dir, f"{i}.pickle")

                # Write the test case to a file using cloudpickle
                with open(test_case_file, "wb") as file:
                    test_case_dict = {
                        "args": test_case.args,
                        "kwargs": test_case.kwargs,
                    }

                    pickle.dump(test_case_dict, file)

            # Use the container engine to copy files to the container
            cmd = (
                f"{cls.executable} cp {temp_dir}/. "
                f"{CONTAINER_NAME}:/inputs"
            )

            logging.debug(f"Copying test cases to container: {cmd}")
            os.system(cmd)
        except Exception as e:
            logging.error(f"Failed to upload test cases: {e}")
            raise
        finally:
            # Remove the temporary directory after copying
            shutil.rmtree(temp_dir)

    def __init__(
        self,
        workspace_path: pathlib.Path,
        eval_file: pathlib.Path,
        implementations_path: pathlib.Path,
        python_path: str = "/usr/local/bin/python3",
        setup_file: pathlib.Path | None = None,
    ):
        """
        Initializes the container sandbox.

        Args:
            workspace_path (pathlib.Path): The absolute path to the modified workspace root on the host.
            eval_file (pathlib.Path): The path to the evaluation entry point Python file on the host.
            implementations_path (pathlib.Path): The path to the implementations directory on the host.
            python_path (str): The path to the Python interpreter to use. Defaults to "python".
            setup_file (pathlib.Path | None): The path to the setup file for installation, located on the host. This will be copied to the container.
        """
        super().__init__()

        self.workspace_path = workspace_path
        self.python_path = python_path
        self.implementations_path = implementations_path
        self.eval_file = eval_file
        self.setup_file = setup_file

        # Check if setup_file is a shell script
        if setup_file is not None and setup_file.suffix != ".sh":
            raise ValueError(
                f"setup_file must be a shell script, got {setup_file.suffix}"
            )

        # Check that the evaluation entry point is a Python file
        if not eval_file.is_file() or eval_file.suffix != ".py":
            raise ValueError(
                f"eval_file must be a Python file, got {eval_file.suffix}"
            )

        """
        # Build the container image if it has not been built yet
        if ContainerSandbox.built:
            logging.warning("Container image already built! Skipping build.")
        else:
            ContainerSandbox.build_image(workspace_path, implementations_path, eval_file, setup_file)
        """

        ContainerSandbox.start_container()

    @classmethod
    def start_container(cls):
        """
        Starts the container if it is not already running.
        If the container is already running, it will not do anything.
        """
        cmd = (
            f"{cls.executable} start "
            f"{CONTAINER_NAME}"
        )

        logging.debug(f"Executing: {cmd}")
        os.system(cmd)

    def execute(
        self,
        implementation_id: str,
        test_id: int,
        timeout: float = 30.0,
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
            # Use the container engine to run the command
            f"{self.executable} exec "
            # Set the container to run on
            f"{CONTAINER_NAME} "
            f"/bin/bash -c '"
            # Set the environment variable for hot-swapping
            f"{HOTSWAP_ENVVAR}={implementation_id} "
            # Call the Python interpreter in the container
            f"{self.python_path} "
            # Execute the main Python script in the container
            f"{CONTAINER_MAIN} "
            # Pass the paths to the eval program, input file, output file, and log directory
            f"/eval.py "
            f"/inputs/{test_id}.pickle "
            f"/outputs/{implementation_id}/output_{test_id}.pickle "
            # Pass the timeout to the main script
            f"{timeout} "
            # Pipe the standard output to a log file
            f"> /logs/{implementation_id}_test_{test_id}_stdout.txt "
            # Pipe the standard error output to a log file
            f"2> /logs/{implementation_id}_test_{test_id}_stderr.txt' "
        )
        logging.debug(f"Executing: {cmd}")
        return os.system(cmd)

    def run(
        self,
        implementation_id: str,
        test_id: int,
        timeout: float = 30.0,
    ) -> Any:
        """
        Runs the container sandbox with the specified entry point and implementation ID.

        Args:
            implementation_id (str): The ID of the implementation to run.
            test_id (int): The ID of the test case to execute.
            timeout (float): The maximum time in seconds to allow for the function execution.

        Returns:
            Any: The returned output from the evaluator function, run on the test case.
        """
        self.execute(implementation_id, test_id, timeout)

        # Create a temporary file to store the output
        output_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pickle")
        
        # Copy the output file from the container to the host
        cmd = (
            f"{self.executable} cp "
            f"{CONTAINER_NAME}:/outputs/{implementation_id}/output_{test_id}.pickle "
            f"{output_file.name}"
        )
        logging.debug(f"Copying output file from container: {cmd}")
        os.system(cmd)

        # Load the output data from the temporary file
        with open(output_file.name, "rb") as file:
            output_data = pickle.load(file)

        # Clean up the temporary file
        os.remove(output_file.name)

        return output_data

def main(workspace_root, implementations_root):
    ImplementationsManager.set_workspace_root(workspace_root)
    ImplementationsManager.set_implementations_root(implementations_root)
    ImplementationsManager.set_program_meta(program_meta)

    implementation_manager = ImplementationsManager()
    implementation_manager.save_implementation(initial_program, "random")

    # Example usage of the ContainerSandbox
    sandbox = ContainerSandbox(
        workspace_path=pathlib.Path("./astropy"),
        eval_file=pathlib.Path("./astropy/eval.py"),
        implementations_path=implementations_root,
        setup_file=pathlib.Path("./examples/astropy_example/setup.sh")
    )

    sandbox.upload_test_cases(test_cases)

    sandbox.execute("random", 0)

if __name__ == "__main__":
    # Set log level to DEBUG for detailed output
    logging.basicConfig(level=logging.DEBUG)

    import funsearch.extractor as extractor
    from funsearch.evaluator2 import ImplementationsManager
    from pathlib import Path

    workspace_root = Path("/Users/ryanrudes/openevolve/astropy")
    implementations_root = pathlib.Path("/Users/ryanrudes/openevolve/implementations")

    test_cases = [
        TestCase(args=[], kwargs={}),
        TestCase(args=[], kwargs={}),
    ]

    """
    initial_program, evolve_path, program_meta = extractor.extract_code(Path("/Users/ryanrudes/openevolve/astropy/eval.py"), test_cases)

    with open("initial_program.pickle", "wb") as f:
        pickle.dump(initial_program, f)

    with open("program_meta.pickle", "wb") as f:
        pickle.dump(program_meta, f)

    exit()
    """

    with open("initial_program.pickle", "rb") as f:
        initial_program = pickle.load(f)

    with open("program_meta.pickle", "rb") as f:
        program_meta = pickle.load(f)

    extractor.add_decorators(program_meta)

    try:
        main(workspace_root, implementations_root)
    except:
        pass
    finally:
        extractor.remove_decorators(program_meta)