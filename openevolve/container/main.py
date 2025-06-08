"""This file will be used as an executable script by the ContainerSandbox and ExternalProcessSandbox."""

from openevolve.code_manipulation import yield_decorated

from importlib.util import spec_from_file_location, module_from_spec

import concurrent.futures
import logging
import pickle
import sys
import os

def main(eval_file: str, input_file: str, output_file: str, timeout: str):
    """The method takes executable function as a cloudpickle file, then executes it with input data,
    and writes the output data to another file."""
    logging.debug(f"Running main(): {eval_file}, {input_file}, {output_file}, {timeout}")
    
    # Create the output directory if it does not exist
    # /outputs/{implementation_id}
    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)

    timeout = float(timeout)

    with open(input_file, "rb") as file:
        input_data_dict = pickle.load(file)
    
    args = input_data_dict["args"]
    kwargs = input_data_dict["kwargs"]

    # Dynamically import the eval module
    spec = spec_from_file_location("eval", eval_file)
    eval_mod = module_from_spec(spec)
    spec.loader.exec_module(eval_mod)

    # Read the eval file source code
    with open(eval_file, "r") as f:
        eval_source = f.read()

    # Find the function decorated with @funsearch.run
    run_functions = list(yield_decorated(eval_source, "funsearch", "run"))
    if len(run_functions) != 1:
        raise ValueError("Expected exactly one function decorated with @funsearch.run")
    func_name = run_functions[0]
    func = getattr(eval_mod, func_name)

    # Run the eval function on the test case input
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args, **kwargs)

        try:
            eval_output = future.result(timeout=timeout)
            json_dump = {
                "eval_output": eval_output,
                "timeout": False
            }
        except concurrent.futures.TimeoutError:
            logging.warning(f"Function execution exceeded timeout of {timeout} seconds")
            json_dump = {
                "eval_output": None,
                "timeout": True
            }

    # Make the directory for the output file if it does not exist
    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)

    # Save the test case output to the container file system
    with open(output_file, "wb") as file:
        pickle.dump(json_dump, file)

if __name__ == "__main__":
    if len(sys.argv) != 5:
        sys.exit(-1)
    
    args = sys.argv[1:]
    main(*args)