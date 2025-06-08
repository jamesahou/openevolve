"""This file will be used as an executable script by the ContainerSandbox and ExternalProcessSandbox."""

from funsearch.code_manipulation import yield_decorated

from importlib.util import spec_from_file_location, module_from_spec

import logging
import pickle
import sys
import os

def main(eval_file: str, input_file: str, output_file: str):
    """The method takes executable function as a cloudpickle file, then executes it with input data,
    and writes the output data to another file."""
    logging.debug(f"Running main(): {eval_file}, {input_file}, {output_file}")

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
    ret = func(*args, **kwargs)

    # Make the directory for the output file if it does not exist
    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)

    # Save the test case output to the container file system
    with open(output_file, "wb") as file:
        pickle.dump(ret, file)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(-1)
    
    args = sys.argv[1:]
    main(*args)