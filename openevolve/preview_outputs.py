from openevolve.constants import CONTAINER_OUTPUTS_PATH

import cloudpickle as pickle
import os

for impl_id in os.listdir(CONTAINER_OUTPUTS_PATH):
    impl_path = os.path.join(CONTAINER_OUTPUTS_PATH, impl_id)
    if not os.path.isdir(impl_path):
        continue

    for output_file in os.listdir(impl_path):
        output_path = os.path.join(impl_path, output_file)
        if not output_file.endswith(".pickle"):
            continue

        with open(output_path, "rb") as f:
            try:
                data = pickle.load(f)
                print(f"Implementation: {impl_id}, Output File: {output_file}, Data: {data}")
            except Exception as e:
                print(f"Failed to load {output_path}: {e}")