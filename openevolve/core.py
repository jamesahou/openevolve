# Copyright 2023 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A single-threaded implementation of the FunSearch pipeline."""
from concurrent.futures import ThreadPoolExecutor

import logging
import time

def run(samplers, database, iterations: int = -1):
    """Launches a FunSearch experiment.

    Args:
        samplers: List of sampler objects, each with a .sample() method.
        database: Database object with a .backup() method.
        iterations: Number of iterations to run. If -1, runs indefinitely.
    """
    counter = 0

    try:
        with ThreadPoolExecutor(max_workers=len(samplers)) as executor:
            while iterations != 0:
                t0 = time.time()

                futures = [
                    executor.submit(s.sample)
                    for s in samplers
                ]

                for future in futures:
                    future.result()

                t1 = time.time()
                logging.info(
                    f"Iteration: {counter}, Time taken: {t1 - t0:.3f} seconds"
                )
                counter += 1

                if iterations > 0:
                    iterations -= 1

    except KeyboardInterrupt:
        logging.info("Keyboard interrupt. Stopping.")

    database.backup()