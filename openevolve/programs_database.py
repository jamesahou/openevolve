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

"""A programs database that implements the evolutionary algorithm."""

import pathlib
import pickle
from collections.abc import Mapping, Sequence
import copy
import dataclasses
import time
from typing import Any, Iterable, Tuple

from absl import logging
import numpy as np
import scipy

from openevolve import code_manipulation
from openevolve import config as config_lib
from openevolve.project_indexer import ProjectIndexer

Signature = tuple[float, ...]
ScoresPerTest = Mapping[Any, float]


def _softmax(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Returns the tempered softmax of 1D finite `logits`."""
    if not np.all(np.isfinite(logits)):
        non_finites = set(logits[~np.isfinite(logits)])
        raise ValueError(f"`logits` contains non-finite value(s): {non_finites}")
    if not np.issubdtype(logits.dtype, np.floating):
        logits = np.array(logits, dtype=np.float32)

    result = scipy.special.softmax(logits / temperature, axis=-1)
    # Ensure that probabilities sum to 1 to prevent error in `np.random.choice`.
    index = np.argmax(result)
    result[index] = 1 - np.sum(result[0:index]) - np.sum(result[index + 1 :])
    return result


def _reduce_score(scores_per_test: ScoresPerTest) -> float:
    """Reduces per-test scores into a single score."""
    return scores_per_test[list(scores_per_test.keys())[-1]]


def _get_signature(scores_per_test: ScoresPerTest) -> Signature:
    """Represents test scores as a canonical signature."""
    return tuple(scores_per_test[k] for k in sorted(scores_per_test.keys()))


# TODO: FIX
@dataclasses.dataclass(frozen=True)
class Prompt:
    """A prompt produced by the ProgramsDatabase, to be sent to Samplers.

    Attributes:
      code: The prompt, ending with the header of the function to be completed.
      version_generated: The function to be completed is `_v{version_generated}`.
      island_id: Identifier of the island that produced the implementations
         included in the prompt. Used to direct the newly generated implementation
         into the same island.
    """

    code: str
    version_generated: int
    island_id: int


class ProgramsDatabase:
    """A collection of programs, organized as islands."""

    def __init__(
        self,
        config: config_lib.ProgramsDatabaseConfig,
        template: code_manipulation.Program,
        identifier: str = "",
    ) -> None:
        self._config: config_lib.ProgramsDatabaseConfig = config
        self._template: code_manipulation.Program = template

        self._last_reset_time: float = time.time()
        self._program_counter = 0
        self._backups_done = 0
        self.identifier = identifier

        self._file_hierarchy = ProjectIndexer.get_tree_description(self._template)

        # Initialize empty islands.
        self._islands: list[Island] = []
        for _ in range(config.num_islands):
            self._islands.append(
                Island(
                    template,
                    self._file_hierarchy,
                    config.functions_per_prompt,
                    config.cluster_sampling_temperature_init,
                    config.cluster_sampling_temperature_period,
                )
            )
        self._best_score_per_island: list[float] = [-float("inf")] * config.num_islands
        self._best_program_per_island: list[code_manipulation.Program | None] = [
            None
        ] * config.num_islands
        self._best_scores_per_test_per_island: list[ScoresPerTest | None] = [
            None
        ] * config.num_islands

    def get_best_programs_per_island(
        self,
    ) -> Iterable[Tuple[code_manipulation.Program | None]]:
        return sorted(
            zip(self._best_program_per_island, self._best_score_per_island),
            key=lambda t: t[1],
            reverse=True,
        )

    def save(self, file):
        """Save database to a file"""
        data = {}
        keys = [
            "_islands",
            "_best_score_per_island",
            "_best_program_per_island",
            "_best_scores_per_test_per_island",
        ]
        for key in keys:
            data[key] = getattr(self, key)
        pickle.dump(data, file)

    def load(self, file):
        """Load previously saved database"""
        data = pickle.load(file)
        for key in data.keys():
            setattr(self, key, data[key])

    def backup(self):
        filename = f"program_db_{self.identifier}_{self._backups_done}.pickle"
        p = pathlib.Path(self._config.backup_folder)
        if not p.exists():
            p.mkdir(parents=True, exist_ok=True)
        filepath = p / filename
        logging.info(f"Saving backup to {filepath}.")

        with open(filepath, mode="wb") as f:
            self.save(f)
        self._backups_done += 1

    def get_prompt(self) -> Prompt:
        """Returns a prompt containing implementations from one chosen island."""
        island_id = np.random.randint(len(self._islands))
        code, version_generated = self._islands[island_id].get_prompt()
        return Prompt(code, version_generated, island_id)

    def _register_program_in_island(
        self,
        program: code_manipulation.Program,
        island_id: int,
        scores_per_test: ScoresPerTest,
    ) -> None:
        """Registers `program` in the specified island."""
        self._islands[island_id].register_program(program, scores_per_test)
        score = _reduce_score(scores_per_test)
        if score > self._best_score_per_island[island_id]:
            self._best_program_per_island[island_id] = program
            self._best_scores_per_test_per_island[island_id] = scores_per_test
            self._best_score_per_island[island_id] = score
            logging.info("Best score of island %d increased to %s", island_id, score)

    def register_program(
        self,
        program: code_manipulation.Program,
        island_id: int | None,
        scores_per_test: ScoresPerTest,
    ) -> None:
        """Registers `program` in the database."""
        # In an asynchronous implementation we should consider the possibility of
        # registering a program on an island that had been reset after the prompt
        # was generated. Leaving that out here for simplicity.
        if island_id is None:
            # This is a program added at the beginning, so adding it to all islands.
            for island_id in range(len(self._islands)):
                self._register_program_in_island(program, island_id, scores_per_test)
        else:
            self._register_program_in_island(program, island_id, scores_per_test)

        # Check whether it is time to reset an island.
        if time.time() - self._last_reset_time > self._config.reset_period:
            self._last_reset_time = time.time()
            self.reset_islands()

        # Backup every N iterations
        if self._program_counter > 0:
            self._program_counter += 1
            if self._program_counter > self._config.backup_period:
                self._program_counter = 0
                self.backup()

    def reset_islands(self) -> None:
        """Resets the weaker half of islands."""
        # We sort best scores after adding minor noise to break ties.
        indices_sorted_by_score: np.ndarray = np.argsort(
            self._best_score_per_island
            + np.random.randn(len(self._best_score_per_island)) * 1e-6
        )
        num_islands_to_reset = self._config.num_islands // 2
        reset_islands_ids = indices_sorted_by_score[:num_islands_to_reset]
        keep_islands_ids = indices_sorted_by_score[num_islands_to_reset:]
        for island_id in reset_islands_ids:
            self._islands[island_id] = Island(
                self._template,
                self._file_hierarchy,
                self._config.functions_per_prompt,
                self._config.cluster_sampling_temperature_init,
                self._config.cluster_sampling_temperature_period,
            )
            self._best_score_per_island[island_id] = -float("inf")
            founder_island_id = np.random.choice(keep_islands_ids)
            founder = self._best_program_per_island[founder_island_id]
            founder_scores = self._best_scores_per_test_per_island[founder_island_id]
            self._register_program_in_island(founder, island_id, founder_scores)


class Island:
    """A sub-population of the programs database."""

    def __init__(
        self,
        template: code_manipulation.Program,
        file_hierarchy: str,
        functions_per_prompt: int,
        cluster_sampling_temperature_init: float,
        cluster_sampling_temperature_period: int,
    ) -> None:
        self._template: code_manipulation.Program = template
        self._file_hierarchy = file_hierarchy
        self._functions_per_prompt: int = functions_per_prompt
        self._cluster_sampling_temperature_init = cluster_sampling_temperature_init
        self._cluster_sampling_temperature_period = cluster_sampling_temperature_period

        self._clusters: dict[Signature, Cluster] = {}
        self._num_programs: int = 0

    def register_program(
        self,
        program: code_manipulation.Program,
        scores_per_test: ScoresPerTest,
    ) -> None:
        """Stores a program on this island, in its appropriate cluster."""
        signature = _get_signature(scores_per_test)
        if signature not in self._clusters:
            score = _reduce_score(scores_per_test)
            self._clusters[signature] = Cluster(score, program)
        else:
            self._clusters[signature].register_program(program)
        self._num_programs += 1

    def get_prompt(self) -> tuple[str, int]:
        """Constructs a prompt containing functions from this island."""
        signatures = list(self._clusters.keys())
        cluster_scores = np.array(
            [self._clusters[signature].score for signature in signatures]
        )

        # Convert scores to probabilities using softmax with temperature schedule.
        period = self._cluster_sampling_temperature_period
        temperature = self._cluster_sampling_temperature_init * (
            1 - (self._num_programs % period) / period
        )
        probabilities = _softmax(cluster_scores, temperature)

        # At the beginning of an experiment when we have few clusters, place fewer
        # programs into the prompt.
        functions_per_prompt = min(len(self._clusters), self._functions_per_prompt)

        idx = np.random.choice(
            len(signatures), size=functions_per_prompt, p=probabilities
        )
        chosen_signatures = [signatures[i] for i in idx]
        implementations = []
        scores = []
        for signature in chosen_signatures:
            cluster = self._clusters[signature]
            implementations.append(cluster.sample_program())
            scores.append(cluster.score)

        indices = np.argsort(scores)
        sorted_implementations = [implementations[i] for i in indices]
        version_generated = len(sorted_implementations) + 1
        return self._generate_prompt(sorted_implementations), version_generated

    def _generate_prompt(
        self, implementations: Sequence[code_manipulation.Program]
    ) -> str:
        implementations = copy.deepcopy(implementations)
        prompt = f"# File Hierarchy \n{self._file_hierarchy}\n\n"
        for program_version, implementation in enumerate(implementations):
            prompt += f"# Start of Program  Version {program_version} (*_v{program_version})\n"
            if program_version >= 1:
                prompt += f"# This is an improved version of the previous program (*_v{program_version - 1})\n\n"

            for function in implementation.functions:
                path = function.path
                func_loc_comment = f"#{path}: {function.qualname} ({function.decorator if function.decorator else ''}\n)"
                prompt += f"{func_loc_comment}\n"

                func_str = function.to_str(version=program_version)
                prompt += f"{func_str}\n\n"

            prompt += f"# End of Program Version {program_version}\n\n"

        return prompt


class Cluster:
    """A cluster of programs on the same island and with the same Signature."""

    def __init__(self, score: float, implementation: code_manipulation.Program):
        self._score = score
        self._programs: list[code_manipulation.Program] = [implementation]
        self._lengths: list[int] = [len(str(implementation))]

    @property
    def score(self) -> float:
        """Reduced score of the signature that this cluster represents."""
        return self._score

    def register_program(self, program: code_manipulation.Program) -> None:
        """Adds `program` to the cluster."""
        self._programs.append(program)
        self._lengths.append(len(str(program)))

    def sample_program(self) -> code_manipulation.Program:
        """Samples a program, giving higher probability to shorther programs."""
        normalized_lengths = (np.array(self._lengths) - min(self._lengths)) / (
            max(self._lengths) + 1e-6
        )
        probabilities = _softmax(-normalized_lengths, temperature=1.0)
        return np.random.choice(self._programs, p=probabilities)