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

"""Class for sampling new programs."""
from collections.abc import Collection, Sequence

import llm
import numpy as np
import time
import logging
from openai import OpenAI

from funsearch import evaluator2
from funsearch import programs_database


class LLM:
  """Language model that predicts continuation of provided source code."""

  def __init__(self, samples_per_prompt: int, model: llm.Model, log_path=None) -> None:
    self._samples_per_prompt = samples_per_prompt
    self.model = model
    self.prompt_count = 0
    self.log_path = log_path

  def _draw_sample(self, prompt: str) -> str:
    """Returns a predicted continuation of `prompt`."""
    response = self.model.prompt(prompt)
    self._log(prompt, response, self.prompt_count)
    self.prompt_count += 1
    return response

  def draw_samples(self, prompt: str) -> Collection[str]:
    """Returns multiple predicted continuations of `prompt`."""
    return [self._draw_sample(prompt) for _ in range(self._samples_per_prompt)]

  def _log(self, prompt: str, response: str, index: int):
    if self.log_path is not None:
      with open(self.log_path / f"prompt_{index}.log", "a") as f:
        f.write(prompt)
      with open(self.log_path / f"response_{index}.log", "a") as f:
        f.write(str(response))

class vLLM:
  def __init__(self, samples_per_prompt: int, key: str, url: str, model: str, log_path=None):
    self.client = OpenAI(api_key=key, base_url=url)
    self.samples_per_prompt = samples_per_prompt
    self.log_path = log_path
    self.model = model
    self.prompt_count = 0

  def draw_sample(self, prompt: str) -> str:
    """Draws a single sample from the LLM."""
    response = self.client.chat.completions.create(
      model=self.model,
      messages=[{"role": "user", "content": prompt}],
      max_tokens=4000
    )
    return response.choices[0].message.content
  
  def draw_samples(self, prompt: str) -> Collection[str]:
    """Batch processes multiple prompts at once."""
    responses = self.client.chat.completions.create(
      model=self.model,
      messages=[{"role": "user", "content": prompt} for _ in range(self.samples_per_prompt)],
      max_tokens=4000
    )
    responses =[r.message.content for r in responses.choices]

    for response in responses:
      self._log(prompt, response, self.prompt_count)
      self.prompt_count += 1
    
    return responses
  
  def _log(self, prompt: str, response: str, index: int):
    if self.log_path is not None:
      with open(self.log_path / f"prompt_{index}.log", "a") as f:
        f.write(prompt)
      with open(self.log_path / f"response_{index}.log", "a") as f:
        f.write(str(response))
  
class Sampler:
  """Node that samples program continuations and sends them for analysis."""

  def __init__(
      self,
      database: programs_database.ProgramsDatabase,
      evaluators: Sequence[evaluator2.Evaluator],
      model: LLM | vLLM,
      uid: int = 0,
  ) -> None:
    self._database = database
    self._evaluators = evaluators
    self._llm = model
    self.uid = uid
    self.generation_number = 0  

  def sample(self):
    """Continuously gets prompts, samples programs, sends them for analysis."""
    # Time getting prompt
    t0 = time.time()
    prompt = self._database.get_prompt()
    t1 = time.time()
    prompt_time = t1 - t0
    
    # Time LLM sampling
    t0 = time.time() 
    samples = self._llm.draw_samples(prompt.code)
    t1 = time.time()
    llm_time = t1 - t0

    # Time evaluation
    eval_times = []
    for sample in samples:
        curr_id = str(self.uid) + "_" + str(self.generation_number)
        self.generation_number += 1
        t0 = time.time()
        chosen_evaluator = np.random.choice(self._evaluators)
        chosen_evaluator.analyse(
            sample, prompt.island_id, curr_id)
        t1 = time.time()
        eval_times.append(t1 - t0)
    
    # Log timing results
    avg_eval_time = sum(eval_times) / len(eval_times)

