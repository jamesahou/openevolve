import json
import logging
import os
import pathlib
import pickle
import time
import cloudpickle

import click
from openai import OpenAI

from dotenv import load_dotenv
from typing import List
from funsearch.test_case import TestCase

import tempfile

from funsearch.custom_types import (
    HostAbsPath,
    HostRelPath,
    ContainerAbsPath,
    ContainerRelPath
)
# from funsearch import config, core, sandbox, sampler, programs_database, code_manipulation, evaluator, extractor
from funsearch import config, core, sandbox, sampler, programs_database_2, code_manipulation_2, evaluator2, extractor

LOGLEVEL = os.environ.get('LOGLEVEL', 'INFO').upper()
logging.basicConfig(level=LOGLEVEL)


def get_all_subclasses(cls):
  all_subclasses = []

  for subclass in cls.__subclasses__():
    all_subclasses.append(subclass)
    all_subclasses.extend(get_all_subclasses(subclass))

  return all_subclasses


def parse_input(filename_or_data: str):
  if len(filename_or_data) == 0:
    raise Exception("No input data specified")
  p = pathlib.Path(filename_or_data)
  if p.exists():
    if p.name.endswith(".json"):
      return json.load(open(filename_or_data, "r"))
    if p.name.endswith(".pickle"):
      return pickle.load(open(filename_or_data, "rb"))
    raise Exception("Unknown file format or filename")
  if "," not in filename_or_data:
    data = [filename_or_data]
  else:
    data = filename_or_data.split(",")
  if data[0].isnumeric():
    f = int if data[0].isdecimal() else float
    data = [f(v) for v in data]
  return data

@click.group()
@click.pass_context
def main(ctx):
  pass


@main.command()
@click.argument("project_root", type=click.Path(file_okay=False))
@click.argument("setup_file", type=click.Path(file_okay=True))
@click.argument("eval_file", type=click.Path(file_okay=True))
@click.argument("tests_file", type=click.Path(file_okay=True))
@click.option("--evolve_depth", default=-1, type=click.INT, help='How many levels to evolve the program')
@click.option('--model_name', default="gpt-3.5-turbo-instruct", help='LLM model')
@click.option('--output_path', default="./data/", type=click.Path(file_okay=False), help='path for logs and data')
@click.option('--load_backup', default=None, type=click.File("rb"), help='Use existing program database')
@click.option('--iterations', default=-1, type=click.INT, help='Max iterations per sampler')
@click.option('--samplers', default=15, type=click.INT, help='Samplers')
def run(project_root, setup_file, eval_file, tests_file, evolve_depth, model_name, output_path, load_backup, iterations, samplers):
  timestamp = str(int(time.time()))
  log_path: HostAbsPath = pathlib.Path(output_path) / timestamp
  if not log_path.exists():
    log_path.mkdir(parents=True)
    logging.info(f"Writing logs to {log_path}")
  
  project_root: HostAbsPath = pathlib.Path(project_root)
  setup_file: HostAbsPath = pathlib.Path(setup_file)
  eval_file: HostRelPath = pathlib.Path(eval_file)  # relative to project_root
  tests_file: HostAbsPath = pathlib.Path(tests_file)
  
  assert(project_root.is_absolute())
  assert(setup_file.is_absolute())
  assert(not eval_file.is_absolute())
  assert(tests_file.is_absolute())

  tests: List[TestCase] = cloudpickle.load(open(tests_file, "rb"))
  
  imps_path = tempfile.mkdtemp(suffix=f"_impementations_{timestamp}")

  eval_file = pathlib.Path(eval_file)
  initial_program, _, program_meta = extractor.extract_code(project_root, eval_file, tests)
  exit()
  template = code_manipulation_2.structured_output_to_prog_meta(initial_program, program_meta)

  extractor.add_decorators(program_meta)
  try:
    conf = config.Config(num_evaluators=1)
    database = programs_database_2.ProgramsDatabase(
      conf.programs_database, template, worskpace=workspace, identifier=timestamp)
    if load_backup:
      database.load(load_backup)

    sbox = sandbox.ContainerSandbox(workspace, eval_file, imps_path, setup_file=setup_file)
    evaluators = [evaluator2.AsyncEvaluator(
      database,
      sbox,
      template,
      workspace,
      eval_file,
      imps_path,
      program_meta,
      inputs,
    ) for _ in range(conf.num_evaluators)]
    # We send the initial implementation to be analysed by one of the evaluators.
    evaluators[0].analyse(initial_program, island_id=None, implementation_id="-1")
    assert len(database._islands[0]._clusters) > 0, ("Initial analysis failed. Make sure that Sandbox works! "
                                                    "See e.g. the error files under sandbox data.")

    lm = sampler.vLLM(samples_per_prompt=2, url="https://generativelanguage.googleapis.com/v1beta/openai/", model="gemini-2.0-flash-lite", log_path=log_path)

    samplers = [sampler.Sampler(database, evaluators, lm, uid=i)
                for i in range(samplers)]

    core.run(samplers, database, iterations)
  finally:
    extractor.remove_decorators(program_meta)


@main.command()
@click.argument("db_file", type=click.File("rb"))
def ls(db_file):
  """List programs from a stored database (usually in data/backups/ )"""
  conf = config.Config(num_evaluators=1)

  # A bit silly way to list programs. This probably does not work if config has changed any way
  database = programs_database_2.ProgramsDatabase(
    conf.programs_database, None, identifier="")
  database.load(db_file)

  progs = database.get_best_programs_per_island()
  print("Found {len(progs)} programs")
  for i, (prog, score) in enumerate(progs):
    print(f"{i}: Program with score {score}")
    print(prog)
    print("\n")


if __name__ == '__main__':
  main()
