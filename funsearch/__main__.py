import json
import logging
import os
import pathlib
import pickle
import time
import cloudpickle

import click
import llm
from openai import OpenAI

from dotenv import load_dotenv
from typing import List
from funsearch.test_case import TestCase


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


SANDBOX_TYPES = get_all_subclasses(sandbox.DummySandbox) + [sandbox.DummySandbox]
SANDBOX_NAMES = [c.__name__ for c in SANDBOX_TYPES]


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
@click.argument("workspace", type=click.Path(file_okay=False))
@click.argument("setup_file", type=click.Path(file_okay=True))
@click.argument("eval_file", type=click.Path(file_okay=True))
@click.argument("inputs", type=click.Path(file_okay=True))
@click.option("--evolve_depth", default=-1, type=click.INT, help='How many levels to evolve the program')
@click.option('--model_name', default="gpt-3.5-turbo-instruct", help='LLM model')
@click.option('--output_path', default="./data/", type=click.Path(file_okay=False), help='path for logs and data')
@click.option('--load_backup', default=None, type=click.File("rb"), help='Use existing program database')
@click.option('--iterations', default=-1, type=click.INT, help='Max iterations per sampler')
@click.option('--samplers', default=15, type=click.INT, help='Samplers')
@click.option('--sandbox_type', default="ContainerSandbox", type=click.Choice(SANDBOX_NAMES), help='Sandbox type')
def run(workspace, setup_file, eval_file, inputs, evolve_depth, model_name, output_path, load_backup, iterations, samplers, sandbox_type):
  timestamp = str(int(time.time()))
  log_path = pathlib.Path(output_path) / timestamp
  if not log_path.exists():
    log_path.mkdir(parents=True)
    logging.info(f"Writing logs to {log_path}")
  
  logging.info("Loading test cases from input file...")
  inputs: List[TestCase] = cloudpickle.load(open(inputs, "rb"))
  workspace = pathlib.Path(workspace)
  imps_path = pathlib.Path(f"imps_{timestamp}")
  if not imps_path.exists():
    imps_path.mkdir(parents=True)
    logging.info(f"Writing implementations to {imps_path}")
  setup_file = pathlib.Path(setup_file) if setup_file else None

  logging.info("Initializing language model and extractor...")
  lm = sampler.vLLM(samples_per_prompt=2, url="https://generativelanguage.googleapis.com/v1beta/openai/", model="gemini-2.0-flash-lite", log_path=log_path)

  eval_file = pathlib.Path(eval_file)
  logging.info(f"Extracting code from {eval_file} ...")
  initial_program, evolve_path, program_meta = extractor.extract_code(eval_file, inputs)

  logging.info("Converting structured output to program meta...")
  template = code_manipulation_2.structured_output_to_prog_meta(initial_program, program_meta)

  logging.info("Adding decorators to extracted functions...")
  extractor.add_decorators(program_meta)
  try:
    logging.info("Creating config and initializing program database...")
    conf = config.Config(num_evaluators=1)
    database = programs_database_2.ProgramsDatabase(
      conf.programs_database, template, worskpace=workspace, identifier=timestamp)
    if load_backup:
      logging.info("Loading backup database...")
      database.load(load_backup)

    # sandbox_class = next(c for c in SANDBOX_TYPES if c.__name__ == sandbox_type)
    logging.info("Setting up sandbox and evaluators...")
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
    logging.info("Analyzing initial program implementation...")
    evaluators[0].analyse(initial_program, island_id=None, implementation_id="-1")
    assert len(database._islands[0]._clusters) > 0, ("Initial analysis failed. Make sure that Sandbox works! "
                                                    "See e.g. the error files under sandbox data.")

    logging.info(f"Starting {samplers} samplers and running core loop for {iterations if iterations > 0 else 'unlimited'} iterations...")
    samplers = [sampler.Sampler(database, evaluators, lm, uid=i)
                for i in range(samplers)]

    core.run(samplers, database, iterations)
    logging.info("Run completed successfully.")
  finally:
    logging.info("Removing decorators from extracted functions...")
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
