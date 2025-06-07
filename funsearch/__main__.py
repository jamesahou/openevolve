import json
import logging
import os
import pathlib
import pickle
import time

import click
import llm
from openai import OpenAI

from dotenv import load_dotenv


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
@click.argument("eval_file", type=click.Path(file_okay=True))
@click.argument('inputs')
@click.option("--evolve_depth", default=-1, type=click.INT, help='How many levels to evolve the program')
@click.option('--model_name', default="gpt-3.5-turbo-instruct", help='LLM model')
@click.option('--output_path', default="./data/", type=click.Path(file_okay=False), help='path for logs and data')
@click.option('--load_backup', default=None, type=click.File("rb"), help='Use existing program database')
@click.option('--iterations', default=-1, type=click.INT, help='Max iterations per sampler')
@click.option('--samplers', default=15, type=click.INT, help='Samplers')
@click.option('--sandbox_type', default="ContainerSandbox", type=click.Choice(SANDBOX_NAMES), help='Sandbox type')
def run(workspace, eval_file, inputs, evolve_depth, model_name, output_path, load_backup, iterations, samplers, sandbox_type):
  timestamp = str(int(time.time()))
  log_path = pathlib.Path(output_path) / timestamp
  if not log_path.exists():
    log_path.mkdir(parents=True)
    logging.info(f"Writing logs to {log_path}")

  lm = sampler.vLLM(2, "token-abc123", "http://0.0.0.0:11440/v1/", 
                       "meta-llama/Llama-3.3-70B-Instruct", log_path)
  workspace = pathlib.Path(workspace)
  imps_path = pathlib.Path(f"imps_{timestamp}")
  if not imps_path.exists():
    imps_path.mkdir(parents=True)
    logging.info(f"Writing implementations to {imps_path}")
  
  eval_file = pathlib.Path(eval_file)
  initial_program, evolve_path, program_meta = extractor.extract_code(eval_file, inputs)

  template = code_manipulation_2.str_to_program(initial_program)

  conf = config.Config(num_evaluators=1)
  database = programs_database_2.ProgramsDatabase(
    conf.programs_database, template, identifier=timestamp)
  if load_backup:
    database.load(load_backup)

  inputs = parse_input(inputs)

  sandbox_class = next(c for c in SANDBOX_TYPES if c.__name__ == sandbox_type)
  evaluators = [evaluator2.Evaluator(
    database,
    sandbox_class(base_path=log_path),
    template,
    workspace,
    eval_file,
    imps_path,
    program_meta,
    inputs,
  ) for _ in range(conf.num_evaluators)]
  # We send the initial implementation to be analysed by one of the evaluators.
  evaluators[0].analyse(initial_program, island_id=None, curr_id="-1")
  assert len(database._islands[0]._clusters) > 0, ("Initial analysis failed. Make sure that Sandbox works! "
                                                   "See e.g. the error files under sandbox data.")

  samplers = [sampler.Sampler(database, evaluators, lm, uid=i)
              for i in range(samplers)]

  core.run(samplers, database, iterations)


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
