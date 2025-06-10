import json
import logging
import os
import pathlib
import pickle
import time
import cloudpickle

import click
from openai import OpenAI
import shutil
from dotenv import load_dotenv
from typing import List
from openevolve.test_case import TestCase
from openevolve.evaluator import ImplementationsManager
from pathlib import Path
import tempfile

from openevolve.custom_types import (
    HostAbsPath,
    HostRelPath,
    ContainerAbsPath,
    ContainerRelPath,
)
from openevolve import (
    code_manipulation,
    config,
    core,
    evaluator,
    programs_database,
    sandbox,
    sampler,
    extractor,
)

LOGLEVEL = os.environ.get("LOGLEVEL", "INFO").upper()
logging.basicConfig(level=LOGLEVEL)


@click.group()
@click.pass_context
def main(ctx):
    pass


@main.command()
@click.argument("project_root", type=click.Path(file_okay=False))
@click.argument("setup_file", type=click.Path(file_okay=True))
@click.argument("eval_file", type=click.Path(file_okay=True))
@click.argument("tests_file", type=click.Path(file_okay=True))
@click.option(
    "--evolve_depth",
    default=-1,
    type=click.INT,
    help="How many levels to evolve the program",
)
@click.option("--model_name", default="gpt-3.5-turbo-instruct", help="LLM model")
@click.option(
    "--output_path",
    default="./data/",
    type=click.Path(file_okay=False),
    help="path for logs and data",
)
@click.option(
    "--load_backup",
    default=None,
    type=click.File("rb"),
    help="Use existing program database",
)
@click.option(
    "--iterations", default=-1, type=click.INT, help="Max iterations per sampler"
)
@click.option("--samplers", default=15, type=click.INT, help="Samplers")
def run(
    project_root,
    setup_file,
    eval_file,
    tests_file,
    evolve_depth,
    model_name,
    output_path,
    load_backup,
    iterations,
    samplers,
):
    timestamp = str(int(time.time()))
    log_path: HostAbsPath = pathlib.Path(output_path) / timestamp
    if not log_path.exists():
        log_path.mkdir(parents=True)
        logging.info(f"Writing logs to {log_path}")

    project_root: HostAbsPath = pathlib.Path(project_root)
    setup_file: HostRelPath = pathlib.Path(setup_file)
    eval_file: HostRelPath = pathlib.Path(eval_file)  # relative to project_root
    tests_file: HostAbsPath = pathlib.Path(tests_file)
    openevolve_path: HostAbsPath = pathlib.Path(__file__).parent.parent

    assert project_root.is_absolute()
    assert not setup_file.is_absolute()
    assert not eval_file.is_absolute()
    assert tests_file.is_absolute()

    tests: List[TestCase] = cloudpickle.load(open(tests_file, "rb"))
    assert len(tests) > 0, "No tests found in the provided tests file."

    imps_path = Path(tempfile.mkdtemp(suffix=f"_impementations_{timestamp}"))
    logging.info(f"Using temporary directory for implementations: {imps_path}")

    eval_file = pathlib.Path(eval_file)

    ex: extractor.Extractor = extractor.Extractor(project_root, eval_file)

    #initial_program, path, program_meta = ex.run(tests[0], evolve_depth)
    #if path is None:
    #    raise ValueError(
    #        "No initial program found. Make sure that the eval_file is correct and contains a valid program."
    #    )

    # Load initial_program and program_meta from the cache directoy
    cache_dir = pathlib.Path(__file__).parent.parent / "cache"

    with open(cache_dir / "spec_structured.pickle", "rb") as f:
        initial_program: extractor.ProgramImplementation = cloudpickle.load(f)
    with open(cache_dir / "program_meta.pickle", "rb") as f:
        program_meta: dict[str, extractor.FuncMeta] = cloudpickle.load(f)

    ImplementationsManager.set_workspace_root(project_root)
    ImplementationsManager.set_implementations_root(imps_path)
    ImplementationsManager.set_program_meta(program_meta)

    template = code_manipulation.structured_output_to_prog_meta(
        initial_program, program_meta
    )
    ex.add_decorators(program_meta)

    try:
        conf = config.Config(num_evaluators=1)
        database = programs_database.ProgramsDatabase(
            conf.programs_database, template, identifier=timestamp
        )
        if load_backup:
            database.load(load_backup)

        sbox = sandbox.ContainerSandbox(
            project_root, imps_path, eval_file, setup_file, openevolve_path = openevolve_path,# force_rebuild_container = True
        )
        evaluators = [
            evaluator.Evaluator(
                database,
                sbox,
                tests,
            )
            for _ in range(conf.num_evaluators)
        ]
        # We send the initial implementation to be analysed by one of the evaluators.
        evaluators[0].analyse(initial_program, island_id=None, implementation_id="initial")
        assert len(database._islands[0]._clusters) > 0, (
            "Initial analysis failed. Make sure that Sandbox works! "
            "See e.g. the error files under sandbox data."
        )

        lm = sampler.vLLM(
            samples_per_prompt=2,
            url="https://generativelanguage.googleapis.com/v1beta/openai/",
            model="gemini-2.0-flash-lite",
            log_path=log_path,
        )

        samplers = [
            sampler.Sampler(database, evaluators, lm, uid=i) for i in range(samplers)
        ]

        core.run(samplers, database, iterations)
    finally:
        ex.remove_decorators(program_meta)
        #if os.path.exists(imps_path):
        #    shutil.rmtree(imps_path)


if __name__ == "__main__":
    main()
