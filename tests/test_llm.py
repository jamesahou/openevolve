from openevolve import project_indexer, evaluator, code_manipulation

from openevolve.project_indexer import ProjectIndexer
from openevolve.prompting import build_prompt
from openevolve.extractor import Extractor

from openevolve.code_manipulation import Function
from openevolve.structured_outputs import ProgramImplementation
from openevolve.custom_types import HostAbsPath, HostRelPath, FuncMeta
from openevolve.test_case import TestCase

from openai import OpenAI
from pathlib import Path

import cloudpickle as pickle
import json
import sys
import os

OPENEVOLVE_ROOT = Path(__file__).parent.parent
sys.path.insert(0, OPENEVOLVE_ROOT)

CACHE_DIR = Path(__file__).parent.parent / "cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def test_extractor():
    project_root: HostAbsPath = Path("/Users/ryanrudes/GitHub/OpenEvolve/circle_packing")
    eval_path: HostRelPath = Path("./eval.py")

    test_case = TestCase(
        args = [],
        kwargs = {},
    )

    extactor = Extractor(
        base_dir=project_root,
        eval_path=eval_path,
    )

    spec_structured, path, program_meta = extactor.run(test_case)

    with open(CACHE_DIR / "circle_packing_spec_structured.pickle", "wb") as f:
        pickle.dump(spec_structured, f)

    with open(CACHE_DIR / "circle_packing_program_meta.pickle", "wb") as f:
        pickle.dump(program_meta, f)

def test_call_tree():
    with open(CACHE_DIR / "circle_packing_spec_structured.pickle", "rb") as f:
        spec_structured: ProgramImplementation = pickle.load(f)

    with open(CACHE_DIR / "circle_packing_program_meta.pickle", "rb") as f:
        program_meta: dict[str, FuncMeta] = pickle.load(f)

    program = code_manipulation.structured_output_to_prog_meta(spec_structured, program_meta, 0)
    file_hierarchy = ProjectIndexer.get_tree_description(program)

    with open(CACHE_DIR / "circle_packing_program.pickle", "wb") as f:
        pickle.dump(program, f)

    with open(CACHE_DIR / "circle_packing_file_hierarchy.txt", "w") as f:
        f.write(file_hierarchy)

    print()
    print("=== FILE HIERARCHY ===")
    print(file_hierarchy)
    print("======================")
    
def test_prompt_builder():
    with open(CACHE_DIR / "circle_packing_program.pickle", "rb") as f:
        program: ProgramImplementation = pickle.load(f)

    file_hierarchy = ProjectIndexer.get_tree_description(program)

    prompt = build_prompt(
        programs = [program],
        file_hierarchy = file_hierarchy,
        num_versions = 1,
        extra_prompt=None,
    )

    print("======= PROMPT =======")
    print(prompt)
    print("======================")

    with open(CACHE_DIR / "circle_packing_prompt.txt", "w") as f:
        f.write(prompt)

def test_prompt_builder_with_task_prompt():
    with open(CACHE_DIR / "circle_packing_program.pickle", "rb") as f:
        program: ProgramImplementation = pickle.load(f)

    extra_prompt_path = OPENEVOLVE_ROOT / "examples" / "circle_packing" / "prompt.txt"

    with open(extra_prompt_path, "r") as f:
        extra_prompt = f.read()

    file_hierarchy = ProjectIndexer.get_tree_description(program)

    prompt = build_prompt(
        programs = [program],
        file_hierarchy = file_hierarchy,
        num_versions = 1,
        extra_prompt = extra_prompt,
    )

    print("======= PROMPT =======")
    print(prompt)
    print("======================")

    with open(CACHE_DIR / "circle_packing_prompt_extended.txt", "w") as f:
        f.write(prompt)
        
def test_get_response():
    from openevolve.prompting.prompt import ROLE_PROMPT

    with open(CACHE_DIR / "prompt.txt", "r") as f:
        prompt = f.read()

    api_key = os.getenv("GEMINI_API_KEY")

    client = OpenAI(
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        api_key=api_key,
    )

    model = "gemini-2.0-flash"
    messages = [
        {"role": "system", "content": ROLE_PROMPT},
        {"role": "user", "content": prompt},
    ]
    response_format = ProgramImplementation

    completion = client.beta.chat.completions.parse(
        model=model,
        messages=messages,
        response_format=response_format
    )

    results = completion.choices[0].message.parsed

    # Cache the response
    api_call = {
        "model": model,
        "messages": messages,
        "response_format": response_format,
        "response": results,
    }

    # Ensure the api cache directory exists
    api_cache_dir = CACHE_DIR / "api"
    os.makedirs(api_cache_dir, exist_ok=True)

    # Find the next available index
    existing_files = [f for f in os.listdir(api_cache_dir) if f.startswith("response_") and f.endswith(".pickle")]
    indices = []
    for fname in existing_files:
        try:
            idx = int(fname[len("response_"):-len(".pickle")])
            indices.append(idx)
        except ValueError:
            continue
    next_index = max(indices, default=0) + 1

    # Save the response with the next index
    with open(api_cache_dir / f"response_{next_index}.pickle", "wb") as file:
        pickle.dump(api_call, file)

def test_analyze_response():
    with open(CACHE_DIR / "api" / "response_9.pickle", "rb") as f:
        api_call = pickle.load(f)

    response: ProgramImplementation = api_call["response"]

    print("=== ANALYZING RESPONSE ===")
    for function in response.functions:
        print(f"Filepath: {function.filepath}")
        print(f"Qualified Name: {function.qualname}")
        print(f"Code:\n{function.code}")
        print("--------------------------")
        print()