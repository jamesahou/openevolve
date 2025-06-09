from openai import OpenAI
import pathlib

from funsearch import extractor, project_indexer, evaluator2, code_manipulation_2
from funsearch.code_manipulation_2 import Function
from funsearch.structured_outputs import ProgramImplementation
import textwrap

eval_path = pathlib.Path("examples/astropy_example/repo/astropy/eval.py")
args = [0]
base_dir = pathlib.Path("/Users/jameshou/Documents/Code/openevolve/examples/astropy_example/repo/astropy")

spec_structured, path, program_meta = extractor.extract_code(eval_path, args)

program = code_manipulation_2.structured_output_to_prog_meta(spec_structured, program_meta)
file_hierarchy = project_indexer.ProjectIndexer.get_tree_description(program, base_dir)

def build_prompt_from_spec_structured(program, file_hierarchy: str = "# File Hierarchy\n") -> str:
    prompt = f"{file_hierarchy}\n"
    prompt += "# Start of Program Version 0 (*_v0)\n"
    for function in program.functions:
        # You may want to add more details here if available in your FunctionImplementation
        func_loc_comment = f"#{pathlib.Path(function.path).relative_to(base_dir)}: {function.qualname}\n"
        prompt += f"{func_loc_comment}\n"
        prompt += f"{function.to_str(version=0)}\n\n"
    prompt += "# End of Program Version 0\n\n"
    return prompt

prompt = build_prompt_from_spec_structured(program, file_hierarchy)
print(prompt)

client = OpenAI(
    api_key="###########",
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

completion = client.beta.chat.completions.parse(
    model="gemini-2.0-flash-lite",
    messages=[
        {"role": "system", "content": "You are an expert iterative code-evolution assistant. For each provided function analyse the existing version(s) and improve them, optimize for clarity, performance, and correctness while preserving the existing codebase structure. NEVER CHANGE THE FUNCTION NAME OR HEADER."},
        {"role": "user", "content": prompt},
    ],
    response_format=ProgramImplementation
)

results = completion.choices[0].message.parsed