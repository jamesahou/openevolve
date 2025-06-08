from openai import OpenAI
import pathlib

from funsearch import extractor, project_indexer, evaluator2, code_manipulation_2

eval_path = pathlib.Path("examples/astropy_example/eval.py")
args = [0]

spec_structured, path, program_meta = extractor.extract_code(eval_path, args)

program = code_manipulation_2.structured_output_to_prog_meta(spec_structured, program_meta)

file_hierarchy = project_indexer.ProjectIndexer.get_tree_description(program)

def build_prompt_from_spec_structured(program, file_hierarchy: str = "# File Hierarchy\n") -> str:
    prompt = f"{file_hierarchy}\n"
    prompt += "# Start of Program Version 0 (*_v0)\n"
    for function in program.functions:
        # You may want to add more details here if available in your FunctionImplementation
        func_loc_comment = f"#{getattr(function, 'filepath', '')}: {getattr(function, 'qualname', '')}\n"
        prompt += f"{func_loc_comment}\n"
        prompt += f"{function.to_str(version=0)}\n\n"
    prompt += "# End of Program Version 0\n\n"
    return prompt

prompt = build_prompt_from_spec_structured(program)
print(prompt)
client = OpenAI(
    api_key="AIzaSyCR8oHYcm21ntZFVob5Ch751g5XRR43HAg",
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

completion = client.beta.chat.completions.parse(
    model="gemini-2.0-flash-lite",
    messages=[
        {"role": "system", "content": "Extract the event information."},
        {"role": "user", "content": "John and Susan are going to an AI conference on Friday."},
    ],
    response_format=CalendarEvent,
)

print(completion.choices[0].message.parsed)