from openai import OpenAI
import pathlib

from funsearch import extractor, project_indexer, evaluator2

eval_path = pathlib.Path("examples/astropy_example/eval.py")
args = [0]

spec_structured, path, loc_dict = extractor.extract_code(eval_path, args)

program = evaluator2._save_sample(
    spec_struct)

file_hierarchy = project_indexer.ProjectIndexer.get_tree_description()

def build_prompt_from_spec_structured(spec_structured, file_hierarchy: str = "# File Hierarchy\n") -> str:
    prompt = f"{file_hierarchy}\n"
    prompt += "# Start of Program Version 0 (*_v0)\n"
    for function in spec_structured.functions:
        # You may want to add more details here if available in your FunctionImplementation
        func_loc_comment = f"#{getattr(function, 'filepath', '')}: {getattr(function, 'qualname', '')}\n"
        prompt += f"{func_loc_comment}\n"
        prompt += f"{function.code}\n\n"
    prompt += "# End of Program Version 0\n\n"
    return prompt

prompt = build_prompt_from_spec_structured(spec_structured)

client = OpenAI(
    api_key="GEMINI_API_KEY",
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

completion = client.beta.chat.completions.parse(
    model="gemini-2.0-flash",
    messages=[
        {"role": "system", "content": "Extract the event information."},
        {"role": "user", "content": "John and Susan are going to an AI conference on Friday."},
    ],
    response_format=CalendarEvent,
)

print(completion.choices[0].message.parsed)