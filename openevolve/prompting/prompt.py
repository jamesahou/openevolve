from openevolve.project_indexer import ProjectIndexer
from openevolve.code_manipulation import Program

from pathlib import Path
import ast

PROMPTS_DIR = Path(__file__).parent / "prompts"

"""
### EXAMPLE ###
```
# foo/bar.py: Bar.baz
def baz_v0():
    print("Naive baz implementation")

# foo/bar.py: Bar.baz
def baz_v1():
    print("Improved baz implementation")

...

# foo/bar.py Bar.baz
def baz_v2():
    print("Awesome baz implementation!")

### SPECIFYING LOCATIONS ###
* file_path: used to specify the location of a file within the project (e.g. `project/foo/bar.py`)
* qualified_name: used to specify the location of a function within the file it appears in (e.g. BarParent.BarClass.bar_method_v0)
* version_number: used to specify the version of the function (e.g. v0, v1, etc.)

You should use the full name to refer to the functions you implement. Use the format below to specify the location of a function in your response:
```
{{file_path}}: {{qualified_name}}_v{{version_number}}
```

For example, if you want to refer to version 2 of the function `bar_method` implemented in the class `BarClass` in the file `project/foo/bar.py`, which inherits from `BarParent`, you would write:
```
project/foo/bar.py: BarParent.BarClass.bar_method_v2
```
```
"""

PROMPT_HEADER_TEMPLATE = """
### ROLE ###
{role}

### GUIDELINES ###
{guidelines}
3. YOUR FUNCTION NAMES (IN YOUR CODE) MUST END WITH THE SUFFIX "_v{num_versions}"
4. YOUR QUALNAME MUST NOT INCLUDE THE VERSION SUFFIX
5. YOUR QUALNAME MUST BE COMPLETE, AS SHOWN IN THE PREVIOUS VERSIONS

### FILE HIERARCHY ###
{file_hierarchy}
""".strip()

def load_prompt(filepath: Path) -> str:
    """
    Load a prompt from a file.
    
    Args:
        filepath (Path): The path to the prompt file.
        
    Returns:
        str: The content of the prompt file.
    """
    with open(filepath, 'r') as file:
        return file.read()

ROLE_PROMPT = load_prompt(PROMPTS_DIR / "role.txt")
GUIDELINES_PROMPT = load_prompt(PROMPTS_DIR / "guidelines.txt")
RESPONSE_FORMAT_PROMPT = load_prompt(PROMPTS_DIR / "response_format.txt")

def build_prompt(programs: list[Program], file_hierarchy: str, num_versions: int) -> str:
    """
    Build a prompt for the LLM based on the provided program structure.
    
    Args:
        programs (list[Program]): The program structures containing functions and their metadata.
        file_hierarchy (str): A string representation of the file hierarchy.
        num_versions (int): The number of versions of the program.

    Returns:
        str: The formatted prompt string.
    """
    prompt = PROMPT_HEADER_TEMPLATE.format(
        role=ROLE_PROMPT,
        guidelines=GUIDELINES_PROMPT,
        response_format=RESPONSE_FORMAT_PROMPT,
        file_hierarchy=file_hierarchy,
        num_versions=num_versions,
    )
    
    for program_id, program in enumerate(programs):
        prompt += f"\n\n### START OF PROGRAM VERSION {program_id} (*_v{program_id}) ###\n"
        
        for function in program.functions:
            func_fullname_comment = f"# filepath: {function.path}\n# qualname: {function.qualname}_v{program_id}\n"
            prompt += f"{func_fullname_comment}"
            prompt += f"{function.to_str(version=program_id, remove_imports=True)}\n\n"

        prompt += f"### END OF PROGRAM VERSION {program_id} ###\n\n"
    
    return prompt.strip()