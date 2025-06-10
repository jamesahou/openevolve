from openevolve.project_indexer import ProjectIndexer
from openevolve.code_manipulation import Program

from dataclasses import dataclass
from pathlib import Path

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

TASK_PROMPT_TEMPLATE = """
The user has provided you some special information to help you understand the context of the code you will be working on. Read the following carefully and use it to inform your work. The message from the user is enclosed in triple backticks below:

```
{task}
```
""".strip()

@dataclass
class PromptSection:
    """
    Represents a section of the prompt.
    
    Attributes:
        title (str): The title of the section.
        content (str): The content of the section.
    """
    title: str
    content: str

def compose_prompt(sections: list[PromptSection]) -> str:
    """
    Compose a prompt from multiple sections.
    
    Args:
        sections (list[PromptSection]): A list of sections to include in the prompt.
        
    Returns:
        str: The composed prompt string.
    """
    prompt = ""
    for section in sections:
        prompt += f"### {section.title.upper()} ###\n{section.content}\n\n"
    return prompt.rstrip()

def get_role_prompt(role: str) -> PromptSection:
    return PromptSection(title="ROLE", content=role)

def get_task_prompt(task: str) -> PromptSection:
    prompt_text = TASK_PROMPT_TEMPLATE.format(task=task)
    return PromptSection(title="TASK", content=prompt_text)

def get_guidelines_prompt(guidelines: str) -> PromptSection:
    return PromptSection(title="GUIDELINES", content=guidelines)

def get_file_hierarchy_prompt(file_hierarchy: str) -> PromptSection:
    return PromptSection(title="FILE HIERARCHY", content=file_hierarchy)

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

def build_prompt(
    programs: list[Program],
    file_hierarchy: str,
    num_versions: int,
    extra_prompt: str | None = None,
) -> str:
    """
    Build a prompt for the LLM based on the provided program structure.
    
    Args:
        programs (list[Program]): The program structures containing functions and their metadata.
        file_hierarchy (str): A string representation of the file hierarchy.
        num_versions (int): The number of versions of the program.
        extra_prompt (str | None): Additional prompt text to append to the main prompt.

    Returns:
        str: The formatted prompt string.
    """
    # Get the prompt sections
    sections = [
        get_role_prompt(ROLE_PROMPT),
    ]

    if extra_prompt:
        sections.append(get_task_prompt(extra_prompt))

    guidelines_prompt_text = GUIDELINES_PROMPT.format(num_versions=num_versions)
    sections.append(get_guidelines_prompt(guidelines_prompt_text))

    sections.append(get_file_hierarchy_prompt(file_hierarchy))

    # Compose the main prompt
    prompt = compose_prompt(sections)

    for program_id, program in enumerate(programs):
        prompt += f"\n\n### START OF PROGRAM VERSION {program_id} (*_v{program_id}) ###\n"
        
        for function in program.functions:
            func_fullname_comment = f"# filepath: {function.path}\n# qualname: {function.qualname}_v{program_id}\n"
            prompt += f"{func_fullname_comment}"
            prompt += f"{function.to_str(version=program_id, remove_imports=True)}\n\n"

        prompt += f"### END OF PROGRAM VERSION {program_id} ###\n\n"
    
    return prompt.strip()