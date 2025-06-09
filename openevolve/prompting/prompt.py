from openevolve.project_indexer import ProjectIndexer
from openevolve.code_manipulation import Program

from pathlib import Path

PROMPTS_DIR = Path(__file__).parent / "prompts"

PROMPT_HEADER_TEMPLATE = """
### ROLE ###
{role}

### GUIDELINES ###
{guidelines}

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

def build_prompt(program: Program) -> str:
    """
    Build a prompt for the LLM based on the provided program structure.
    
    Args:
        program (Program): The program structure containing functions and their metadata.
        
    Returns:
        str: The formatted prompt string.
    """
    file_hierarchy = ProjectIndexer.get_tree_description(program)

    prompt = PROMPT_HEADER_TEMPLATE.format(
        role=ROLE_PROMPT,
        guidelines=GUIDELINES_PROMPT,
        response_format=RESPONSE_FORMAT_PROMPT,
        file_hierarchy=file_hierarchy
    )
    
    prompt += "\n\n### START OF PROGRAM VERSION 0 (*_v0) ###\n"
    
    for function in program.functions:
        func_fullname_comment = f"# {function.path}: {function.qualname}\n"
        prompt += f"{func_fullname_comment}"
        prompt += f"{function.to_str(version=0)}\n\n"

    prompt += "### END OF PROGRAM VERSION 0 ###\n"
    
    return prompt.strip()