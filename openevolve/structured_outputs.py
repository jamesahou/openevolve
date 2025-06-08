from pydantic import BaseModel, Field
import dataclasses


class FunctionImplementation(BaseModel):
    """
    Represents the implementation details of a function.
    """
    filepath: str = Field(
        description="The path to the file containing the function implementation."
    )
    qualname: str = Field(
        description="The qualified name of the function (e.g. MyClass.myfunction)."
    )
    code: str = Field(
        description="The source code of the function implementation."
    )

class ProgramImplementation(BaseModel):
    """
    Represents the implementation details of a structured output.
    """
    functions: list[FunctionImplementation] = Field(
        default_factory=list,
        description="A list of function implementations that are part of the program."
    )