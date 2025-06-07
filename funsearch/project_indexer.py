from funsearch.code_manipulation_2 import Program
import os

class ProjectIndexer:
    def __init__(self, project_root: str):
        self.project_root = project_root

    def index_project(self):
        for dirpath, _, filenames in os.walk(self.project_root):
            for filename in filenames:
                if filename.endswith(".py"):
                    filepath = os.path.join(dirpath, filename)
                    with open(filepath, "r") as file:
                        code = file.read()

                    # Extract all functions from the code string
                    functions = self._extract_functions(code)

                    # TODO

    def _extract_functions(self, code: str):
        """Extracts function definitions from the provided code."""
        raise NotImplementedError

    def get_tree_description(self, program: Program) -> str:
        """
        Returns an indented string representation of the project structure,
        including modules, classes, methods, and functions, as specified.

        Example output:

        ├── package/
        │   ├── module1.py
        │   │   ├── class ClassName:
        │   │   │   ├── method1(self, arg)
        │   │   │   ├── @classmethod
        │   │   │   │   └── class_method(cls, arg)
        │   │   │   └── @staticmethod
        │   │   │       └── static_method(arg)
        │   │   └── function1(arg1, arg2)
        │   └── subpackage/
        │       └── module2.py
        │           └── function2(arg1)
        └── main.py
            └── main()
        """
        raise NotImplementedError