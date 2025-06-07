from funsearch.code_manipulation_2 import Program, Function, Decorator, str_to_functions
from typing import Iterable
from enum import Enum
import os

class TreeSymbol(Enum):
    """Symbols used to represent the tree structure in the pretty print."""
    BRANCH = '├── '
    LAST_BRANCH = '└── '
    VERTICAL = '│   '
    HORIZONTAL = '── '

class Node:
    def __init__(self, name: str):
        self.name = name
        self.children = {}

    def add_child(self, child: "Node") -> "Node":
        return self.children.setdefault(child.name, child)
        
    def get_children(self) -> Iterable["Node"]:
        yield from self.children.values()

class FileNode(Node):
    def __str__(self, prefix=""):
        return f"{prefix}{self.name} ({self.qual_name})\n"

class FolderNode(Node):
    def __str__(self):
        return f"{self.name}/"

class ClassNode(Node):
    def __str__(self):
        return f"class {self.name}:"

class FunctionNode(Node):
    def __init__(self, name: str, function: Function):
        self.name = name
        self.children = {}
        self.function = function

    def __str__(self):
        return self.function.name + '(' + ', '.join(arg.name for arg in self.function.header.args) + ')'

class ProjectTree:
    def __init__(self, project_root: str):
        self.project_root = project_root
        self.tree = FolderNode("root")

    def insert_function(self, function: Function) -> FunctionNode:
        # Parse the relative path of the function
        rel_path = os.path.relpath(function.path, self.project_root)

        path_parts = rel_path.split(os.sep)
        qual_parts = function.qual_name.split('.')
        
        # Start at the root of the tree
        node = self.tree

        # Insert folders leading up to the file
        for folder_name in path_parts[:-1]:
            folder = FolderNode(folder_name)
            node = node.add_child(folder)

        # Insert the file node
        file_name = path_parts[-1]
        file_node = FileNode(file_name)

        node = node.add_child(file_node)

        # Insert the classes into the file node
        if len(qual_parts) > 1:
            class_names = qual_parts[:-1]
            method_name = qual_parts[-1]
        else:
            class_names = []
            method_name = qual_parts[0]
        
        for class_name in class_names:
            class_node = ClassNode(class_name)
            node = node.add_child(class_node)
        
        # Insert the function node into the class or file node
        function_node = FunctionNode(method_name, function)
        node.add_child(function_node)
        assert node.children[method_name] == function_node, f"Function node {method_name} already exists in {node.name}"
        return function_node

    def pretty_print(self) -> str:
        """
        Returns a string representation of the project tree structure.

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
        def _pretty_print(node: Node, prefix: str = "", is_last: bool = True) -> str:
            result = ""
            if isinstance(node, FileNode):
                result += f"{prefix}{TreeSymbol.LAST_BRANCH.value if is_last else TreeSymbol.BRANCH.value}{node}\n"
            elif isinstance(node, FolderNode):
                result += f"{prefix}{TreeSymbol.LAST_BRANCH.value if is_last else TreeSymbol.BRANCH.value}{node}\n"
            elif isinstance(node, ClassNode):
                result += f"{prefix}{TreeSymbol.LAST_BRANCH.value if is_last else TreeSymbol.BRANCH.value}{node}\n"
            elif isinstance(node, FunctionNode):
                result += f"{prefix}{TreeSymbol.LAST_BRANCH.value if is_last else TreeSymbol.BRANCH.value}{node}\n"

            children = list(node.get_children())
            for i, child in enumerate(children):
                result += _pretty_print(child, prefix + (TreeSymbol.VERTICAL.value if not is_last else "    "), i == len(children) - 1)
            return result

        return _pretty_print(self.tree)

class ProjectIndexer:
    def __new__(cls, project_root: str):
        cls.project_root = project_root
        cls.project_tree = ProjectTree(project_root)
        cls._build_project_tree()

    @classmethod
    def _build_project_tree(cls):
        for dirpath, _, filenames in os.walk(cls.project_root):
            for filename in filenames:
                if filename.endswith(".py"):
                    # Read the file content
                    filepath = os.path.join(dirpath, filename)

                    with open(filepath, "r") as file:
                        code = file.read()

                    # Extract all functions from the code string
                    program = Program.from_code(code)

                    # Insert each function into the project tree
                    for function in program.functions:
                        cls.project_tree.insert_function(function)

    def get_tree_description(cls, program: Program) -> str:
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
        # Build a subtree of the project tree for the given program
        subtree = ProjectTree(cls.project_root)

        for function in program.functions:
            subtree.insert_function(function)

        return subtree.pretty_print()
        
if __name__ == "__main__":
    # Example usage
    project_root = "example_project/"
    indexer = ProjectIndexer(project_root)
    print(indexer.project_tree.pretty_print())