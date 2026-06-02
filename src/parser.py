"""read json files"""
import json
import os
from models import FunctionDefinition, TestPrompt

def load_functions(filename:str) -> list[FunctionDefinition]:
    """
    reads and validates functions_definition.json
    Args:
        filename: Path to the JSON file containing function definitions.

    Returns:
        List of FunctionDefinition objects parsed from the file.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file contains invalid JSON.
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"file with functions {filename} not found")
    with open(filename, 'r') as f:
        try:
            data: list[Any] = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Function file {filename} contains invalid JSON: {e}"
            )
    return [FunctionDefinition(**item) for item in data]


def load_prompts(filename:str) -> list[InputPrompts]:
    """
    reads and validates function_calling_tests.json
    Args:
        filename: Path to the JSON file containing prompts.

    Returns:
        List of InputPrompts objects parsed from the file.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file contains invalid JSON.
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f" prompts file {filename} was not found")
    with open(filename, "r") as f:
        try:
            data: list[Any] = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"prompt file {filename} is not in a valid json format {e}")
    return [InputPrompts(**item) for item in data]