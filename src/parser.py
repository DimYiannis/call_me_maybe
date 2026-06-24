"""Read and validate JSON input files into pydantic models."""
import json
import os
from typing import Any

from pydantic import ValidationError

from .models import FunctionDefinition, TestPrompt


def load_functions(filename: str) -> list[FunctionDefinition]:
    """
    Read and validate functions_definition.json.

    Args:
        filename: Path to the JSON file containing function definitions.

    Returns:
        List of FunctionDefinition objects parsed from the file.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file contains invalid JSON.
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Functions file {filename} not found.")
    with open(filename, 'r') as f:
        try:
            data: list[Any] = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Functions file {filename} contains invalid JSON: {e}"
            )
    if not isinstance(data, list):
        raise ValueError(
            f"Functions file {filename} must contain a JSON list."
        )
    try:
        return [FunctionDefinition(**item) for item in data]
    except ValidationError as e:
        raise ValueError(
            f"Functions file {filename} does not match schema: {e}"
        )


def load_prompts(filename: str) -> list[TestPrompt]:
    """
    Read and validate function_calling_tests.json.

    Args:
        filename: Path to the JSON file containing prompts.

    Returns:
        List of TestPrompt objects parsed from the file.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file contains invalid JSON.
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Prompts file {filename} not found.")
    with open(filename, "r") as f:
        try:
            data: list[Any] = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Prompts file {filename} contains invalid JSON: {e}"
            )
    if not isinstance(data, list):
        raise ValueError(
            f"Prompts file {filename} must contain a JSON list."
        )
    try:
        return [TestPrompt(**item) for item in data]
    except ValidationError as e:
        raise ValueError(
            f"Prompts file {filename} does not match schema: {e}"
        )
