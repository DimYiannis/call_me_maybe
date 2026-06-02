"""Serialize FunctionCall results to JSON output."""
import json
import os
from typing import Any

from .models import FunctionCall


def save_results(results: list[FunctionCall], filename: str) -> None:
    """
    Write function call results to output JSON file.

    Args:
        results: List of FunctionCall objects to serialize.
        filename: Path to the output JSON file.

    Raises:
        OSError: If the output directory cannot be created or file cannot be
            written.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    data: list[dict[str, Any]] = [r.model_dump() for r in results]
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)
