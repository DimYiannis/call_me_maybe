"""Execute decoded function calls and return their results."""
import math
import re
from typing import Any

from .models import FunctionCall


def execute(call: FunctionCall) -> Any:
    """
    Execute a decoded function call and return the result.

    Args:
        call: The decoded function call with name and parameters.

    Returns:
        The result of executing the function.

    Raises:
        ValueError: If the function name is unknown.
    """
    p = call.parameters
    if call.name == "fn_add_numbers":
        return p["a"] + p["b"]
    if call.name == "fn_greet":
        return f"Hello, {p['name']}!"
    if call.name == "fn_reverse_string":
        return p["s"][::-1]
    if call.name == "fn_get_square_root":
        return math.sqrt(p["a"])
    if call.name == "fn_substitute_string_with_regex":
        return re.sub(p["regex"], p["replacement"], p["source_string"])
    raise ValueError(f"Unknown function: {call.name}")
