"""Define data shapes with pydantic."""
from pydantic import BaseModel
from typing import Any


class ParameterSchema(BaseModel):
    """Schema for a single function parameter."""

    type: str
    description: str | None = None


class ReturnSchema(BaseModel):
    """Schema for a function return value."""

    type: str


class FunctionDefinition(BaseModel):
    """Full specification of a callable function."""

    name: str
    description: str
    parameters: dict[str, ParameterSchema]
    returns: ReturnSchema | None = None


class TestPrompt(BaseModel):
    """Single test input prompt."""

    prompt: str


class FunctionCall(BaseModel):
    """Generated function call result."""

    prompt: str
    name: str
    parameters: dict[str, Any]
