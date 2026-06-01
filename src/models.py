""" define data shapes with pydantic"""
from pydantic import BaseModel
from typing import Any

class ParameterSchema(BaseModel):
    type: str
    description: str

class FunctionDefinition(BaseModel):
    name: str
    description: str
    parameters: dict[str, ParametersSchema]

class TestPrompt(BaseModel):
    prompt: str

class FunctionCall(BaseModel):
    prompt: str
    name: str
    parameters: dict[str, Any]


