*This project has been created as part of the 42 curriculum by ydimitra.*


# call me maybe — Function Calling in LLMs
Function calling tool that translates natural language
prompts into structured function calls
## Description
This project implements a function calling system for Large Language Models using constrained decoding. 
Given a natural language prompt like "What is the sum of 40 and 2?", the system identifies the correct function to call and extracts its arguments as structured JSON — guaranteed to be valid every time.
**The core challenge**: a 0.6B parameter model (Qwen3-0.6B) left to its own devices produces valid JSON maybe 30% of the time. With **constrained decoding** — intervening at the logit level to mask out invalid tokens before each selection step — we reach 100% schema-compliant output.

#### Algorithm Explanation
At each generation step, the model outputs logits (raw scores) over its entire vocabulary. Instead of just picking the highest score, we:

1. Track the current JSON parse state (e.g. "expecting a key", "inside a string value", "expecting a number")
2. Consult the vocabulary mapping to find which token strings are valid continuations
3. Set all invalid token logits to -inf
4. Sample from the remaining valid tokens

This enforces both syntactic JSON validity and semantic schema compliance — the function name must be one of the known functions, argument keys must match the definition, and types must match (number, string, boolean).

## Why is This Important

Function calling enables LLMs to:

• **Interact with external systems**: Call APIs, query databases, control devices

• **Execute code**: Perform calculations, data transformations, file operations

• **Chain operations**: Break complex tasks into executable steps

• **Provide structured output**: Generate JSON, XML, or other machine-readable
formats

• **Extract structured data from unstructured text**: For example, given a large
book, extract fields such as {protagonist name, protagonist sex, protagonist
age}


This technology powers modern AI assistants, code generation tools, and autonomous
agents, while also enabling tasks like automatic information extraction and knowledge
structuring from raw text


# Instructions

Requirements: Python 3.10+, uv
bash

## Install dependencies
make install

## Run with default input paths
make run

## Run with custom paths
```
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calls.json
```
## Lint
make lint

## Clean
make clean

Place llm_sdk/ in the same directory as src/. Run uv sync to set up the environment.

Example Usage:
```
Input (function_calling_tests.json):
json[
  { "prompt": "What is the sum of 40 and 2?" },
  { "prompt": "Reverse the string 'hello'" }
]
Output (function_calls.json):
json[
  {
    "prompt": "What is the sum of 40 and 2?",
    "name": "fn_add_numbers",
    "parameters": { "a": 40.0, "b": 2.0 }
  },
  {
    "prompt": "Reverse the string 'hello'",
    "name": "fn_reverse_string",
    "parameters": { "s": "hello" }
  }
]
```

## Design Decisions

**Pydantic for all input/output validation** — functions definitions and results are modeled as typed classes
Function selection via the LLM — no keyword matching or heuristics; the model chooses the function from context
Vocabulary JSON used to map token IDs to strings, enabling token-level constraint checking
Greedy decoding for constrained steps — since invalid tokens are already masked, temperature/sampling adds noise without benefit

Performance Analysis
MetricResult JSON validity 100% (guaranteed by design) Function selection accuracy~90%+Processing speed < 5 min for standard test sets
**The reliability comes entirely from structural guidance, not model size.**
Challenges Faced

Subword tokenization: function names and JSON structure tokens often span multiple subword pieces, requiring careful prefix-matching against the vocabulary
Parse state tracking: maintaining a lightweight JSON state machine that knows what's valid at each position without fully parsing incomplete JSON
Type coercion: numbers in prompts may appear as integers but must be output as floats per the schema

**Testing Strategy**

Ran against the provided example inputs to verify basic correctness
Tested edge cases: empty strings, large numbers, special characters, ambiguous prompts
Swapped in custom functions_definition.json files with different function sets to verify generalization
Verified mypy and flake8 pass cleanly

## Resources

Qwen3 model — HuggingFace

Constrained decoding — outline by the Outlines library

JSON Schema specification

Pydantic docs

**AI usage**: Claude was used to help structure the README and think through edge cases in the constrained decoding logic. All implementation decisions were made and validated independently.
