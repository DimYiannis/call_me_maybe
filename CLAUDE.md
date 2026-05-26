# call me maybe — CLAUDE.md

## Project Overview
42 School project implementing LLM function calling via constrained decoding.
The goal is to translate natural language prompts into structured JSON function calls
using a 0.6B parameter model (Qwen/Qwen3-0.6B), with 100% valid JSON output
guaranteed through logit masking at each generation step.

## Key Constraint
This is NOT prompt engineering. The solution must intervene at the logit level
to mask invalid tokens before each selection step. Simply prompting the model
to output JSON is explicitly forbidden by the subject.

## Project Structure
src/                  # main module (run with uv run python -m src)
llm_sdk/              # provided package, copied here (do not modify)
data/input/           # functions_definition.json + function_calling_tests.json
data/output/          # generated at runtime, not in repo
pyproject.toml        # uv managed
Makefile

## Rules & Forbidden Things
- Python 3.10+, flake8, mypy (must pass without errors)
- All classes must use Pydantic for validation
- No dspy, pytorch, transformers, outlines, huggingface — forbidden
- No private methods/attributes from llm_sdk (no underscore access)
- Only allowed extra packages: numpy, json (stdlib)
- Function selection must be done by the LLM, not heuristics

## LLM SDK (llm_sdk package)
Only public methods available:
- get_logits_from_input_ids(input_ids: Tensor) -> Tensor
- get_path_to_vocabulary_json() -> str
- encode(text: str) -> List[int]
- decode(token_ids: List[int]) -> str  (optional)

## Constrained Decoding Logic
At each generation step:
1. Get logits from current input_ids
2. Determine valid next tokens given current JSON parse state + schema
3. Set invalid token logits to -inf
4. Pick argmax (or sample) from remaining
5. Append token, repeat until generation complete

JSON state machine must track:
- Expecting opening brace
- Expecting key (function name field, parameters field)
- Inside string vs number vs boolean value
- Which argument we are currently filling and its expected type
- Schema constraints (function name must be one of known functions)

## Input/Output Format
Input: data/input/function_calling_tests.json — array of {prompt: string}
Functions: data/input/functions_definition.json — array of function schemas
Output: data/output/function_calls.json — array of {prompt, name, parameters}

## Run Commands
uv sync                          # setup
make run                         # run with defaults
make lint                        # flake8 + mypy
make clean                       # remove caches

## Known Tricky Parts
- Subword tokenization: JSON structural tokens and function names
  may span multiple subword pieces — need prefix matching on vocab
- Type enforcement: numbers must be float in output even if integer in prompt
- Graceful error handling: malformed/missing input files must never crash
