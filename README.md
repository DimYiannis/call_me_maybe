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

**Fully static constrained decoding with runtime-selected branches** — Static grammars, dynamically selected. All argument grammars are compiled ahead of time, one per function. The active grammar is selected at inference time by the model's own function-name choice — the constraint program for the argument phase is not determined until the model commits to a name. Generation is fully constrained throughout; no phase is left unconstrained.

Performance Analysis
MetricResult JSON validity 100% (guaranteed by design) Function selection accuracy~90%+Processing speed < 5 min for standard test sets
**The reliability comes entirely from structural guidance, not model size.**
Challenges Faced

Subword tokenization: function names and JSON structure tokens often span multiple subword pieces, requiring careful prefix-matching against the vocabulary
Parse state tracking: maintaining a lightweight JSON state machine that knows what's valid at each position without fully parsing incomplete JSON
Type coercion: numbers in prompts may appear as integers but must be output as floats per the schema

## Known Limitations

These are model-size limitations — the constraint layer is correct, but a 0.6B model lacks the reasoning capacity to handle them reliably:

- **Word-number parsing**: "three hundred and forty two" may be parsed as 342 (one number) instead of 300 + 42. English "and" in numbers is ambiguous — the model reads it as a single number, which is technically correct English.
- **Regex escaping**: model does not reliably emit `\.` to match a literal dot — it defaults to `.` (any char). The constraint allows `\` inside strings, but the model never chooses to use it.
- **Names with apostrophes**: `O'Brien` → model produces empty string. Apostrophe is allowed by the constraint; the model simply fails to extract the name correctly.
- **No matching function**: when the prompt implies an operation with no corresponding function (e.g. "subtract"), the constraint forces a valid function name — the model picks the closest match, which may be semantically wrong. This is by design: the grammar guarantees structural validity, not semantic correctness.

**Testing Strategy**

Ran against the provided example inputs to verify basic correctness
Tested edge cases: empty strings, large numbers, special characters, ambiguous prompts
Swapped in custom functions_definition.json files with different function sets to verify generalization
Verified mypy and flake8 pass cleanly

## Theoretical Basis

This project implements the approach described in:

> **Efficient Guided Generation for Large Language Models**
> Brandon T. Willard, Rémi Louf (2023)
> [arXiv:2307.09702](https://arxiv.org/abs/2307.09702)

The paper formalises constrained decoding as transitions through a finite-state machine (FSM). At each generation step, the FSM determines which vocabulary tokens are valid continuations; all others are masked to -inf before token selection. Key properties we inherit from this approach:

- **100% structural validity** — invalid JSON is impossible by construction, not by prompt
- **Model-agnostic** — works with any LLM that exposes per-step logits
- **Negligible overhead** — vocabulary indexing is done once at startup; per-step masking is O(vocab_size)
- **Schema compliance** — the FSM enforces not just JSON syntax but the specific function name and argument types

---

> **Don't Fine-Tune, Decode: Syntax Error-Free Tool Use via Constrained Decoding**
> Zhang et al. (2023) · [arXiv:2310.07075](https://arxiv.org/abs/2310.07075)

### Core argument

Syntax constraints are only learned **implicitly** during fine-tuning — models still make frequent syntax errors. Enforcing constraints **explicitly at decode time** via finite state machines is more reliable and requires no expensive fine-tuning.

### System: TOOLDEC

A constrained decoding algorithm using FSMs to enforce tool syntax compliance. Works on top of instruction-tuned LLMs without modifying weights.

### Key result

Mistral-Instruct: tool use accuracy **0% → 52%** with constrained decoding. Zero syntax errors across all tested models and benchmarks.

### Key insight

> "Syntax constraints are better enforced explicitly during decoding than implicitly during training."

This is the direct theoretical backing for this project's approach. We don't fine-tune Qwen3-0.6B on JSON examples and hope it learns the format. We enforce the format structurally at every token step.

### Connection to Willard & Louf (2023)

Willard & Louf provide the FSM-based vocabulary indexing mechanism. Zhang et al. apply the same principle specifically to tool/function calling and make the empirical case against fine-tuning.

---

> **Thinking Before Constraining: A Unified Decoding Framework**
> Nguyen et al. (2025) · [arXiv:2601.07525](https://arxiv.org/abs/2601.07525)

### Core idea: In-Writing

Hybrid decoding that decouples reasoning from formatting. The model generates unconstrained free-form reasoning, then a trigger token switches on structured constrained decoding for the output field.

### The problem it solves

Applying constraints too early interrupts the model's reasoning ("premature triggering"). Fully unconstrained output lacks verifiable structure. In-Writing threads the needle: reason freely, then enforce format.

### Key result

Up to **27% accuracy improvement** over purely natural generation on classification and reasoning tasks.

### Connection to this project

Points to a natural extension: instead of constraining the entire output from token one, allow the model to reason first:

```json
{
  "reasoning": "<unconstrained — model thinks freely>",
  "function_call": { "name": "<constrained>", "parameters": { "<constrained>" } }
}
```

Current implementation constrains from the first token. In-Writing suggests reasoning-before-committing could improve function selection accuracy further, especially on ambiguous prompts.

## Complexity Insight

Prior approaches (e.g. Guidance) matched from sequence start and scanned the full vocabulary at every step — O(N) per token. Outlines amortizes all computation to preprocessing → O(1) at inference.

## Resources

Qwen3 model — HuggingFace

Willard & Louf (2023) — [Efficient Guided Generation for Large Language Models](https://arxiv.org/abs/2307.09702)

Nguyen et al. (2025) - [Thinking Before Constraining: A Unified Decoding Framework](https://arxiv.org/abs/2601.07525)

Zhang et al. (2023) - [Don't Fine-Tune, Decode: Syntax Error-Free Tool Use via Constrained Decoding](https://arxiv.org/abs/2310.07075)

[Let's Build the GPT Tokenizer — Andrej Karpathy](https://www.fast.ai/posts/2025-10-16-karpathy-tokenizers.html)

[A visual introduction to tokenization in LLMs — BPE Algorithm](https://www.youtube.com/watch?v=APnKbi448O4)

[Tool Calling Explained: Turn Your LLM into an AI Agent](https://www.youtube.com/watch?v=KJf7SqPCRXg)

Pydantic docs

**AI usage**: Claude was used to help structure the README and think through edge cases in the constrained decoding logic. All implementation decisions were made and validated independently.
