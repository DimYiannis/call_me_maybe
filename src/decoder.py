"""Constrained decoding loop: prompt → valid JSON function call."""
import json
from llm_sdk import Small_LLM_Model

from .constraints import Constraint
from .models import FunctionCall, FunctionDefinition, TestPrompt
from .vocabulary import Vocabulary


def _build_prompt(
    prompt: str,
    functions: list[FunctionDefinition],
) -> str:
    """
    Build the model input string for a single test prompt.

    Args:
        prompt: The natural-language user request.
        functions: Known function definitions.

    Returns:
        Full prompt string ready for encoding.
    """
    lines = [
        "You are a function-calling assistant.",
        "Choose the correct function and arguments for the user request.",
        "",
        "Functions:",
    ]
    for func in functions:
        args = ", ".join(
            f"{name}: {schema.type}"
            for name, schema in func.parameters.items()
        )
        lines.append(f"- {func.name}({args}): {func.description}")
    lines += ["", f"User request: {prompt}", "", "JSON:"]
    return "\n".join(lines)


def _parse_output(generated: str, prompt: str) -> FunctionCall:
    """
    Parse the generated JSON string into a FunctionCall.

    Args:
        generated: Raw JSON string produced by decoding.
        prompt: The original user prompt (stored in output).

    Returns:
        FunctionCall with name and parameters extracted.

    Raises:
        ValueError: If generated string is not valid JSON or missing keys.
    """
    try:
        data = json.loads(generated)
    except json.JSONDecodeError as e:
        raise ValueError(f"Generated output is not valid JSON: {e}")
    if "name" not in data or "parameters" not in data:
        raise ValueError(f"Generated JSON missing required keys: {generated}")
    return FunctionCall(
        prompt=prompt,
        name=data["name"],
        parameters=data["parameters"],
    )


def _print_logit_step(
    step: int,
    logits: list[float],
    valid_ids: list[int],
    masked: list[float],
    next_id: int,
    vocab: Vocabulary,
    top_n: int = 5,
) -> None:
    """
    Print a human-readable summary of one logit-masking step.

    Args:
        step: Generation step index.
        logits: Raw logits from the model (full vocab).
        valid_ids: Token ids allowed by the constraint.
        masked: Logits after masking invalid tokens to -inf.
        next_id: The token id selected (argmax of masked).
        vocab: Vocabulary for id→string lookup.
        top_n: How many top tokens to show before/after masking.
    """
    print(f"\n--- Step {step} ---")
    print(f"  Valid tokens: {len(valid_ids)} / {len(logits)}")

    top_before = sorted(
        range(len(logits)), key=lambda i: logits[i], reverse=True
    )[:top_n]
    print(f"  Top {top_n} BEFORE masking:")
    for i in top_before:
        marker = "✓" if i in set(valid_ids) else "✗"
        try:
            tok = repr(vocab.get_token(i))
        except KeyError:
            tok = f"<id={i}>"
        print(f"    [{marker}] {tok:20s} logit={logits[i]:.3f}")

    top_after = sorted(
        [i for i in range(len(masked)) if masked[i] != float("-inf")],
        key=lambda i: masked[i],
        reverse=True,
    )[:top_n]
    print(f"  Top {top_n} AFTER masking:")
    for i in top_after:
        try:
            tok = repr(vocab.get_token(i))
        except KeyError:
            tok = f"<id={i}>"
        print(f"    {tok:20s} logit={masked[i]:.3f}")

    try:
        chosen = repr(vocab.get_token(next_id))
    except KeyError:
        chosen = f"<id={next_id}>"
    print(f"  → Chosen: {chosen}")


def decode(
    prompt: TestPrompt,
    functions: list[FunctionDefinition],
    model: Small_LLM_Model,
    vocab: Vocabulary,
    max_tokens: int = 256,
    debug: bool = False,
) -> FunctionCall:
    """
    Run constrained decoding for one prompt and return the function call.

    At each step: get logits, mask invalid token ids to -inf, pick argmax,
    advance constraint state, repeat until constraint is satisfied.

    Args:
        prompt: Test prompt to process.
        functions: Known function definitions (schema).
        model: Loaded LLM instance.
        vocab: Loaded model vocabulary.
        max_tokens: Safety cap on generation length.
        debug: Print logit masking details at each step.

    Returns:
        FunctionCall produced by constrained decoding.

    Raises:
        ValueError: If no valid tokens exist at some step (should not
            happen if constraint and vocab are consistent).
        RuntimeError: If max_tokens reached before generation completes.
    """
    input_text = _build_prompt(prompt.prompt, functions)
    input_ids: list[int] = list(model.encode(input_text).squeeze().tolist())

    constraint = Constraint(functions, vocab)
    generated_ids: list[int] = []

    for step in range(max_tokens):
        if constraint.is_complete():
            break

        logits: list[float] = model.get_logits_from_input_ids(input_ids)
        valid_ids = constraint.valid_next_ids()

        if not valid_ids:
            raise ValueError(
                "No valid tokens available — constraint/vocab mismatch."
            )

        valid_set = set(valid_ids)
        masked: list[float] = [
            logit if i in valid_set else float("-inf")
            for i, logit in enumerate(logits)
        ]

        next_id = int(max(range(len(masked)), key=lambda i: masked[i]))

        if debug:
            _print_logit_step(step, logits, valid_ids, masked, next_id, vocab)

        constraint.accept(next_id)
        generated_ids.append(next_id)
        input_ids.append(next_id)

    else:
        raise RuntimeError(
            f"max_tokens ({max_tokens}) reached before generation completed."
        )

    generated_text = model.decode(generated_ids)
    return _parse_output(generated_text, prompt.prompt)
