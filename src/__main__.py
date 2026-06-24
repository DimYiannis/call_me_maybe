"""Entry point for call_me_maybe."""
import argparse
import sys


def main() -> None:
    """Parse args, load data, run constrained decoding pipeline."""
    parser = argparse.ArgumentParser(
        description="LLM function calling via constrained decoding"
    )
    parser.add_argument(
        "--functions_definition",
        default="data/input/functions_definition.json",
    )
    parser.add_argument(
        "--input",
        default="data/input/function_calling_tests.json",
    )
    parser.add_argument(
        "--output",
        default="data/output/function_calls.json",
    )
    parser.add_argument(
        "--visual",
        action="store_true",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
    )
    args = parser.parse_args()

    try:
        from llm_sdk import Small_LLM_Model
    except ImportError:
        print("Error: llm_sdk not found. Copy llm_sdk folder to project root.")
        sys.exit(1)

    from .parser import load_functions, load_prompts
    from .serializer import save_results
    from .vocabulary import Vocabulary
    from .decoder import decode

    functions = load_functions(args.functions_definition)
    prompts = load_prompts(args.input)


    model = Small_LLM_Model()
    # model_name="Qwen/Qwen3-1.7B"
    vocab = Vocabulary(model)

    results = []
    for test_prompt in prompts:
        if args.visual:
            print(f"Processing: {test_prompt.prompt}")
        result = decode(test_prompt, functions, model, vocab, debug=args.debug)
        if args.visual:
            print(f"  -> {result.name}({result.parameters})")
        results.append(result)

    save_results(results, args.output)
    print(f"Done. {len(results)} results written to {args.output}")


if __name__ == "__main__":
    main()
