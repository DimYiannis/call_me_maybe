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
    args = parser.parse_args()

    try:
        from llm_sdk import Small_LLM_Model  # noqa: F401
    except ImportError:
        print("Error: llm_sdk not found. Copy llm_sdk folder to project root.")
        sys.exit(1)

    funcs_def_path = args.functions_definition
    input_path = args.input
    output_path = args.output
    visual = args.visual



if __name__ == "__main__":
    main()
