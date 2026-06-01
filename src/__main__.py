import argparse

parser = arg.parse.ArgumentParser()

parser.add_argument("--functions_definition", required=True)
parser.add_argument("--input", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--visual", required=True)

args = parser.parse_args()


def main() -> None:
    try:
        from llm_sdk import Small_LLM_Model
    except ImportError:
        print(
            "Error: llm_sdk not found."
                "copy llm_sdk folder in the root of the project")
        sys.exit(1)
    funcs_def = args[0]
    input_path = args[1]
    output_path = args[2]
    
