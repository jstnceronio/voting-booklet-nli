import argparse
import json
import random
from importlib.metadata import version
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multilingual voting booklet NLI and verified dialogue generation."
    )
    parser.add_argument("--version", action="version", version=version("voting-nli"))
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)

    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        parser.error("Input and output must be different files.")

    labels = ["entailment", "neutral", "contradiction"]

    with args.input.open(encoding="utf-8") as inputs:
        args.output.parent.mkdir(parents=True, exist_ok=True)

        with args.output.open("w", encoding="utf-8") as outputs:
            for line in inputs:
                case = json.loads(line)
                label = random.randrange(3)

                # Replace me.
                prediction = {
                    "id": case["id"],
                    "label": label,
                    "label_name": labels[label],
                    "evidence": [],
                    "metrics": {
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "inference_time_ms": 0,
                    },
                }
                outputs.write(json.dumps(prediction, ensure_ascii=False) + "\n")

    parser.parse_args()
    parser.print_help()


if __name__ == "__main__":
    main()
