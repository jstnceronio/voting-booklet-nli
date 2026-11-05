import argparse as ap
import json
from importlib.metadata import version
from pathlib import Path

from voting_nli.model import Model
from voting_nli.nli import Classifier


def parse_cases(args, parser: ap.ArgumentParser) -> list[dict]:
    cases = []
    with args.input.open(encoding="utf-8") as inputs:
        for line_number, line in enumerate(inputs, start=1):
            case = json.loads(line)
            if "reference" not in case:
                parser.error(
                    f"Input line {line_number} (id={case.get('id')!r}) has no 'reference'. "
                    "Booklet inputs are not supported yet; use reference-NLI input."
                )
            cases.append(case)
    return cases


def start_classification(args, cases: list[dict]):
    model = Model()
    nli = Classifier(model)
    labels = ["entailment", "neutral", "contradiction"]

    with args.output.open("w", encoding="utf-8") as outputs:
        for case in cases:
            label, metrics = nli.classify(case["reference"]["text"], case["claim"]["text"])

            prediction = {
                "id": case["id"],
                "label": label,
                "label_name": labels[label],
                "evidence": [],
                "metrics": metrics,
            }
            outputs.write(json.dumps(prediction, ensure_ascii=False) + "\n")

def main() -> None:
    parser = ap.ArgumentParser(
        description="Multilingual voting booklet NLI and verified dialogue generation."
    )
    parser.add_argument("--version", action="version", version=version("voting-nli"))
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)

    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        parser.error("Input and output must be different files.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    cases = parse_cases(args, parser)
    start_classification(args, cases)


if __name__ == "__main__":
    main()
