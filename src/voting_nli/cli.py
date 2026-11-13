"""CLI commands for batch NLI and individual booklet checks."""

import argparse
import json
from importlib.metadata import version
from pathlib import Path

from voting_nli.cases import Case, load_cases, read_booklet
from voting_nli.model import Model
from voting_nli.nli import Classifier

LABEL_NAMES = ["entailment", "neutral", "contradiction"]


def classify_batch(cases: list[Case], output_path: Path) -> None:
    classifier = Classifier(Model())

    with output_path.open("w", encoding="utf-8") as outputs:
        for case in cases:
            label, metrics = classifier.classify(
                reference=case.reference,
                claim=case.claim,
                vote=case.vote,
            )

            prediction = {
                "id": case.case_id,
                "label": label,
                "label_name": LABEL_NAMES[label],
                "evidence": [],
                "metrics": metrics,
            }
            outputs.write(json.dumps(prediction, ensure_ascii=False) + "\n")


def check_booklet(pdf_path: Path, claim: str) -> dict:
    reference = read_booklet(pdf_path)
    classifier = Classifier(Model())
    label, metrics = classifier.classify(reference, claim)

    return {
        "label": label,
        "label_name": LABEL_NAMES[label],
        "evidence": [],
        "metrics": metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multilingual voting booklet NLI and verified dialogue generation."
    )
    parser.add_argument("--version", action="version", version=version("voting-nli"))
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)

    commands = parser.add_subparsers(dest="command")
    check = commands.add_parser("check", help="Check a claim against a booklet.")
    check.add_argument("--pdf", type=Path, required=True)
    check.add_argument("--claim", required=True)

    args = parser.parse_args()

    try:
        if args.command == "check":
            result = check_booklet(args.pdf, args.claim)
            print(json.dumps(result, ensure_ascii=False))
            return

        if args.input is None or args.output is None:
            parser.error("Batch mode requires --input and --output.")
        if args.input.resolve() == args.output.resolve():
            parser.error("Input and output must be different files.")

        cases = load_cases(args.input, args.output)
    except (ValueError, OSError, RuntimeError) as error:
        parser.error(str(error))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    classify_batch(cases, args.output)


if __name__ == "__main__":
    main()
