import argparse as ap
import json
from importlib.metadata import version
from pathlib import Path

import pymupdf

from voting_nli.model import Model
from voting_nli.nli import Classifier

LABELS = ["entailment", "neutral", "contradiction"]

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


def batch_classification(args, cases: list[dict]):
    model = Model()
    nli = Classifier(model)

    with args.output.open("w", encoding="utf-8") as outputs:
        for case in cases:
            label, metrics = nli.classify(case["reference"]["text"], case["claim"]["text"])

            prediction = {
                "id": case["id"],
                "label": label,
                "label_name": LABELS[label],
                "evidence": [],
                "metrics": metrics,
            }
            outputs.write(json.dumps(prediction, ensure_ascii=False) + "\n")


def pdf_classification(pdf, claim: str):
    with pymupdf.open(pdf) as document:
        # reconstruct text's reading order
        pages = [page.get_text(sort=True) for page in document]

    if not any(text.strip() for text in pages):
        raise ValueError("PDF contains no extractable text.")

    reference = "\n\n".join(f"[Page {number}]\n{text}"
        for number, text in enumerate(pages, start=1)
        if text.strip()
    )

    classifier = Classifier(Model())
    label, metrics = classifier.classify(reference, claim)

    result = {
        "label": label,
        "label_name": LABELS[label],
        "evidence": [],
        "metrics": metrics
    }
    print(json.dumps(result, ensure_ascii=False))

def main() -> None:
    parser = ap.ArgumentParser(
        description="Multilingual voting booklet NLI and verified dialogue generation."
    )
    # jsonl mode
    parser.add_argument("--version", action="version", version=version("voting-nli"))
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)

    # pdf mode
    commands = parser.add_subparsers(dest="command")
    check = commands.add_parser("check", help="Check a claim against a booklet.")
    check.add_argument("--pdf", type=Path, required=True)
    check.add_argument("--claim", required=True)

    args = parser.parse_args()

    if args.command == "check":
        pdf_classification(args.pdf, args.claim)
        return

    if args.input is None or args.output is None:
        parser.error("Batch mode requires --input and --output.")

    if args.input.resolve() == args.output.resolve():
        parser.error("Input and output must be different files.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    cases = parse_cases(args, parser)
    batch_classification(args, cases)


if __name__ == "__main__":
    main()
