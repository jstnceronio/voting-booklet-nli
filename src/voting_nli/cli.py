"""CLI commands for batch NLI and individual booklet checks."""

import argparse
import json
from importlib.metadata import version
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

from openai import OpenAIError
from pydantic import ValidationError

from voting_nli.cases import Case, load_cases, read_booklet
from voting_nli.model import Model
from voting_nli.nli import Classifier

LABEL_NAMES = ["entailment", "neutral", "contradiction"]


def classify_batch(cases: list[Case], output_path: Path) -> None:
    classifier = Classifier(Model())

    with TemporaryDirectory(dir=output_path.parent, prefix=f".{output_path.name}.") as directory:
        temporary_path = Path(directory) / "predictions.jsonl"
        with temporary_path.open("w", encoding="utf-8") as outputs:
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
        temporary_path.replace(output_path)


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="voting-nli",
        description="Check claims against voting booklets or reference text.",
        epilog=(
            "Examples:\n"
            "  voting-nli batch --input cases.jsonl --output runs/predictions.jsonl\n"
            '  voting-nli check --pdf booklet.pdf --claim "The vote is in June."\n\n'
            "Run voting-nli COMMAND --help for command options.\n"
            "Model requests require API_KEY, BASE_URL and MODEL in the environment or .env."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {version('voting-nli')}"
    )

    commands = parser.add_subparsers(dest="command", title="commands")
    batch = commands.add_parser(
        "batch", help="Classify cases from a JSONL file.",
        description="Classify reference and booklet cases, writing one JSON prediction per line.",
        epilog="Existing output is replaced only after every case succeeds.",
    )
    batch.add_argument(
        "-i", "--input", type=Path, required=True, metavar="JSONL",
        help="Input cases; booklet paths are relative to this file.",
    )
    batch.add_argument(
        "-o", "--output", type=Path, required=True, metavar="JSONL",
        help="Predictions file; parent directories are created automatically.",
    )
    check = commands.add_parser(
        "check", help="Check one claim against a PDF booklet.",
        description="Check a claim against a PDF booklet and print a JSON prediction to stdout.",
    )
    check.add_argument(
        "--pdf", type=Path, required=True, metavar="PDF", help="Booklet with extractable text.",
    )
    check.add_argument(
        "--claim", required=True, metavar="TEXT", help="Claim to check; quote multiword claims.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return

    try:
        if args.command == "check":
            result = check_booklet(args.pdf, args.claim)
            print(json.dumps(result, ensure_ascii=False))
            return

        if args.input.resolve() == args.output.resolve():
            parser.error("Input and output must be different files.")

        cases = load_cases(args.input, args.output)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        print(f"Classifying {len(cases)} cases...", file=sys.stderr)
        classify_batch(cases, args.output)
        print(f"Wrote {len(cases)} predictions to {args.output}.", file=sys.stderr)

    except ValidationError:
        parser.exit(
            1, "voting-nli: error: Invalid model configuration. Set API_KEY, BASE_URL "
            "and MODEL in your environment or .env.\n",
        )
    except (ValueError, OSError, RuntimeError, OpenAIError) as error:
        parser.exit(1, f"voting-nli: error: {error}\n")
    except KeyboardInterrupt:
        parser.exit(130, "\nvoting-nli: interrupted.\n")


if __name__ == "__main__":
    main()
