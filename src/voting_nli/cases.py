from dataclasses import dataclass
import json
from pathlib import Path

import pymupdf


@dataclass
class Case:
    case_id: str
    reference: str
    claim: str
    vote: str | None = None


def required_text(data: object, field: str) -> str:
    value = data.get(field) if isinstance(data, dict) else None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Expected a non-empty string for '{field}'.")
    return value


def read_booklet(pdf_path: Path) -> str:
    with pymupdf.open(pdf_path) as document:
        pages = [page.get_text(sort=True) for page in document]

    if not any(text.strip() for text in pages):
        raise ValueError("PDF contains no extractable text.")

    return "\n\n".join(
        f"[Page {number}]\n{text}"
        for number, text in enumerate(pages, start=1)
        if text.strip()
    )


def load_cases(input_path: Path, output_path: Path) -> list[Case]:
    """Validate and load every source before the output file is opened."""
    cases = []
    with input_path.open(encoding="utf-8") as inputs:
        for line in inputs:
            data = json.loads(line)
            case_id = required_text(data, "id")
            claim = required_text(data.get("claim"), "text")
            vote = required_text(data, "vote") if "vote" in data else None

            source_fields = data.keys() & {"reference", "booklet"}
            if len(source_fields) != 1:
                raise ValueError("Provide exactly one of 'reference' or 'booklet'.")

            if "booklet" in data:
                pdf_path = input_path.parent / required_text(data["booklet"], "path")
                if pdf_path.resolve() == output_path.resolve():
                    raise ValueError("Output must not overwrite a source booklet.")
                reference = read_booklet(pdf_path)
            else:
                reference = required_text(data["reference"], "text")

            cases.append(Case(
                case_id=case_id,
                reference=reference,
                claim=claim,
                vote=vote,
            ))
    return cases
