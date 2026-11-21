from dataclasses import dataclass
import json
from pathlib import Path

import pymupdf


@dataclass(frozen=True)
class Passage:
    id: str
    page: int
    text: str


@dataclass
class Booklet:
    reference: str
    passages: list[Passage]


@dataclass
class Case:
    case_id: str
    reference: str
    claim: str
    vote: str | None = None
    passages: list[Passage] | None = None


def required_text(data: object, field: str) -> str:
    value = data.get(field) if isinstance(data, dict) else None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Expected a non-empty string for '{field}'.")
    return value


def read_booklet(pdf_path: Path) -> Booklet:
    with pymupdf.open(pdf_path) as document:
        pages = [page.get_text(sort=True) for page in document]

    if not any(text.strip() for text in pages):
        raise ValueError("PDF contains no extractable text.")

    reference = "\n\n".join(
        f"[Page {number}]\n{text}"
        for number, text in enumerate(pages, start=1)
        if text.strip()
    )
    passages = []
    # Split each page separately so evidence keeps its original PDF page number.
    for number, text in enumerate(pages, start=1):
        if not text.strip():
            continue
        start = 0
        while start < len(text):
            end = min(start + 5_000, len(text))
            if end < len(text):
                # Prefer the last paragraph break within the limit; otherwise use the character cutoff.
                boundary = text.rfind("\n\n", start, end)
                if boundary > start:
                    # Keep the separator so joining the passages reproduces the extracted page text.
                    end = boundary + 2
            # Page number and character offset identify the exact source passage.
            passages.append(Passage(f"p{number}:{start}", number, text[start:end]))
            start = end
    return Booklet(reference, passages)


def load_cases(input_path: Path, output_path: Path) -> list[Case]:
    """Validate and load every source before the output file is opened."""
    cases = []
    with input_path.open(encoding="utf-8") as inputs:
        for line_number, line in enumerate(inputs, start=1):
            try:
                data = json.loads(line)
                case_id = required_text(data, "id")
                claim = required_text(data.get("claim"), "text")
                vote = required_text(data, "vote") if "vote" in data else None

                source_fields = data.keys() & {"reference", "booklet"}
                if len(source_fields) != 1:
                    raise ValueError("Provide exactly one of 'reference' or 'booklet'.")

                passages = None
                if "booklet" in data:
                    pdf_path = input_path.parent / required_text(data["booklet"], "path")
                    if pdf_path.resolve() == output_path.resolve():
                        raise ValueError("Output must not overwrite a source booklet.")
                    booklet = read_booklet(pdf_path)
                    reference, passages = booklet.reference, booklet.passages
                else:
                    reference = required_text(data["reference"], "text")

                cases.append(Case(
                    case_id=case_id,
                    reference=reference,
                    claim=claim,
                    vote=vote,
                    passages=passages,
                ))
            except (ValueError, OSError, RuntimeError) as error:
                raise ValueError(f"{input_path}:{line_number}: {error}") from error
    if not cases:
        raise ValueError(f"{input_path}: Input file contains no cases.")
    return cases
