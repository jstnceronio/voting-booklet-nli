# voting-nli

GenAI Challenge 2026: multilingual fact-checking of Swiss voting booklets,
followed by evidence-grounded, two-speaker dialogue and audio generation.

## Setup

Requires Python 3.11+ and uv. `.python-version` selects Python 3.11 to match
the course tooling. Run from this directory:

```sh
uv sync --no-editable
uv run --no-editable voting-nli --help
uv run --no-editable voting-nli --version
```

The CLI supports reference NLI and full-document PDF NLI using Apertus.
Evidence extraction, dialogue generation and audio are not implemented yet.
Use `--no-editable` to avoid editable-import issues in restricted environments;
after editing Python source, refresh the installed package with
`uv sync --no-editable --reinstall-package voting-nli`.

Configure `API_KEY`, `BASE_URL` and `MODEL` in your environment or `.env`.
The model client currently uses `swiss-ai/Apertus-v1.5-70B-thinking`.

## Reference NLI

Run a reference-only example with:

```sh
uv run --no-editable voting-nli --input data/examples/reference_cases.jsonl --output runs/predictions.jsonl
```

## Booklet NLI

Run the included German 2026 booklet example with:

```sh
uv run --no-editable voting-nli --input data/examples/booklet_cases.jsonl --output runs/booklet-predictions.jsonl
```

An input JSONL file can mix reference and booklet cases. Each case must contain
exactly one of `reference` or `booklet`, an `id` and a `claim.text`. The optional
`vote` name is passed to the classifier as proposal context. Booklet paths resolve
relative to the input JSONL file; absolute paths also work. For example,
`booklets/example.pdf` in `/data/cases.jsonl` resolves to `/data/booklets/example.pdf`.
Predictions preserve case IDs, labels and per-case metrics. Booklet pages retain
their 1-based PDF page numbers, including gaps for blank pages. Evidence is
currently returned as an empty list, so this baseline does not yet satisfy the
benchmark's Task A evidence requirements.

The direct PDF command is also available:

```sh
uv run --no-editable voting-nli check --pdf data/booklets/2026_06_14_de.pdf --claim "Die Volksabstimmung findet am 14. Juni 2026 statt."
```

## Structure

- `src/voting_nli/`: application code, starting with the CLI.
- `tests/`: automated checks using Python's standard-library unittest.
- `data/booklets/`: original voting booklet PDFs (ignored by Git).
- `data/benchmarks/`: official evaluation inputs (ignored by Git).
- `runs/`: generated predictions, evidence logs, scripts and audio (ignored by Git).
- `report/`: report, architecture diagram and evaluation summaries.

## Reading the code

The batch flow is `main → load_cases → classify_batch → Classifier.classify`.

- `cli.py` handles command arguments and prediction output.
- `cases.py` validates JSONL input and loads reference text or page-marked PDF
  text into `Case` records with named fields.
- `nli.py` builds the classification prompt and validates the returned label.
- `model.py` calls Apertus and records token usage and timing.
- `config.py` loads environment settings.

All input cases and PDFs are loaded before opening the output file, so invalid
inputs cannot truncate existing predictions. Booklet paths are also checked to
prevent overwriting a source PDF.

## Check

```sh
uv run --no-editable python -m unittest discover -s tests -v
```

## Implementation order

1. Reference NLI with Apertus v1.5 and the official challenge I/O adapter.
2. Page-aware PDF extraction, evidence retrieval and booklet NLI.
3. Dialogue generation, factual claim extraction, verification and revision.
4. Two-speaker audio from the final verified script.
5. Evaluation results, report and live demonstration.

Both official NLI tasks and dialogue verification will share the same NLI code.
Add modules as each step is implemented.
