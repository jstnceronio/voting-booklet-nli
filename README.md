# voting-nli

GenAI Challenge 2026: multilingual fact-checking of Swiss voting booklets,
followed by evidence-grounded, two-speaker dialogue and audio generation.

## Setup

Requires Python 3.11+ and uv. `.python-version` selects Python 3.11 to match
the course tooling. Run from this directory:

```sh
uv sync --locked --no-editable
uv run --no-editable voting-nli
uv run --no-editable voting-nli --version
```

Running without arguments shows help. Use `batch --help` or `check --help` for
command options. Help and version work without API credentials.
These examples use a non-editable install because editable imports can fail in
restricted environments. See Development below for running source edits directly.

The CLI supports reference NLI and full-document PDF NLI with source-grounded
evidence using Apertus. Dialogue generation and audio are not implemented yet.

For classification, copy `.env.example` to `.env` and configure `API_KEY`,
`BASE_URL` and `MODEL`, or set those environment variables. For Apertus, use
`swiss-ai/Apertus-v1.5-70B-thinking` as the model. Configuration is loaded when a
model client is created, so importing the package does not require credentials.

## Commands

| Command | Input | Output |
| --- | --- | --- |
| `voting-nli batch -i cases.jsonl -o predictions.jsonl` | Reference or booklet cases in JSONL | Predictions in a JSONL file |
| `voting-nli check --pdf booklet.pdf --claim "Claim text"` | One PDF and one claim | A JSON prediction on stdout |

Batch progress and completion messages go to stderr, keeping stdout available
for machine-readable output. Output directories are created automatically.
Existing predictions are replaced only after the whole batch succeeds; failures
and Ctrl+C preserve them. Input errors include the JSONL filename and line number.

Exit codes are `0` for success or help, `1` for input, configuration, model or
file errors, `2` for invalid command arguments, and `130` for Ctrl+C during a command.

**Syntax change:** batch processing now requires the `batch` command. Update
existing scripts from `voting-nli --input ... --output ...` to
`voting-nli batch --input ... --output ...`.

## Reference NLI

Run a reference-only example with:

```sh
uv run --no-editable voting-nli batch --input data/examples/reference_cases.jsonl --output runs/predictions.jsonl
```

## Booklet NLI

Run the included German 2026 booklet example with:

```sh
uv run --no-editable voting-nli batch --input data/examples/booklet_cases.jsonl --output runs/booklet-predictions.jsonl
```

An input JSONL file can mix reference and booklet cases. Each case must contain
exactly one of `reference` or `booklet`, an `id` and a `claim.text`. The optional
`vote` name is passed to the classifier as proposal context. Booklet paths resolve
relative to the input JSONL file; absolute paths also work. For example,
`booklets/example.pdf` in `/data/cases.jsonl` resolves to `/data/booklets/example.pdf`.
Predictions preserve case IDs, labels and per-case metrics. Booklet pages retain
their 1-based PDF page numbers, including gaps for blank pages.

For booklet entailment (`0`) and contradiction (`2`), a second Apertus call selects
1–5 source passages. Python resolves the selected passage IDs to verbatim text
and 1-based PDF page numbers; the model does not generate quotations. Passages
stay within a page and contain at most 5,000 characters. Neutral and reference-only
predictions keep `evidence: []`. An evidence item has this shape:

```json
{"page": 3, "text": "Verbatim text extracted from this PDF page."}
```

JSON selections may be plain or wrapped in a Markdown code block; ID strings
and objects containing only an `id` are accepted. Invalid or empty selections
get one repair attempt with the rejected response and validation
error. A second failure reports the validation error and a preview of the last
response.
A failure discards the temporary batch output and preserves existing predictions.
Per-case metrics include classification, evidence selection and any repair call;
unavailable token counts remain `null`. The additional calls increase cost and
latency. Source membership guarantees quotation provenance, but relevance and
coverage still require evaluation. Textless PDFs need OCR before use.

Model calls use a 180-second timeout with automatic API retries disabled.
Progress messages go to standard error; final JSON remains on standard output.

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

The batch flow is `main → load_cases → classify_batch → Classifier.classify_with_evidence`.

- `cli.py` handles command arguments, command dispatch and prediction output.
  `build_parser()` defines the interface; `main(argv)` also accepts argument lists
  directly for tests and embedded use.
- `cases.py` validates JSONL input and loads reference text or page-marked PDF
  text into `Case` records, retaining booklet passages for evidence selection.
  `read_booklet` returns a `Booklet` with `reference` and `passages` fields.
- `nli.py` classifies claims, selects and validates evidence IDs, and resolves
  quotations from stored passages. `Classifier.classify` remains label-only.
- `model.py` calls Apertus and records token usage and timing.
- `config.py` defines environment settings, loaded by `Model` when needed.

All input cases and PDFs are loaded before opening the output file, so invalid
inputs cannot truncate existing predictions. Booklet paths are also checked to
prevent overwriting a source PDF.

## Development

The console command and `python -m voting_nli` run the same CLI. To pick up source
changes immediately without reinstalling, run from the project directory with
`PYTHONPATH=src`:

```sh
PYTHONPATH=src uv run --no-editable python -m voting_nli --help
```

Run the tests with placeholder configuration; model calls are mocked, so no API
access or real credentials are needed:

```sh
API_KEY=test BASE_URL=https://example.invalid MODEL=test PYTHONPATH=src \
  uv run --no-editable python -m unittest discover -s tests -v
```

To refresh the installed console command after source changes, run
`uv sync --no-editable --reinstall-package voting-nli`. In environments supporting
editable imports, `uv sync` followed by `uv run voting-nli ...` also picks up edits
automatically.

## Implementation order

1. Reference NLI with Apertus v1.5 and the official challenge I/O adapter.
2. Page-aware PDF extraction, evidence retrieval and booklet NLI.
3. Dialogue generation, factual claim extraction, verification and revision.
4. Two-speaker audio from the final verified script.
5. Evaluation results, report and live demonstration.

Both official NLI tasks and dialogue verification will share the same NLI code.
Add modules as each step is implemented.
