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

This scaffold provides the package and CLI entry point. NLI, model calls,
PDF processing, dialogue generation and audio are not implemented yet.
No runtime dependencies or API credentials are needed for the scaffold.
Use `--no-editable` to avoid editable-import issues in restricted environments;
uv rebuilds the installed package when the source changes.

When adding Apertus access, copy `.env.example` to `.env` and set your
`CSCS_API_KEY`. Environment-file loading will be added with the model client.

## Structure

- `src/voting_nli/`: application code, starting with the CLI.
- `tests/`: automated checks using Python's standard-library unittest.
- `data/booklets/`: original voting booklet PDFs (ignored by Git).
- `data/benchmarks/`: official evaluation inputs (ignored by Git).
- `runs/`: generated predictions, evidence logs, scripts and audio (ignored by Git).
- `report/`: report, architecture diagram and evaluation summaries.

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
