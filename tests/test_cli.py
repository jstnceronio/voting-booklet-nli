"""Smoke check for the installed command-line entry point."""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from voting_nli.cli import main


class CLITest(unittest.TestCase):
    def test_booklet_input_rejected_before_overwriting_output(self):
        case = {
            "id": "booklet-example",
            "booklet": {"path": "booklets/example.pdf", "language": "de"},
            "claim": {"text": "Eine Behauptung.", "language": "de"},
        }
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "cases.jsonl"
            output_path = Path(directory) / "predictions.jsonl"
            valid_case = {
                "id": "reference-example",
                "reference": {"text": "Reference text."},
                "claim": {"text": "Claim text."},
            }
            input_path.write_text(
                json.dumps(valid_case) + "\n" + json.dumps(case) + "\n", encoding="utf-8"
            )
            output_path.write_text("previous output\n", encoding="utf-8")
            errors = io.StringIO()
            with (
                patch.object(sys, "argv", ["voting-nli", "--input", str(input_path),
                                          "--output", str(output_path)]),
                patch("voting_nli.cli.Model") as model,
                redirect_stderr(errors),
                self.assertRaises(SystemExit) as raised,
            ):
                main()

            self.assertEqual(raised.exception.code, 2)
            self.assertIn("booklet-example", errors.getvalue())
            self.assertIn("line 2", errors.getvalue())
            self.assertIn("Booklet inputs are not supported", errors.getvalue())
            model.assert_not_called()
            self.assertEqual(output_path.read_text(encoding="utf-8"), "previous output\n")

    def test_nested_reference_and_claim(self):
        case = {
            "id": "example",
            "reference": {"text": "The vote was in December.", "language": "en"},
            "claim": {"text": "Die Abstimmung war im Dezember.", "language": "de"},
        }
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "cases.jsonl"
            output_path = Path(directory) / "predictions.jsonl"
            cases = [case, {**case, "id": "second-example"}]
            input_path.write_text(
                "".join(json.dumps(row) + "\n" for row in cases), encoding="utf-8"
            )
            stdout = io.StringIO()
            metrics = [
                {"input_tokens": 120, "output_tokens": 8, "inference_time_ms": 125.5},
                {"input_tokens": 130, "output_tokens": 9, "inference_time_ms": 150.25},
            ]
            with (
                patch.object(sys, "argv", ["voting-nli", "--input", str(input_path),
                                          "--output", str(output_path)]),
                patch("voting_nli.cli.Model") as model,
                redirect_stdout(stdout),
            ):
                model.return_value.get_response.side_effect = [("0", row) for row in metrics]
                main()

            self.assertEqual(stdout.getvalue(), "")
            predictions = [
                json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual([row["id"] for row in predictions], [row["id"] for row in cases])
            self.assertEqual([row["label"] for row in predictions], [0, 0])
            self.assertEqual([row["metrics"] for row in predictions], metrics)
            self.assertEqual(model.return_value.get_response.call_count, 2)
            prompt = model.return_value.get_response.call_args.args[0]
            data = json.loads(prompt.split("Input:", 1)[1].strip())
            self.assertEqual(data, {
                "reference": case["reference"]["text"],
                "claim": case["claim"]["text"],
            })

    def test_help(self):
        result = subprocess.run(
            ["voting-nli", "--help"], capture_output=True, text=True, check=True
        )
        self.assertIn("--version", result.stdout)


if __name__ == "__main__":
    unittest.main()
