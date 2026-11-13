"""CLI behavior with real input files and a mocked model."""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import pymupdf

from voting_nli.cli import main


class CLITest(unittest.TestCase):
    def test_pdf_check_without_batch_arguments(self):
        with tempfile.TemporaryDirectory() as directory:
            pdf = Path(directory) / "booklet.pdf"
            with pymupdf.open() as document:
                document.new_page().insert_text((72, 72), "First page.")
                document.new_page()
                document.new_page().insert_text((72, 72), "Third page.")
                document.save(pdf)

            stdout = io.StringIO()
            metrics = {"input_tokens": 120, "output_tokens": 1, "inference_time_ms": 10}
            with (
                patch.object(sys, "argv", ["voting-nli", "check", "--pdf", str(pdf),
                                          "--claim", "Eine Behauptung."]),
                patch("voting_nli.cli.Model") as model,
                redirect_stdout(stdout),
            ):
                model.return_value.get_response.return_value = ("0", metrics)
                main()

            self.assertEqual(json.loads(stdout.getvalue()), {
                "label": 0,
                "label_name": "entailment",
                "evidence": [],
                "metrics": metrics,
            })
            prompt = model.return_value.get_response.call_args.args[0]
            data = json.loads(prompt.split("Input:", 1)[1].strip())
            self.assertEqual(data["claim"], "Eine Behauptung.")
            self.assertEqual(
                [line for line in data["reference"].splitlines() if line.strip()],
                ["[Page 1]", "First page.", "[Page 3]", "Third page."],
            )

    def test_batch_requires_both_paths(self):
        for arguments in ([], ["--input", "cases.jsonl"], ["--output", "predictions.jsonl"]):
            with (
                self.subTest(arguments=arguments),
                patch.object(sys, "argv", ["voting-nli", *arguments]),
                patch("voting_nli.cli.Model") as model,
                redirect_stderr(io.StringIO()) as errors,
                self.assertRaises(SystemExit) as raised,
            ):
                main()

            self.assertEqual(raised.exception.code, 2)
            self.assertIn("Batch mode requires --input and --output.", errors.getvalue())
            model.assert_not_called()

    def test_missing_booklet_rejected_before_overwriting_output(self):
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
            self.assertIn("example.pdf", errors.getvalue())
            model.assert_not_called()
            self.assertEqual(output_path.read_text(encoding="utf-8"), "previous output\n")

    def test_mixed_reference_and_booklet_cases(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pdf = root / "booklets" / "example.pdf"
            pdf.parent.mkdir()
            with pymupdf.open() as document:
                document.new_page().insert_text((72, 72), "First page.")
                document.new_page()
                document.new_page().insert_text((72, 72), "Third page.")
                document.save(pdf)
            reference_case = {
                "id": "reference", "reference": {"text": "Reference text.", "language": "fr"},
                "vote": "Example proposal", "claim": {"text": "Eine Behauptung.", "language": "de"},
            }
            booklet_case = {
                "id": "booklet", "booklet": {"path": "booklets/example.pdf", "language": "de"},
                "vote": "Example proposal", "claim": {"text": "Un'affermazione.", "language": "it"},
            }
            cases = [reference_case, booklet_case, {**reference_case, "id": "second-reference"},
                     {**booklet_case, "id": "absolute-booklet", "booklet": {"path": str(pdf)}}]
            input_path = root / "cases.jsonl"
            output_path = root / "predictions.jsonl"
            input_path.write_text("".join(json.dumps(row) + "\n" for row in cases), encoding="utf-8")
            metrics = {"input_tokens": 120, "output_tokens": 1, "inference_time_ms": 10}
            stdout = io.StringIO()
            with (
                patch.object(sys, "argv", ["voting-nli", "--input", str(input_path), "--output", str(output_path)]),
                patch("voting_nli.cli.Model") as model,
                redirect_stdout(stdout),
            ):
                model.return_value.get_response.side_effect = [(str(label), metrics) for label in (1, 0, 2, 0)]
                main()

            self.assertEqual(stdout.getvalue(), "")
            predictions = [json.loads(line) for line in output_path.read_text().splitlines()]
            self.assertEqual([row["id"] for row in predictions], [case["id"] for case in cases])
            self.assertEqual([row["label"] for row in predictions], [1, 0, 2, 0])
            self.assertEqual([row["metrics"] for row in predictions], [metrics] * 4)
            self.assertEqual(model.call_count, 1)
            prompts = [json.loads(call.args[0].split("Input:", 1)[1].strip())
                       for call in model.return_value.get_response.call_args_list]
            self.assertEqual(prompts[0]["reference"], "Reference text.")
            self.assertEqual(prompts[1]["claim"], booklet_case["claim"]["text"])
            self.assertEqual(prompts[1]["vote"], "Example proposal")
            self.assertEqual([line for line in prompts[1]["reference"].splitlines() if line.strip()],
                             ["[Page 1]", "First page.", "[Page 3]", "Third page."])
            self.assertEqual(prompts[1], prompts[3])

    def test_invalid_cases_preserve_existing_output(self):
        valid = {"id": "example", "claim": {"text": "Claim."}, "reference": {"text": "Reference."}}
        invalid_cases = [
            {"id": "missing-source", "claim": valid["claim"]},
            {**valid, "booklet": {"path": "example.pdf"}},
            {"id": "missing-path", "claim": valid["claim"], "booklet": {}},
            {**valid, "reference": {"text": " "}},
            {**valid, "id": " "},
            {**valid, "claim": None},
            {**valid, "vote": 42},
            [],
        ]
        for case in invalid_cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                inputs = root / "cases.jsonl"
                outputs = root / "predictions.jsonl"
                inputs.write_text(json.dumps(case) + "\n", encoding="utf-8")
                outputs.write_text("previous output\n", encoding="utf-8")
                with (
                    patch.object(sys, "argv", ["voting-nli", "--input", str(inputs), "--output", str(outputs)]),
                    patch("voting_nli.cli.Model") as model,
                    redirect_stderr(io.StringIO()),
                    self.assertRaises(SystemExit) as raised,
                ):
                    main()
                self.assertEqual(raised.exception.code, 2)
                model.assert_not_called()
                self.assertEqual(outputs.read_text(), "previous output\n")

    def test_output_cannot_overwrite_booklet(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pdf = root / "booklet.pdf"
            with pymupdf.open() as document:
                document.new_page().insert_text((72, 72), "Keep this source.")
                document.save(pdf)
            original = pdf.read_bytes()
            inputs = root / "cases.jsonl"
            inputs.write_text(json.dumps({"id": "booklet", "booklet": {"path": pdf.name},
                                          "claim": {"text": "Claim."}}) + "\n", encoding="utf-8")
            with (
                patch.object(sys, "argv", ["voting-nli", "--input", str(inputs), "--output", str(pdf)]),
                patch("voting_nli.cli.Model") as model,
                redirect_stderr(io.StringIO()) as errors,
                self.assertRaises(SystemExit),
            ):
                main()
            self.assertIn("must not overwrite", errors.getvalue())
            model.assert_not_called()
            self.assertEqual(pdf.read_bytes(), original)

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
