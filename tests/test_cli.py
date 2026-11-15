"""CLI behavior with real input files and a mocked model."""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import pymupdf
from openai import APIConnectionError

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
                patch("voting_nli.cli.Model") as model,
                redirect_stdout(stdout),
            ):
                model.return_value.get_response.return_value = ("0", metrics)
                main(["check", "--pdf", str(pdf), "--claim", "Eine Behauptung."])

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
                patch("voting_nli.cli.Model") as model,
                redirect_stderr(io.StringIO()) as errors,
                self.assertRaises(SystemExit) as raised,
            ):
                main(["batch", *arguments])

            self.assertEqual(raised.exception.code, 2)
            self.assertIn("required", errors.getvalue())
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
                patch("voting_nli.cli.Model") as model,
                redirect_stderr(errors),
                self.assertRaises(SystemExit) as raised,
            ):
                main(["batch", "--input", str(input_path), "--output", str(output_path)])

            self.assertEqual(raised.exception.code, 1)
            self.assertIn("example.pdf", errors.getvalue())
            self.assertIn(f"{input_path}:2:", errors.getvalue())
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
                patch("voting_nli.cli.Model") as model,
                redirect_stdout(stdout),
                redirect_stderr(io.StringIO()) as messages,
            ):
                model.return_value.get_response.side_effect = [(str(label), metrics) for label in (1, 0, 2, 0)]
                main(["batch", "--input", str(input_path), "--output", str(output_path)])

            self.assertEqual(stdout.getvalue(), "")
            self.assertIn(f"Wrote 4 predictions to {output_path}.", messages.getvalue())
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
                    patch("voting_nli.cli.Model") as model,
                    redirect_stderr(io.StringIO()),
                    self.assertRaises(SystemExit) as raised,
                ):
                    main(["batch", "--input", str(inputs), "--output", str(outputs)])
                self.assertEqual(raised.exception.code, 1)
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
                patch("voting_nli.cli.Model") as model,
                redirect_stderr(io.StringIO()) as errors,
                self.assertRaises(SystemExit),
            ):
                main(["batch", "--input", str(inputs), "--output", str(pdf)])
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
                patch("voting_nli.cli.Model") as model,
                redirect_stdout(stdout),
                redirect_stderr(io.StringIO()),
            ):
                model.return_value.get_response.side_effect = [("0", row) for row in metrics]
                main(["batch", "--input", str(input_path), "--output", str(output_path)])

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
        source = Path(__file__).resolve().parents[1] / "src"
        environment = {
            key: value for key, value in os.environ.items()
            if key not in {"API_KEY", "BASE_URL", "MODEL"}
        }
        environment["PYTHONPATH"] = str(source)
        for arguments, expected in (
            ([], "commands:"),
            (["--help"], "Examples:"),
            (["--version"], "voting-nli 0.1.0"),
            (["batch", "--help"], "--input"),
            (["check", "--help"], "--claim"),
        ):
            with self.subTest(arguments=arguments), tempfile.TemporaryDirectory() as directory:
                result = subprocess.run(
                    [sys.executable, "-m", "voting_nli", *arguments],
                    cwd=directory, env=environment, capture_output=True, text=True,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn(expected, result.stdout)
                self.assertEqual(result.stderr, "")

    def test_invalid_command_arguments_do_not_start_model(self):
        for arguments in (
            ["unknown"],
            ["--input", "cases.jsonl", "--output", "predictions.jsonl"],
            ["check"],
            ["check", "--pdf", "booklet.pdf"],
            ["check", "--pdf", "booklet.pdf", "--claim", "Claim.", "--input", "cases.jsonl"],
            ["--input", "cases.jsonl", "check", "--pdf", "booklet.pdf", "--claim", "Claim."],
        ):
            with (
                self.subTest(arguments=arguments),
                patch("voting_nli.cli.Model") as model,
                redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit) as raised,
            ):
                main(arguments)
            self.assertEqual(raised.exception.code, 2)
            model.assert_not_called()

    def test_batch_cannot_overwrite_input(self):
        with (
            patch("voting_nli.cli.Model") as model,
            redirect_stderr(io.StringIO()) as errors,
            self.assertRaises(SystemExit) as raised,
        ):
            main(["batch", "-i", "cases.jsonl", "-o", "./cases.jsonl"])
        self.assertEqual(raised.exception.code, 2)
        self.assertIn("different files", errors.getvalue())
        model.assert_not_called()

    def test_malformed_or_empty_input_preserves_output(self):
        valid = {"id": "example", "claim": {"text": "Claim."}, "reference": {"text": "Reference."}}
        for contents, expected in (("", "contains no cases"), (json.dumps(valid) + "\n{", ":2:")):
            with self.subTest(contents=contents), tempfile.TemporaryDirectory() as directory:
                inputs = Path(directory) / "cases.jsonl"
                outputs = Path(directory) / "predictions.jsonl"
                inputs.write_text(contents, encoding="utf-8")
                outputs.write_text("previous output\n", encoding="utf-8")
                with (
                    patch("voting_nli.cli.Model") as model,
                    redirect_stderr(io.StringIO()) as errors,
                    self.assertRaises(SystemExit) as raised,
                ):
                    main(["batch", "-i", str(inputs), "-o", str(outputs)])
                self.assertEqual(raised.exception.code, 1)
                self.assertIn(expected, errors.getvalue())
                self.assertEqual(outputs.read_text(), "previous output\n")
                model.assert_not_called()

    def test_failed_or_interrupted_batch_preserves_output_and_cleans_up(self):
        failures = [
            ValueError("Invalid NLI response: 'unknown'"),
            APIConnectionError(request=Mock()),
            OSError("Cannot write predictions"),
            KeyboardInterrupt(),
        ]
        for failure in failures:
            with self.subTest(failure=type(failure).__name__), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                inputs = root / "cases.jsonl"
                outputs = root / "predictions.jsonl"
                cases = [{"id": str(i), "claim": {"text": "Claim."},
                          "reference": {"text": "Reference."}} for i in range(2)]
                inputs.write_text("".join(json.dumps(case) + "\n" for case in cases), encoding="utf-8")
                outputs.write_text("previous output\n", encoding="utf-8")
                with (
                    patch("voting_nli.cli.Model") as model,
                    redirect_stderr(io.StringIO()) as errors,
                    redirect_stdout(io.StringIO()) as stdout,
                    self.assertRaises(SystemExit) as raised,
                ):
                    model.return_value.get_response.side_effect = [("0", {}), failure]
                    main(["batch", "-i", str(inputs), "-o", str(outputs)])
                self.assertEqual(raised.exception.code, 130 if isinstance(failure, KeyboardInterrupt) else 1)
                self.assertEqual(outputs.read_text(), "previous output\n")
                self.assertEqual(set(root.iterdir()), {inputs, outputs})
                self.assertEqual(stdout.getvalue(), "")
                self.assertNotIn("Traceback", errors.getvalue())
                self.assertNotIn("Wrote", errors.getvalue())

    def test_check_failure_prints_error_without_json(self):
        for failure in (ValueError("Invalid NLI response"), RuntimeError("Cannot read PDF")):
            with (
                self.subTest(failure=failure),
                patch("voting_nli.cli.check_booklet", side_effect=failure),
                redirect_stderr(io.StringIO()) as errors,
                redirect_stdout(io.StringIO()) as stdout,
                self.assertRaises(SystemExit) as raised,
            ):
                main(["check", "--pdf", "booklet.pdf", "--claim", "Claim."])
            self.assertEqual(raised.exception.code, 1)
            self.assertIn(str(failure), errors.getvalue())
            self.assertEqual(stdout.getvalue(), "")

    def test_missing_configuration_prints_actionable_error_without_values(self):
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("voting_nli.model.Settings.model_config", {"env_file": None}),
            patch("voting_nli.cli.read_booklet", return_value="Reference."),
            redirect_stderr(io.StringIO()) as errors,
            self.assertRaises(SystemExit) as raised,
        ):
            main(["check", "--pdf", "booklet.pdf", "--claim", "Claim."])
        self.assertEqual(raised.exception.code, 1)
        self.assertIn("Set API_KEY, BASE_URL and MODEL", errors.getvalue())
        self.assertNotIn("Traceback", errors.getvalue())

    def test_configuration_errors_do_not_expose_credentials(self):
        from voting_nli.config import Settings
        from pydantic import ValidationError

        with self.assertRaises(ValidationError) as invalid:
            Settings(_env_file=None, api_key="private-api-key", base_url="url", model="model",
                     unexpected="private-api-key")
        with (
            patch("voting_nli.cli.Model", side_effect=invalid.exception),
            patch("voting_nli.cli.read_booklet", return_value="Reference."),
            redirect_stderr(io.StringIO()) as errors,
            self.assertRaises(SystemExit) as raised,
        ):
            main(["check", "--pdf", "booklet.pdf", "--claim", "Claim."])
        self.assertEqual(raised.exception.code, 1)
        self.assertNotIn("private-api-key", errors.getvalue())


if __name__ == "__main__":
    unittest.main()
