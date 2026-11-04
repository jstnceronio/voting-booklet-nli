"""Smoke check for the installed command-line entry point."""

import subprocess
import unittest


class CLITest(unittest.TestCase):
    def test_help(self):
        result = subprocess.run(
            ["voting-nli", "--help"], capture_output=True, text=True, check=True
        )
        self.assertIn("--version", result.stdout)


if __name__ == "__main__":
    unittest.main()
