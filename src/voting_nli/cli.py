"""Command-line entry point for voting-nli."""

import argparse
from importlib.metadata import version


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multilingual voting booklet NLI and verified dialogue generation."
    )
    parser.add_argument("--version", action="version", version=version("voting-nli"))
    parser.parse_args()
    parser.print_help()


if __name__ == "__main__":
    main()
