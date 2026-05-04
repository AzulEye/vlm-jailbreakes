#!/usr/bin/env python3
"""Wrapper for the textual cipher attack pipeline."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_ROOT = REPO_ROOT / "results" / "attacks" / "textual_cipher"
DEFAULT_CSV = SCRIPT_DIR / "harmbench_behaviors_text_test.csv"


def _flag_present(args: List[str], flag: str) -> bool:
    return flag in args


def _run(cmd: List[str]) -> int:
    return subprocess.call(cmd)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the textual cipher attack pipeline with standardized outputs."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    gen = subparsers.add_parser("generate", help="Batch-generate textual cipher legends and sentences from a CSV.")
    gen.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    gen.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)

    run = subparsers.add_parser("run", help="Run LLM validation over batch outputs.")
    run.add_argument("--batch-root", type=Path, default=DEFAULT_ROOT)
    run.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    run.add_argument(
        "--no-save-output",
        action="store_true",
        help="Do not write llm_reply*.json outputs (wrapper default is to save).",
    )

    args, rest = parser.parse_known_args()

    if args.command == "generate":
        cmd = [sys.executable, str(SCRIPT_DIR / "batch_generate.py")]
        if not _flag_present(rest, "--csv"):
            cmd += ["--csv", str(args.csv)]
        if not _flag_present(rest, "--output-root"):
            cmd += ["--output-root", str(args.output_root)]
        cmd += rest
        sys.exit(_run(cmd))

    if args.command == "run":
        cmd = [sys.executable, str(SCRIPT_DIR / "llm_validator.py")]
        if not _flag_present(rest, "--batch-root"):
            cmd += ["--batch-root", str(args.batch_root)]
        if not _flag_present(rest, "--csv"):
            cmd += ["--csv", str(args.csv)]
        if not args.no_save_output and not _flag_present(rest, "--save-output"):
            cmd.append("--save-output")
        cmd += rest
        sys.exit(_run(cmd))


if __name__ == "__main__":
    main()
