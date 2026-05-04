#!/usr/bin/env python3
"""Merge LLM runs across multiple batch roots into a single output root.

Each input root should contain per-slug directories with llm_reply*.json files.
The script concatenates the "runs" arrays and reindexes run numbers.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List


def iter_roots(roots: List[Path]) -> List[Path]:
    """Return list of valid directories from roots."""
    return [p for p in roots if p.is_dir()]


def merge_runs(input_roots: List[Path], output_root: Path) -> None:
    """Merge llm_reply*.json files across input roots."""
    output_root.mkdir(parents=True, exist_ok=True)
    
    for root in input_roots:
        for slug_dir in root.iterdir():
            if not slug_dir.is_dir():
                continue
            out_slug = output_root / slug_dir.name
            out_slug.mkdir(parents=True, exist_ok=True)
            
            # Copy non-reply files (legend.txt, sentence.txt, codebook.json, metadata.json)
            for f in ["legend.txt", "sentence.txt", "codebook.json", "metadata.json"]:
                src = slug_dir / f
                dst = out_slug / f
                if src.exists() and not dst.exists():
                    dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
            
            # Merge reply files
            for reply_file in slug_dir.glob("llm_reply*.json"):
                data = json.loads(reply_file.read_text())
                out_file = out_slug / reply_file.name
                if out_file.exists():
                    merged = json.loads(out_file.read_text())
                else:
                    merged = {k: v for k, v in data.items() if k != "runs"}
                    merged["runs"] = []
                merged["runs"].extend(data.get("runs", []))
                out_file.write_text(json.dumps(merged, indent=2), encoding="utf-8")

    # Reindex run numbers per file for best-of-k analysis
    for out_slug in output_root.iterdir():
        if not out_slug.is_dir():
            continue
        for reply_file in out_slug.glob("llm_reply*.json"):
            data = json.loads(reply_file.read_text())
            runs = data.get("runs", []) or []
            for idx, run in enumerate(runs, 1):
                if isinstance(run, dict):
                    run["run"] = idx
            data["runs"] = runs
            reply_file.write_text(json.dumps(data, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge llm_reply*.json runs across multiple batch roots into one."
    )
    parser.add_argument(
        "--input-roots",
        type=str,
        default="",
        help="Comma-separated list of batch roots (e.g., results/attacks/textual_cipher_a,results/attacks/textual_cipher_b).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results/attacks/textual_cipher"),
        help="Where to write merged per-slug folders.",
    )
    args = parser.parse_args()

    roots: List[Path] = []
    if args.input_roots:
        roots = [Path(p.strip()) for p in args.input_roots.split(",") if p.strip()]
    
    input_roots = iter_roots(roots)
    if not input_roots:
        raise SystemExit("No input roots found (check --input-roots).")

    merge_runs(input_roots, args.output_root)
    print(f"Merged {len(input_roots)} roots into {args.output_root}")


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        sys.exit(0)
