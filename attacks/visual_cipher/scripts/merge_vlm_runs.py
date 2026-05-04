#!/usr/bin/env python3
"""Merge VLM runs across multiple batch roots into a single output root.

Each input root should contain per-slug directories with vlm_reply*.json files.
The script concatenates the "runs" arrays and reindexes run numbers.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Iterable, List


def iter_roots(roots: List[Path], glob_pattern: str | None) -> List[Path]:
    if glob_pattern:
        expanded = sorted(Path().glob(glob_pattern))
        return [p for p in expanded if p.is_dir()]
    return [p for p in roots if p.is_dir()]


def merge_runs(input_roots: Iterable[Path], output_root: Path) -> None:
    roots_list = list(input_roots)
    output_root.mkdir(parents=True, exist_ok=True)
    for root in roots_list:
        for slug_dir in root.iterdir():
            if not slug_dir.is_dir():
                continue
            out_slug = output_root / slug_dir.name
            out_slug.mkdir(parents=True, exist_ok=True)
            for reply_file in slug_dir.glob("vlm_reply*.json"):
                data = json.loads(reply_file.read_text())
                out_file = out_slug / reply_file.name
                if out_file.exists():
                    merged = json.loads(out_file.read_text())
                else:
                    merged = {k: v for k, v in data.items() if k != "runs"}
                    merged["runs"] = []
                merged["runs"].extend(data.get("runs", []))
                out_file.write_text(json.dumps(merged, indent=2), encoding="utf-8")

    # Reindex run numbers per file for best-of-k plots
    for out_slug in output_root.iterdir():
        if not out_slug.is_dir():
            continue
        for reply_file in out_slug.glob("vlm_reply*.json"):
            data = json.loads(reply_file.read_text())
            runs = data.get("runs", []) or []
            for idx, run in enumerate(runs, 1):
                if isinstance(run, dict):
                    run["run"] = idx
            data["runs"] = runs
            reply_file.write_text(json.dumps(data, indent=2), encoding="utf-8")

    # Copy metadata.json from first root that has it (for evals/judge_attacks.py).
    for out_slug in output_root.iterdir():
        if not out_slug.is_dir():
            continue
        meta_dst = out_slug / "metadata.json"
        if meta_dst.exists():
            continue
        for root in roots_list:
            src = root / out_slug.name / "metadata.json"
            if src.exists():
                shutil.copy2(src, meta_dst)
                break


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge vlm_reply*.json runs across multiple batch roots into one."
    )
    parser.add_argument(
        "--input-roots",
        type=str,
        default="",
        help="Comma-separated list of batch roots (e.g., outputs/batch_a,outputs/batch_b).",
    )
    parser.add_argument(
        "--input-glob",
        type=str,
        default="",
        help="Glob for batch roots (e.g., outputs/batch_harmbench_seed*).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results/attacks/visual_cipher/batch_merged"),
        help="Where to write merged per-slug folders.",
    )
    args = parser.parse_args()

    roots: List[Path] = []
    if args.input_roots:
        roots = [Path(p.strip()) for p in args.input_roots.split(",") if p.strip()]
    input_roots = iter_roots(roots, args.input_glob or None)
    if not input_roots:
        raise SystemExit("No input roots found (check --input-roots or --input-glob).")

    merge_runs(input_roots, args.output_root)
    print(f"Merged {len(input_roots)} roots into {args.output_root}")


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        sys.exit(0)
