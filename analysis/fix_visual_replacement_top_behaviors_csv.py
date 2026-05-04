#!/usr/bin/env python3
"""
Fix schema/content issues in visual replacement "top behaviors" CSVs.

Fixes:
- Drop non-local image path columns (`images_json`, `images_by_slot_json`).
- Recompute `slot_replacements_json` per-row from `images_by_slot_local_json`,
  and include only slots that appear in the row.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple


REPL_RE = re.compile(r"/replace_with_object/([^/]+)/")


def infer_slot_replacement(image_paths: List[str]) -> str:
    repls: List[str] = []
    for p in image_paths:
        m = REPL_RE.search(p)
        if m:
            repls.append(m.group(1))
    if not repls:
        return ""
    return Counter(repls).most_common(1)[0][0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Input visual_replacement_top_behaviors_*.csv")
    ap.add_argument("--in-place", action="store_true", help="Overwrite the input CSV.")
    ap.add_argument("--out", default="", help="Output CSV path (ignored with --in-place).")
    args = ap.parse_args()

    in_path = Path(args.csv)
    if not in_path.exists():
        raise SystemExit(f"CSV not found: {in_path}")

    out_path = Path(args.out) if args.out else in_path.with_suffix(".fixed.csv")
    if args.in_place:
        out_path = in_path.with_suffix(".tmp.csv")

    with in_path.open(newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    keep_cols = [
        "attack_id",
        "behavior_id",
        "results_dir",
        "vlm_model",
        "run",
        "behavior",
        "neutralized_prompt",
        "attack_prompt",
        "images_local_json",
        "images_by_slot_local_json",
        "slot_values_json",
        "slot_replacements_json",
        "judge_agg_classification",
        "judge_avg_safety_score",
        "judge_max_safety_score",
        "judge_class_votes_json",
        "judge_per_model_json",
    ]

    missing = [c for c in keep_cols if c not in (rows[0].keys() if rows else [])]
    if missing:
        raise SystemExit(f"Missing expected columns: {missing}")

    changed_replacements = 0
    dropped_nonlocal = 0

    fixed_rows: List[Dict[str, str]] = []
    for row in rows:
        images_by_slot_local = json.loads(row.get("images_by_slot_local_json", "{}") or "{}")
        if not isinstance(images_by_slot_local, dict):
            images_by_slot_local = {}

        slot_repl: Dict[str, str] = {}
        for slot, paths in images_by_slot_local.items():
            if not isinstance(paths, list):
                continue
            slot_repl[str(slot)] = infer_slot_replacement([str(p) for p in paths])

        prev = row.get("slot_replacements_json", "")
        prev_obj = {}
        try:
            prev_obj = json.loads(prev) if prev else {}
        except Exception:
            prev_obj = {}
        if prev_obj != slot_repl:
            changed_replacements += 1

        out_row: Dict[str, str] = {}
        for c in keep_cols:
            if c == "slot_replacements_json":
                out_row[c] = json.dumps(slot_repl, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
            else:
                out_row[c] = row.get(c, "")

        # Track whether non-local columns existed in the input (for sanity reporting).
        if "images_json" in row or "images_by_slot_json" in row:
            dropped_nonlocal += 1

        fixed_rows.append(out_row)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keep_cols, quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        writer.writerows(fixed_rows)

    if args.in_place:
        out_path.replace(in_path)

    print(
        json.dumps(
            {
                "rows": len(rows),
                "changed_slot_replacements_rows": changed_replacements,
                "dropped_nonlocal_rows": dropped_nonlocal,
                "output": str(in_path if args.in_place else out_path),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

