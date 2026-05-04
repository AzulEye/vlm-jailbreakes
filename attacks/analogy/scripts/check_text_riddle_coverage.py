#!/usr/bin/env python3
"""
Offline utility: scan a text-riddle results root and report targets whose riddle options
are missing or fewer than expected.

Default layout:
  <text_results_root>/<category>/<target>/text_riddle_options.json

Example:
  python attacks/analogy/check_text_riddle_coverage.py \
    --text-results-root results_textriddle_v1/attacks/analogy \
    --expected 3
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _norm_cat(x: str) -> str:
    s = (x or "").strip().lower()
    s = s.replace("_", " ").replace("-", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def main() -> None:
    p = argparse.ArgumentParser(description="Report text riddle option coverage (offline).")
    p.add_argument(
        "--text-results-root",
        type=str,
        default=str(Path(__file__).resolve().parents[2] / "results_textriddle_v1" / "attacks" / "analogy"),
        help="Root containing <category>/<target>/text_riddle_options.json",
    )
    p.add_argument("--expected", type=int, default=3, help="Expected number of riddle options per target.")
    p.add_argument(
        "--only-categories",
        nargs="*",
        default=[],
        help="Only scan these categories (case-insensitive; spaces/_/- treated as equal).",
    )
    p.add_argument("--as-json", action="store_true", help="Print machine-readable JSON instead of text.")
    args = p.parse_args()

    root = Path(args.text_results_root).resolve()
    if not root.exists():
        raise SystemExit(f"not found: {root}")

    only_cats = {_norm_cat(c) for c in (args.only_categories or []) if str(c).strip()}

    missing: List[Tuple[str, str]] = []
    too_few: List[Dict[str, Any]] = []
    ok = 0

    for cat_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        cat = cat_dir.name
        if only_cats and _norm_cat(cat) not in only_cats:
            continue
        for tgt_dir in sorted([p for p in cat_dir.iterdir() if p.is_dir()]):
            opts_path = tgt_dir / "text_riddle_options.json"
            if not opts_path.exists():
                missing.append((cat, tgt_dir.name))
                continue
            try:
                data = _load_json(opts_path)
                riddles = data.get("riddles") or []
                n = len(riddles) if isinstance(riddles, list) else 0
            except Exception as exc:  # noqa: BLE001
                too_few.append(
                    {
                        "category": cat,
                        "target": tgt_dir.name,
                        "count": 0,
                        "expected": int(args.expected),
                        "status": "error",
                        "error": str(exc),
                        "path": str(opts_path),
                    }
                )
                continue
            if n >= int(args.expected):
                ok += 1
            else:
                too_few.append(
                    {
                        "category": cat,
                        "target": tgt_dir.name,
                        "count": n,
                        "expected": int(args.expected),
                        "status": "too_few",
                        "path": str(opts_path),
                        "raw": data.get("raw"),
                    }
                )

    if args.as_json:
        print(
            json.dumps(
                {"root": str(root), "expected": int(args.expected), "ok": ok, "missing": missing, "too_few": too_few},
                indent=2,
                ensure_ascii=False,
            )
        )
        return

    print(f"text_results_root={root}")
    print(f"expected={int(args.expected)} ok={ok} missing={len(missing)} too_few={len(too_few)}")
    if missing:
        print("\n[missing text_riddle_options.json]")
        for cat, tgt in missing:
            print(f"- {cat}/{tgt}")
    if too_few:
        print("\n[too few options]")
        for rec in too_few:
            print(f"- {rec['category']}/{rec['target']}: {rec['count']}/{rec['expected']}")


if __name__ == "__main__":
    main()

