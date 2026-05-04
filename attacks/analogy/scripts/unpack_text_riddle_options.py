#!/usr/bin/env python3
"""
Offline utility: unpack "packed" riddle options that were accidentally concatenated into
one string in text_riddle_options.json.

Problem pattern:
  "riddles": ["... -> ? ... -> ? ... -> ?"]   # multiple riddles jammed together

This script:
- Scans <results_root>/**/text_riddle_options.json
- If it detects packed content, splits it into N options (default: 3)
- Writes back the updated JSON (and saves a .bak backup once)

Example:
  python MARS4-Gandelsman/attacks/analogy/unpack_text_riddle_options.py \
    --results-root ./results_textriddle_v3 \
    --expected 3
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple


ARROW_Q = re.compile(r"(?:->|→)\s*\?")


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _split_packed_string(s: str) -> List[str]:
    """
    Split a packed string into multiple riddle options.
    Strategy:
    - Split by blank lines into blocks
    - For each block, if it contains multiple '-> ?' markers, split into segments ending at each '?'
    """
    s = (s or "").strip()
    if not s:
        return []
    # quick exit
    if len(ARROW_Q.findall(s)) <= 1:
        return [s]

    blocks = [b.strip() for b in re.split(r"\n\s*\n+", s) if b.strip()]
    out: List[str] = []

    for b in blocks:
        if len(ARROW_Q.findall(b)) <= 1:
            out.append(b)
            continue
        lines = [ln.strip() for ln in b.splitlines() if ln.strip()]
        cur: List[str] = []
        for ln in lines:
            cur.append(ln)
            if ARROW_Q.search(ln):
                cand = "\n".join(cur).strip()
                if cand:
                    out.append(cand)
                cur = []
        if cur:
            tail = "\n".join(cur).strip()
            if tail:
                out.append(tail)

    # If split produced nothing useful, keep original
    return out or [s]


def unpack_riddles(riddles: Any, expected: int) -> Tuple[List[str], bool]:
    """
    Returns (new_riddles, changed).
    Only modifies when it can split a packed item into >=2 options.
    """
    expected = max(1, int(expected))
    if not isinstance(riddles, list):
        return [], False
    cleaned = [r for r in riddles if isinstance(r, str) and r.strip()]
    if not cleaned:
        return [], False

    # Case A: single packed string
    if len(cleaned) == 1 and len(ARROW_Q.findall(cleaned[0])) > 1:
        parts = _split_packed_string(cleaned[0])
        parts = [p for p in parts if p.strip()]
        if len(parts) >= 2:
            return parts[:expected], True
        return cleaned, False

    # Case B: list elements that themselves contain multiple riddles (rare)
    changed = False
    out: List[str] = []
    for r in cleaned:
        if len(ARROW_Q.findall(r)) > 1:
            parts = _split_packed_string(r)
            if len(parts) >= 2:
                out.extend(parts)
                changed = True
            else:
                out.append(r)
        else:
            out.append(r)
    if changed:
        return out[:expected], True

    return cleaned, False


def main() -> None:
    p = argparse.ArgumentParser(description="Unpack concatenated text riddle options (offline).")
    p.add_argument("--results-root", type=str, required=True)
    p.add_argument("--expected", type=int, default=3, help="Expected options to keep after unpacking.")
    p.add_argument("--backup", action="store_true", default=True, help="Write .bak before overwriting JSON.")
    p.add_argument("--dry-run", action="store_true", help="Do not write anything; only report.")
    args = p.parse_args()

    root = Path(args.results_root).resolve()
    if not root.exists():
        raise SystemExit(f"not found: {root}")

    paths = sorted(root.rglob("text_riddle_options.json"))
    changed = 0
    skipped = 0
    errors = 0

    for path in paths:
        try:
            data = _load_json(path)
            riddles = data.get("riddles")
            new_riddles, did_change = unpack_riddles(riddles, args.expected)
            if not did_change:
                skipped += 1
                continue

            if args.dry_run:
                changed += 1
                continue

            if args.backup:
                bak = path.with_suffix(path.suffix + ".bak")
                if not bak.exists():
                    bak.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")

            data["riddles"] = new_riddles
            data["n_options"] = len(new_riddles)
            data["unpacked"] = True
            data["unpacked_expected"] = int(args.expected)
            _write_json(path, data)
            changed += 1
        except Exception:
            errors += 1

    print(f"[unpack] results_root={root}")
    print(f"[unpack] files={len(paths)} changed={changed} skipped={skipped} errors={errors}")
    if args.dry_run:
        print("[unpack] dry-run: no files written")


if __name__ == "__main__":
    main()

