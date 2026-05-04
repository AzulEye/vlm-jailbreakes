#!/usr/bin/env python3
"""Run LLM judge over v4 attack outputs (text-only + image-only).

Example:
python MARS4-Gandelsman/evals/judge_attacks_v4.py \
  --results-root MARS4-Gandelsman/results_imageriddle_v4/vlm_text_vs_visual_solvingfirst_v5 \
  --models "x-ai/grok-4.1-fast,google/gemini-3-flash-preview,anthropic/claude-haiku-4.5" \
  --concurrency 10 \
  --skip-existing \
  --allow-categories chemical_biological cybercrime_intrusion misinformation_disinformation harassment_bullying illegal harmful
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set

REPO_ROOT = Path(__file__).resolve().parents[3]
ANALOGY_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from safety_judge import judge_responses


def slugify(text: str, fallback: str) -> str:
    base = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    if not base:
        base = fallback or "item"
    return base


def append_tag_to_filename(filename: str, tag: str) -> str:
    if not tag:
        return filename
    p = Path(filename)
    if p.stem.endswith(f"_{tag}"):
        return filename
    return f"{p.stem}_{tag}{p.suffix}"


def _collect_reply_texts(data: Dict[str, object]) -> List[str]:
    replies: List[str] = []
    runs = data.get("runs") or []
    if isinstance(runs, list) and runs:
        for run in runs:
            if not isinstance(run, dict):
                continue
            text = run.get("reply") or ""
            if isinstance(text, str):
                replies.append(text)
    return replies


def _infer_vlm_model(data: Dict[str, object]) -> str:
    model = data.get("model")
    if isinstance(model, str) and model.strip():
        return model
    if model is not None:
        return str(model)
    provider = data.get("provider")
    if isinstance(provider, str) and provider.strip():
        return provider
    return "vlm"


def _infer_mode_tag(data: Dict[str, object], override: str, path: Optional[Path] = None) -> str:
    if override:
        return override
    if path is not None:
        name = path.name.lower()
        if name.startswith("vlm_text"):
            return "txt"
        if name.startswith("vlm_visual"):
            return "img"
    mode = data.get("mode")
    if isinstance(mode, str) and mode.strip().lower() in {"text", "txt"}:
        return "txt"
    if isinstance(mode, str) and mode.strip().lower() in {"visual", "image", "img", "images"}:
        return "img"
    return ""


def _vlm_suffix(vlm_model: str) -> str:
    if "/" in vlm_model:
        return vlm_model.split("/", 1)[1]
    return vlm_model


def collect_tasks(results_root: Path, patterns: Sequence[str]) -> List[Path]:
    seen: Set[Path] = set()
    for pat in patterns:
        pat = (pat or "").strip()
        if not pat:
            continue
        for p in results_root.rglob(pat):
            if p.is_file():
                seen.add(p)
    return sorted(seen)


def _category_allowed(category: str, allowlist: Sequence[str]) -> bool:
    if not allowlist:
        return True
    c = (category or "").strip().lower()
    return any(c == (a or "").strip().lower() for a in allowlist)


def _load_original_behaviors(path: Path) -> List[str]:
    import csv

    behaviors = []
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            behaviors.append((row.get("Original") or "").strip())
    return behaviors


def _stable_hash_rank(seed: int, value: str) -> int:
    h = hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()
    return int(h[:16], 16)


def _sample_combo_dirs_per_row(
    *,
    tasks: List[Path],
    max_combos_per_row: int,
    seed: int,
) -> List[Path]:
    """Limit judging to at most N combo directories per (category,row)."""
    if max_combos_per_row <= 0:
        return tasks

    # Group by (category,row) and then by combo directory (parent of vlm_*.json).
    groups: Dict[tuple[str, str], Dict[Path, List[Path]]] = {}
    for p in tasks:
        combo_dir = p.parent
        inputs_path = combo_dir / "inputs.json"
        if not inputs_path.exists():
            # If we can't resolve row/category, keep it (conservative).
            groups.setdefault(("", ""), {}).setdefault(combo_dir, []).append(p)
            continue
        try:
            inputs = json.loads(inputs_path.read_text(encoding="utf-8"))
        except Exception:
            groups.setdefault(("", ""), {}).setdefault(combo_dir, []).append(p)
            continue
        category = str(inputs.get("category") or "").strip()
        row = str(inputs.get("row") or "").strip()
        groups.setdefault((category, row), {}).setdefault(combo_dir, []).append(p)

    kept: List[Path] = []
    for (category, row), combo_map in groups.items():
        combo_dirs = sorted(combo_map.keys())
        if len(combo_dirs) <= max_combos_per_row:
            chosen = combo_dirs
        else:
            ranked = sorted(
                combo_dirs,
                key=lambda d: _stable_hash_rank(seed, f"{category}|{row}|{d.name}"),
            )
            chosen = ranked[:max_combos_per_row]
        for d in chosen:
            kept.extend(combo_map[d])
    # Preserve stable order for nicer logs
    return sorted(set(kept))


def _select_behavior(inputs: Dict[str, object], mode_tag: str, original_behaviors: List[str]) -> str:
    # 1. Try to get behavior from row index if original_behaviors is provided
    row_slug = str(inputs.get("row") or "")
    if row_slug.startswith("row_") and original_behaviors:
        try:
            idx = int(row_slug.replace("row_", "")) - 1
            if 0 <= idx < len(original_behaviors):
                b = original_behaviors[idx]
                if b:
                    return b
        except ValueError:
            pass

    # 2. Fallback to prompts in inputs.json (original logic)
    if mode_tag == "img":
        visual = inputs.get("visual_prompt")
        if isinstance(visual, str) and visual.strip():
            return visual
    text = inputs.get("text_prompt")
    if isinstance(text, str) and text.strip():
        return text
    visual = inputs.get("visual_prompt")
    if isinstance(visual, str) and visual.strip():
        return visual
    safe = inputs.get("safe_prompt")
    if isinstance(safe, str) and safe.strip():
        return safe
    return ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Judge v4 attack outputs using the LLM safety judge.")
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results_imageriddle_v4/vlm_text_vs_visual_solvingfirst_v4"),
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="",
        help="Filename to write under each result directory. If empty, defaults to judge_results_{vlmModel}.json.",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="x-ai/grok-4.1-fast,google/gemini-3-flash-preview,anthropic/claude-haiku-4.5",
        help="Comma-separated list of judge model IDs.",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--thinking", action="store_true", default=False, help="Enable model reasoning if supported.")
    parser.add_argument("--concurrency", type=int, default=10, help="How many files to process in parallel.")
    parser.add_argument("--mode-tag", type=str, default="", help="Optional suffix appended to output filenames.")
    parser.add_argument(
        "--input-patterns",
        type=str,
        default="vlm_text*.json,vlm_visual*.json",
        help="Comma-separated glob patterns under results-root to judge (recursive).",
    )
    parser.add_argument(
        "--max-combos-per-row",
        type=int,
        default=0,
        help="If set (>0), judge at most N combo directories per (category,row).",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=0,
        help="Seed for deterministic sampling when --max-combos-per-row is set.",
    )
    parser.add_argument(
        "--allow-categories",
        nargs="*",
        default=[],
        help="Limit judging to specific categories (folder names under results-root).",
    )
    parser.add_argument(
        "--original-csv",
        type=Path,
        default=ANALOGY_DIR / "hb_orignal.csv",
        help="CSV containing original harmful intentions (column 'Original').",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=False,
        help="If set, skip inputs whose judge_results output file already exists.",
    )
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--api-base", type=str, default="")
    args = parser.parse_args()

    if not args.results_root.exists():
        raise SystemExit(f"Results root not found: {args.results_root}")

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    patterns = [p.strip() for p in (args.input_patterns or "").split(",") if p.strip()]
    tasks = collect_tasks(args.results_root, patterns)
    tasks = _sample_combo_dirs_per_row(
        tasks=tasks,
        max_combos_per_row=int(args.max_combos_per_row),
        seed=int(args.sample_seed),
    )
    if not tasks:
        print("No attack result JSON files found.")
        return

    allow = [c for c in (args.allow_categories or []) if str(c).strip()]
    original_behaviors = _load_original_behaviors(args.original_csv)
    if not original_behaviors and args.original_csv.exists():
        print(f"Warning: No behaviors loaded from {args.original_csv}")

    def worker(path: Path) -> Optional[str]:
        try:
            data = json.loads(path.read_text())
            replies = _collect_reply_texts(data)
            if not replies:
                return f"Skipping {path}: no replies found"
            inputs_path = path.parent / "inputs.json"
            if not inputs_path.exists():
                return f"Skipping {path}: inputs.json not found"
            inputs = json.loads(inputs_path.read_text())
            mode_tag = _infer_mode_tag(data, args.mode_tag, path)
            behavior = _select_behavior(inputs, mode_tag, original_behaviors)
            if not behavior:
                return f"Skipping {path}: missing behavior/prompts"
            vlm_model = _infer_vlm_model(data)
            category = (inputs.get("category") or "").strip() if isinstance(inputs.get("category"), str) else ""
            if not _category_allowed(category, allow):
                return None
            row = (inputs.get("row") or "").strip() if isinstance(inputs.get("row"), str) else ""
            combo = (inputs.get("combo") or "").strip() if isinstance(inputs.get("combo"), str) else ""
            slug = slugify(f"{category}-{row}-{combo}", fallback="item")

            judge_results = judge_responses(
                behavior=behavior,
                replies=replies,
                judge_models=models,
                api_key=args.api_key,
                api_base=args.api_base or None,
                temperature=args.temperature,
                thinking=args.thinking,
            )

            if args.output_file:
                out_name = args.output_file
            else:
                out_name = f"judge_results_{_vlm_suffix(vlm_model)}.json"
            out_name = append_tag_to_filename(out_name, mode_tag)
            out_path = path.parent / out_name
            if args.skip_existing and out_path.exists():
                return f"Skipping {path}: found existing {out_path}"
            payload = {
                "category": category or "unknown",
                "row": row,
                "combo": combo,
                "slug": slug,
                "vlm_model": vlm_model,
                "mode": mode_tag,
                "vlm_reply_file": str(path),
                "inputs_file": str(inputs_path),
                "prompt_used": behavior,
                "models": models,
                "temperature": args.temperature,
                "thinking": args.thinking,
                "judge_results": judge_results,
            }
            out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            return f"Wrote judge scores to {out_path}"
        except Exception as exc:  # noqa: BLE001
            return f"Error processing {path}: {exc}"

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as executor:
        futures = [executor.submit(worker, p) for p in tasks]
        for fut in concurrent.futures.as_completed(futures):
            msg = fut.result()
            if msg:
                print(msg)


if __name__ == "__main__":
    main()
