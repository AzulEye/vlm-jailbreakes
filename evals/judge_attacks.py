#!/usr/bin/env python3
"""Run LLM judge over attack outputs (separate from attack execution)."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evals.safety_judge import judge_responses


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


def _infer_mode_tag(data: Dict[str, object], override: str) -> str:
    if override:
        return override
    glyph_mode = data.get("glyph_mode")
    if isinstance(glyph_mode, str) and glyph_mode.strip().lower() == "images":
        return "img"
    return ""


def _vlm_suffix(vlm_model: str) -> str:
    if "/" in vlm_model:
        return vlm_model.split("/", 1)[1]
    return vlm_model


def collect_tasks(results_root: Path) -> List[Path]:
    return list(results_root.rglob("vlm_reply*.json"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Judge attack outputs using the LLM safety judge.")
    parser.add_argument("--results-root", type=Path, default=Path("results/attacks"))
    parser.add_argument(
        "--output-file",
        type=str,
        default="",
        help="Filename to write under each attack result directory. If empty, defaults to judge_results_{vlmModel}.json.",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="x-ai/grok-4.1-fast,google/gemini-3-flash-preview,anthropic/claude-haiku-4.5",
        help="Comma-separated list of judge model IDs.",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--thinking", action="store_true", default=False, help="Enable model reasoning/thinking if supported.")
    parser.add_argument("--concurrency", type=int, default=10, help="How many files to process in parallel.")
    parser.add_argument("--mode-tag", type=str, default="", help="Optional suffix appended to output filenames.")
    args = parser.parse_args()

    if not args.results_root.exists():
        raise SystemExit(f"Results root not found: {args.results_root}")

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    tasks = collect_tasks(args.results_root)
    if not tasks:
        print("No attack result JSON files found.")
        return

    def worker(path: Path) -> Optional[str]:
        try:
            data = json.loads(path.read_text())
            replies = _collect_reply_texts(data)
            if not replies:
                return f"Skipping {path}: no replies found"
            meta_path = path.parent / "metadata.json"
            if not meta_path.exists():
                return f"Skipping {path}: metadata.json not found"
            meta = json.loads(meta_path.read_text())
            behavior = (meta.get("behavior") or "").strip()
            behavior_id = (meta.get("behavior_id") or "").strip()
            if not behavior:
                return f"Skipping {path}: missing behavior in metadata.json"
            vlm_model = _infer_vlm_model(data)
            slug = slugify(behavior_id or behavior, fallback="item")
            mode_tag = _infer_mode_tag(data, args.mode_tag)

            judge_results = judge_responses(
                behavior=behavior,
                replies=replies,
                judge_models=models,
                temperature=args.temperature,
                thinking=args.thinking,
            )

            if args.output_file:
                out_name = args.output_file
            else:
                out_name = f"judge_results_{_vlm_suffix(vlm_model)}.json"
            out_name = append_tag_to_filename(out_name, mode_tag)
            out_path = path.parent / out_name
            payload = {
                "behavior": behavior,
                "behavior_id": behavior_id,
                "slug": slug,
                "vlm_model": vlm_model,
                "glyph_mode": data.get("glyph_mode") or "abstract",
                "vlm_reply_file": str(path),
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