#!/usr/bin/env python3
"""Use an LLM judge to score LLM replies for safety and compliance per batch directory.

Adapted from visual_cipher/judge_responses.py for textual cipher attack.
"""

from __future__ import annotations

import argparse
import csv
import concurrent.futures
import json
import os
import re
from pathlib import Path
from typing import Dict, List

from openai import OpenAI


def slugify(text: str, fallback: str, index: int, used: set[str]) -> str:
    base = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    if not base:
        base = fallback or f"item-{index}"
    slug = base
    counter = 1
    while slug in used:
        slug = f"{base}-{counter}"
        counter += 1
    used.add(slug)
    return slug


def append_tag_to_filename(filename: str, tag: str) -> str:
    if not tag:
        return filename
    p = Path(filename)
    if p.stem.endswith(f"_{tag}"):
        return filename
    return f"{p.stem}_{tag}{p.suffix}"


import sys
from pathlib import Path

# Add parent directory to path to enable imports from evals
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from evals.safety_judge import build_judge_prompt


def judge_responses(client: OpenAI, model: str, behavior: str, replies: List[str], temperature: float, thinking: bool) -> Dict:
    prompt = build_judge_prompt(behavior, replies)
    extra = {"include_reasoning": True} if thinking else {}
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=2000,
        temperature=temperature,
        extra_body=extra if extra else None,
    )
    content = resp.choices[0].message.content
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        cleaned = content.strip()
        if cleaned.startswith("```"):
            if cleaned.startswith("```json"):
                cleaned = cleaned[len("```json") :].strip()
            elif cleaned.startswith("```"):
                cleaned = cleaned[3:].strip()
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3].strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            last = cleaned.rfind("}")
            if last != -1:
                candidate = cleaned[: last + 1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    pass
            return {"error": "failed to parse judge response", "raw": content}


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM judge to score textual cipher replies for safety and compliance.")
    parser.add_argument("--csv", type=Path, default=Path("harmbench_behaviors_text_test.csv"))
    parser.add_argument("--batch-root", type=Path, default=Path("results/attacks/textual_cipher"))
    parser.add_argument(
        "--output-file",
        type=str,
        default="",
        help="Filename to write under each slug directory. If empty, defaults to judge_results_{llmModel}.json.",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="x-ai/grok-4.1-fast,google/gemini-3-flash-preview,anthropic/claude-haiku-4.5",
        help="Comma-separated list of judge model IDs.",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--thinking", action="store_true", default=False)
    parser.add_argument("--concurrency", type=int, default=10)
    args = parser.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("Please set OPENROUTER_API_KEY before running.")
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    if not args.csv.exists():
        raise SystemExit(f"CSV not found: {args.csv}")
    if not args.batch_root.exists():
        raise SystemExit(f"Batch root not found: {args.batch_root}")

    used_slugs: set[str] = set()
    tasks = []
    with args.csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            if (row.get("FunctionalCategory") or "").strip().lower() != "standard":
                continue
            behavior = (row.get("Behavior") or "").strip()
            if not behavior:
                continue
            behavior_id = (row.get("BehaviorID") or "").strip()
            slug = slugify(behavior_id or behavior, fallback="item", index=idx, used=used_slugs)
            base_dir = args.batch_root / slug
            llm_paths = sorted(base_dir.glob("llm_reply*.json"))
            if not llm_paths:
                print(f"Skipping {slug}: no llm_reply*.json found in {base_dir}")
                continue
            for llm_path in llm_paths:
                tasks.append((slug, behavior, behavior_id, base_dir, llm_path))

    def worker(task):
        slug, behavior, behavior_id, base_dir, llm_path = task
        try:
            data = json.loads(llm_path.read_text())
            runs = data.get("runs", [])
            replies = [r.get("reply", "") for r in runs if "reply" in r]
            if not replies:
                return f"Skipping {slug}: no replies found in {llm_path}"

            llm_model = data.get("model", "llm")
            llm_suffix = llm_model.split("/", 1)[1] if "/" in llm_model else llm_model
            cipher_type = data.get("cipher_type", "textual")

            if args.output_file:
                out_name = args.output_file
            else:
                out_name = f"judge_results_{llm_suffix}.json"
            out_path = base_dir / out_name
            if out_path.exists():
                return None

            model_results = []
            for model in models:
                try:
                    result = judge_responses(client, model, behavior, replies, args.temperature, args.thinking)
                    model_results.append({"model": model, "result": result})
                except Exception as e:  # noqa: BLE001
                    model_results.append({"model": model, "error": f"{type(e).__name__}: {e}"})
            payload = {
                "behavior": behavior,
                "behavior_id": behavior_id,
                "slug": slug,
                "llm_model": llm_model,
                "cipher_type": cipher_type,
                "llm_reply_file": str(llm_path),
                "models": models,
                "temperature": args.temperature,
                "thinking": args.thinking,
                "judge_results": model_results,
            }
            out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            return f"Wrote judge scores to {out_path}"
        except Exception as e:
            return f"Error processing {slug}: {e}"

    if not tasks:
        print("No tasks to process.")
        return

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as executor:
        futures = [executor.submit(worker, t) for t in tasks]
        for fut in concurrent.futures.as_completed(futures):
            msg = fut.result()
            if msg:
                print(msg)


if __name__ == "__main__":
    main()
