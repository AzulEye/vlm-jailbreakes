#!/usr/bin/env python3
"""
Run neutralized HarmBench visual object replacement attacks.

Mirrors attacks/visual_text_replacement/run_neutralized.py in CLI, config keys,
and output contract so that evals/judge_attacks.py and analysis/run_results_summary.py
consume both attack types identically.

Usage:
    python -m attacks.visual_object_replacement.run_neutralized --config attacks/visual_object_replacement/attack_config_neutralized.json
    python -m attacks.visual_object_replacement.run_neutralized --config attacks/visual_object_replacement/attack_config_neutralized.json --quiet
"""

import argparse
import csv
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from attacks.visual_object_replacement.execute_attack import collect_slot_images, run_attack_for_images
from attacks.visual_object_replacement.attack_prompts import (
    SLOT_ORDER,
    build_neutralized_prompt,
    render_neutralized_prompt,
)

MAX_PARALLEL_DEFAULT = 4
MAX_PARALLEL_CAP = 8
ATTACK_NAME = "visual_object_replacement_neutralized"


def load_config(config_path: Path) -> Dict[str, Any]:
    with config_path.open("r") as f:
        return json.load(f)


def _resolve_csv_path(config_path: Path, csv_value: str) -> Path:
    csv_path = Path(csv_value)
    if not csv_path.is_absolute():
        csv_path = (config_path.parent / csv_path).resolve()
    else:
        csv_path = csv_path.resolve()
    return csv_path


def load_harmbench_csv(csv_path: Path) -> List[Dict[str, Any]]:
    """Load harmbench_neutralized CSV; return rows with slot_values parsed."""
    if not csv_path.exists():
        raise ValueError(f"Neutralized CSV not found: {csv_path}")

    rows: List[Dict[str, Any]] = []
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        required = {"Neutralized", "X1", "X2", "X3", "X4", "BehaviorID"}
        missing = sorted(required - set(fieldnames))
        if missing:
            raise ValueError(f"Neutralized CSV missing columns: {', '.join(missing)}")

        for idx, row in enumerate(reader, start=1):
            neutralized = (row.get("Neutralized") or "").strip()
            if not neutralized:
                continue
            slot_values: Dict[str, str] = {}
            for slot in SLOT_ORDER:
                value = (row.get(slot) or "").strip()
                if value:
                    slot_values[slot] = value
            if not slot_values:
                continue
            behavior_id = (row.get("BehaviorID") or "").strip()
            rows.append({
                "row_index": idx,
                "behavior_id": behavior_id,
                "neutralized": neutralized,
                "slot_values": slot_values,
                "original": (row.get("Original") or "").strip(),
                "semantic_category": (row.get("SemanticCategory") or "").strip(),
            })
    return rows


def _sanitize_name(value: str) -> str:
    value = re.sub(r'[^\w\s-]', '', value)
    value = re.sub(r'[-\s]+', '_', value)
    return value[:100].strip('_')


def _safe_dir_name(value: str, fallback: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9._-]+", "_", value.strip())
    return safe or fallback


def _attack_dir(results_root: Path, behavior_id: str) -> Path:
    return results_root / _sanitize_name(behavior_id)


def _vlm_suffix(model_name: Optional[str]) -> str:
    if not model_name:
        return "vlm"
    if "/" in model_name:
        return model_name.split("/", 1)[1]
    return model_name


def result_file_path(
    results_root: Path, behavior_id: str, model_name: Optional[str],
) -> Path:
    return _attack_dir(results_root, behavior_id) / f"vlm_reply_{_vlm_suffix(model_name)}.json"


def _metadata_path(results_root: Path, behavior_id: str) -> Path:
    return _attack_dir(results_root, behavior_id) / "metadata.json"


def _build_run_id(provider: str, model_name: Optional[str]) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    model_tag = _sanitize_name(model_name or "vlm")
    return f"{timestamp}__{provider}__{model_tag}"


def _collect_runs(attempts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    runs: List[Dict[str, Any]] = []
    run_idx = 1
    for attempt in attempts:
        responses = attempt.get("responses") or []
        for resp in responses:
            if not isinstance(resp, dict):
                continue
            reply = resp.get("response") or resp.get("raw_response") or ""
            runs.append({"run": run_idx, "reply": reply, "parsed_tokens": None})
            run_idx += 1
    return runs


def save_attack_outputs(
    results_root: Path,
    behavior_id: str,
    provider: str,
    model_name: Optional[str],
    prompt: str,
    attempts: List[Dict[str, Any]],
    image_paths: List[str],
    *,
    behavior: str = "",
    neutralized: str = "",
    slot_values: Optional[Dict[str, str]] = None,
    slot_replacements: Optional[Dict[str, str]] = None,
    slot_image_counts: Optional[Dict[str, int]] = None,
    semantic_category: str = "",
    csv_path: str = "",
    error: Optional[str] = None,
    images_by_slot: Optional[Dict[str, List[str]]] = None,
) -> None:
    attack_dir = _attack_dir(results_root, behavior_id)
    attack_dir.mkdir(parents=True, exist_ok=True)

    runs = _collect_runs(attempts)
    vlm_path = result_file_path(results_root, behavior_id, model_name)
    metadata_path = _metadata_path(results_root, behavior_id)

    vlm_payload: Dict[str, Any] = {
        "model": model_name or "vlm",
        "temperature": 0.0,
        "glyph_mode": "abstract",
        "legend": None,
        "sentence": None,
        "decode_prompt": "",
        "instruct_prompt": prompt,
        "decoding": {"reply": None, "parsed_tokens": None, "expected_tokens": None},
        "runs": runs,
    }
    if error:
        vlm_payload["error"] = error

    metadata_payload: Dict[str, Any] = {
        "attack_name": ATTACK_NAME,
        "behavior_id": behavior_id,
        "behavior": behavior,
        "neutralized_question": neutralized,
        "provider": provider,
        "model": model_name,
        "run_id": _build_run_id(provider, model_name),
        "images": image_paths,
        "attempts": len(attempts),
    }
    if slot_values:
        metadata_payload["slot_values"] = slot_values
    if slot_replacements:
        metadata_payload["slot_replacements"] = slot_replacements
    if slot_image_counts:
        metadata_payload["slot_image_counts"] = slot_image_counts
    if images_by_slot:
        metadata_payload["images_by_slot"] = images_by_slot
    if semantic_category:
        metadata_payload["semantic_category"] = semantic_category
    if csv_path:
        metadata_payload["csv_path"] = csv_path
    if error:
        metadata_payload["error"] = error

    vlm_path.write_text(json.dumps(vlm_payload, indent=2), encoding="utf-8")
    metadata_path.write_text(json.dumps(metadata_payload, indent=2), encoding="utf-8")


def run_from_config(
    config: Dict[str, Any],
    config_path: Path,
    quiet: bool,
    redo_existing: bool,
    max_parallel: int,
) -> int:
    csv_value = config.get("harmbench_csv")
    if not csv_value:
        raise ValueError("Config must include 'harmbench_csv'.")
    csv_path = _resolve_csv_path(config_path, str(csv_value))
    rows = load_harmbench_csv(csv_path)
    if not rows:
        raise ValueError(f"No usable rows found in neutralized CSV: {csv_path}")

    models: List[Dict[str, Any]] = config.get("models", [])
    if not models:
        raise ValueError("Config must include a non-empty 'models' list")

    data_root = Path(config.get("data_root", REPO_ROOT / "data" / "visual_replacement")).resolve()
    results_root = Path(
        config.get("results_root", REPO_ROOT / "results" / "attacks" / "visual_object_replacement")
    ).resolve()
    max_attempts = config.get("max_attempts", 5)

    slot_replacements = config.get("slot_replacements") or {
        "X1": "banana",
        "X2": "carrot",
        "X3": "water bottle",
        "X4": "broccoli",
    }
    max_images_per_slot = int(config.get("max_images_per_slot", 0) or 0)
    start_image_index = int(config.get("start_image_index", 0) or 0)

    if not quiet:
        print(f"Loaded {len(rows)} behaviors from {csv_path}")
        print(f"Data root: {data_root}")
        print(f"Results root: {results_root}")

    errors = []

    def run_single(task: Dict[str, Any]):
        row = task["row"]
        provider = task["provider"]
        model_name = task["model_name"]
        api_key = task["api_key"]
        quiet_local = task["quiet"]

        behavior_id = row["behavior_id"]

        result_path = result_file_path(results_root, behavior_id, model_name)
        if result_path.exists() and not redo_existing:
            if not quiet_local:
                print(f"[SKIP] {behavior_id} already exists for {model_name}")
            return None

        slot_values = row["slot_values"]
        neutralized = row["neutralized"]

        slot_images = collect_slot_images(
            data_root=data_root,
            slot_values=slot_values,
            slot_replacements=slot_replacements,
            max_images_per_slot=max_images_per_slot,
            start_image_index=start_image_index,
        )
        image_paths: List[str] = []
        slot_images_serialized: Dict[str, List[str]] = {}
        slot_image_counts: Dict[str, int] = {}
        for slot in SLOT_ORDER:
            images = slot_images.get(slot)
            if not images:
                continue
            serialized = [str(p) for p in images]
            slot_images_serialized[slot] = serialized
            slot_image_counts[slot] = len(serialized)
            image_paths.extend(serialized)

        prompt = build_neutralized_prompt(
            neutralized_prompt=neutralized,
            slot_values=slot_values,
            slot_replacements=slot_replacements,
            slot_image_counts=slot_image_counts,
        )
        behavior_text = render_neutralized_prompt(neutralized, slot_values)

        attempts: List[Dict[str, Any]] = []
        for attempt_num in range(1, max_attempts + 1):
            if not quiet_local:
                print(f"\n[ATTEMPT {attempt_num}/{max_attempts}] {behavior_id} {provider}/{model_name}")
            try:
                response = run_attack_for_images(
                    image_paths=image_paths,
                    prompt=prompt,
                    provider=provider,
                    model=model_name,
                    api_key=api_key,
                    quiet=quiet_local,
                )
                attempts.append({
                    "attempt": attempt_num,
                    "responses": [{"images": image_paths, "slot_images": slot_images_serialized, "response": response}],
                })
            except Exception as e:
                if not quiet_local:
                    print(f"  Error: {e}")
                attempts.append({"attempt": attempt_num, "error": str(e), "responses": []})

        save_attack_outputs(
            results_root=results_root,
            behavior_id=behavior_id,
            provider=provider,
            model_name=model_name,
            prompt=prompt,
            attempts=attempts,
            image_paths=image_paths,
            behavior=behavior_text,
            neutralized=neutralized,
            slot_values=slot_values,
            slot_replacements=slot_replacements,
            slot_image_counts=slot_image_counts,
            semantic_category=row.get("semantic_category", ""),
            csv_path=str(csv_path),
            images_by_slot=slot_images_serialized,
        )
        return None

    tasks: List[Dict[str, Any]] = []
    for row_idx, row in enumerate(rows, start=1):
        behavior_id = row["behavior_id"] or f"row_{row['row_index']}"
        if not quiet:
            print(f"\n=== [{row_idx}/{len(rows)}] {behavior_id} ===")

        for model_cfg in models:
            provider = model_cfg.get("provider", "ollama")
            model_name = model_cfg.get("model")
            api_key = model_cfg.get("api_key")

            tasks.append({
                "row": row,
                "provider": provider,
                "model_name": model_name,
                "api_key": api_key,
                "quiet": quiet,
            })

    if not quiet:
        print(f"\n[EXECUTOR] {len(tasks)} total tasks, up to {max_parallel} parallel threads.")

    with ThreadPoolExecutor(max_workers=max_parallel, thread_name_prefix="AttackWorker") as executor:
        futures = []
        for task in tasks:
            futures.append((task, executor.submit(run_single, task)))

        for task, future in futures:
            try:
                future.result()
            except Exception as exc:
                bid = task["row"].get("behavior_id", "unknown")
                errors.append(f"{bid}:{task['provider']}:{task['model_name']} -> {exc}")

    if errors:
        print("\nCompleted with errors:")
        for err in errors:
            print(f"- {err}")
        return 1

    if not quiet:
        print("\nAll runs completed successfully.")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run visual object replacement attacks using neutralized harmbench prompts."
    )
    parser.add_argument(
        "--config",
        default=str(Path(__file__).parent / "attack_config_neutralized.json"),
        help="Path to JSON config.",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output.")
    parser.add_argument(
        "--redo-existing",
        action="store_true",
        help="Rerun and overwrite existing results.",
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=MAX_PARALLEL_DEFAULT,
        help="Maximum parallel worker threads (capped at 8).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    max_parallel = max(1, min(args.max_parallel, MAX_PARALLEL_CAP))

    config_path = Path(args.config).resolve()
    if not config_path.exists():
        print(f"Error: config not found at {config_path}")
        sys.exit(1)

    try:
        cfg = load_config(config_path)
        exit_code = run_from_config(
            cfg,
            config_path=config_path,
            quiet=args.quiet,
            redo_existing=args.redo_existing,
            max_parallel=max_parallel,
        )
    except Exception as exc:
        import traceback
        print(f"Error: {exc}")
        traceback.print_exc()
        sys.exit(1)

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
