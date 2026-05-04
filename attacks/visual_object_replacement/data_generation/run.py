#!/usr/bin/env python3
"""
Pipeline to generate base images and run configured attacks using REVE.

Usage:
  python data_generation/visual_replacement/run.py --config data_generation/visual_replacement/config.json

Reads a JSON config (see data_generation/visual_replacement/config.json) with:
  - output_root
  - base_generation settings (count_per_object, model_version)
  - objects: list of target object strings
  - attacks: list of attack configs:
        {
          "type": "replace_with_object" | "replace_with_BBBLORK" | "replace_with_nothing",
          "replacements": ["banana", ...]   # only for replace_with_object
          "model_version": "reve-edit-fast@20251030"
        }
For each object, generates N base images (via generate_base_image.py) then applies each attack
on each generated image (via generate_attack_image.py).
"""

import argparse
import json
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MAX_PARALLEL_DEFAULT = 4
MAX_TASK_ATTEMPTS = 3


class FailureTracker:
    def __init__(self):
        self._lock = threading.Lock()
        self._failures: List[str] = []

    def record_failure(self, description: str):
        with self._lock:
            self._failures.append(description)

    def get_failures(self) -> List[str]:
        with self._lock:
            return list(self._failures)


def run_generation_task(
    description: str,
    cmd: List[str],
    env: Optional[dict],
    failure_tracker: FailureTracker,
    max_attempts: int,
):
    for attempt in range(1, max_attempts + 1):
        thread_name = threading.current_thread().name
        print(f"[{thread_name}] Launching {description} (attempt {attempt}/{max_attempts})")
        result = None
        try:
            result = subprocess.run(cmd, env=env)
        except Exception as exc:
            print(f"[{thread_name}] {description} raised {exc!r}")
        else:
            if result.returncode == 0:
                print(f"[{thread_name}] Completed {description}")
                return True
            print(
                f"[{thread_name}] {description} failed with exit {result.returncode}"
            )
    failure_tracker.record_failure(description)
    print(f"[{threading.current_thread().name}] Giving up on {description}")
    return False


def generate_base_images(
    obj: str,
    count: int,
    aspect_ratio: str,
    version: str,
    extra_prompt: Optional[str],
    output_dir: Path,
    api_key: Optional[str],
    timeout: int,
    max_retries: int,
    retry_backoff: int,
    redo_existing: bool,
    executor: ThreadPoolExecutor,
    failure_tracker: FailureTracker,
    task_attempts: int,
    attack_type: Optional[str] = None,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    futures = []
    for i in range(1, count + 1):
        output_path = output_dir / f"{obj}_base_{i}.png"
        if output_path.exists() and not redo_existing:
            print(f"[BASE] Skipping existing {output_path}")
            continue
        cmd = [
            sys.executable,
            str(REPO_ROOT / "data_generation/visual_replacement/generate_base_image.py"),
            obj,
        ]
        if extra_prompt:
            cmd.append(extra_prompt)
        cmd += [
            "--output",
            str(output_path),
            "--aspect-ratio",
            aspect_ratio,
            "--version",
            version,
            "--timeout",
            str(timeout),
            "--max-retries",
            str(max_retries),
            "--retry-backoff",
            str(retry_backoff),
        ]
        if attack_type:
            cmd += ["--attack-type", attack_type]
        env = os.environ.copy()
        if api_key:
            env["REVE_API_KEY"] = api_key
        description = f"BASE {obj} #{i} ({output_path.name})"
        futures.append(
            executor.submit(
                run_generation_task,
                description,
                cmd,
                env,
                failure_tracker,
                task_attempts,
            )
        )
    return futures


def run_attack(
    attack_type: str,
    original_object: str,
    source_image: Path,
    output_path: Path,
    aspect_ratio: str,
    version: str,
    api_key: Optional[str],
    replacement_object: Optional[str] = None,
    executor: ThreadPoolExecutor = None,
    failure_tracker: FailureTracker = None,
    task_attempts: int = MAX_TASK_ATTEMPTS,
):
    cmd = [
        sys.executable,
        str(REPO_ROOT / "data_generation/visual_replacement/generate_attack_image.py"),
        str(source_image),
        "--attack-type",
        attack_type,
        "--original-object",
        original_object,
        "--output",
        str(output_path),
        "--aspect-ratio",
        aspect_ratio,
        "--version",
        version,
    ]
    if replacement_object:
        cmd += ["--replacement-object", replacement_object]

    env = os.environ.copy()
    if api_key:
        env["REVE_API_KEY"] = api_key
    attack_label = f"{attack_type}:{replacement_object}" if replacement_object else attack_type
    description = f"ATTACK {attack_label} on {source_image.name}"
    return executor.submit(
        run_generation_task,
        description,
        cmd,
        env,
        failure_tracker,
        task_attempts,
    )


def run_attacks_for_object(
    obj: str,
    base_images_dir: Path,
    attacks_config: list,
    output_root: Path,
    api_key: Optional[str],
    redo_existing: bool,
    executor: ThreadPoolExecutor,
    failure_tracker: FailureTracker,
    task_attempts: int,
):
    futures = []
    for img_path in sorted(base_images_dir.glob("*.png")):
        for attack in attacks_config:
            attack_type = attack["type"]
            aspect_ratio = "16:9"
            version = attack.get("model_version", "reve-edit-fast@20251030")

            if attack_type == "replace_with_object":
                replacements = attack.get("replacements", [])
                if not replacements:
                    print(f"Skipping replace_with_object (no replacements) for {img_path}")
                    continue
                for repl in replacements:
                    attack_dir = output_root / obj / attack_type / repl.replace(" ", "_")
                    attack_dir.mkdir(parents=True, exist_ok=True)
                    output_path = attack_dir / img_path.name.replace("_base_", f"_attack_{attack_type}_{repl}_")
                    if output_path.exists() and not redo_existing:
                        print(f"[ATTACK:{attack_type}] Skipping existing {output_path}")
                        continue
                    futures.append(
                        run_attack(
                            attack_type=attack_type,
                            original_object=obj,
                            source_image=img_path,
                            output_path=output_path,
                            aspect_ratio=aspect_ratio,
                            version=version,
                            api_key=api_key,
                            replacement_object=repl,
                            executor=executor,
                            failure_tracker=failure_tracker,
                            task_attempts=task_attempts,
                        )
                    )
            elif attack_type == "text_replacement":
                replacements = attack.get("replacements", [])
                if not replacements:
                    print(f"Skipping text_replacement (no replacements) for {img_path}")
                    continue
                for repl in replacements:
                    attack_dir = output_root / obj / attack_type / repl.replace(" ", "_")
                    attack_dir.mkdir(parents=True, exist_ok=True)
                    output_path = attack_dir / img_path.name.replace("_base_", f"_attack_{attack_type}_{repl}_")
                    if output_path.exists() and not redo_existing:
                        print(f"[ATTACK:{attack_type}] Skipping existing {output_path}")
                        continue
                    futures.append(
                        run_attack(
                            attack_type=attack_type,
                            original_object=obj,
                            source_image=img_path,
                            output_path=output_path,
                            aspect_ratio=aspect_ratio,
                            version=version,
                            api_key=api_key,
                            replacement_object=repl,
                            executor=executor,
                            failure_tracker=failure_tracker,
                            task_attempts=task_attempts,
                        )
                    )
            else:
                attack_dir = output_root / obj / attack_type
                attack_dir.mkdir(parents=True, exist_ok=True)
                output_path = attack_dir / img_path.name.replace("_base_", f"_attack_{attack_type}_")
                if output_path.exists() and not redo_existing:
                    print(f"[ATTACK:{attack_type}] Skipping existing {output_path}")
                    continue
                futures.append(
                    run_attack(
                        attack_type=attack_type,
                        original_object=obj,
                        source_image=img_path,
                        output_path=output_path,
                        aspect_ratio=aspect_ratio,
                        version=version,
                        api_key=api_key,
                        executor=executor,
                        failure_tracker=failure_tracker,
                        task_attempts=task_attempts,
                    )
                )
    return futures


def main():
    parser = argparse.ArgumentParser(description="Run data generation pipeline (base + attacks).")
    parser.add_argument(
        "--config",
        type=str,
        default=str(REPO_ROOT / "data_generation/visual_replacement/config.json"),
        help="Path to pipeline config JSON.",
    )
    parser.add_argument("--api-key", type=str, default=None, help="REVE API key (overrides env).")
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Create request timeout (seconds).",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Max retries for create requests.",
    )
    parser.add_argument(
        "--retry-backoff",
        type=int,
        default=2,
        help="Exponential backoff base (seconds).",
    )
    parser.add_argument(
        "--redo-existing",
        action="store_true",
        help="If set, re-run generation even when output files already exist.",
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=MAX_PARALLEL_DEFAULT,
        help="Maximum parallel worker threads (hard capped at 8).",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = json.load(f)

    output_root = Path(cfg.get("output_root", "./data_generated")).resolve()
    base_cfg = cfg["base_generation"]
    objects = cfg["objects"]
    attacks = cfg["attacks"]

    base_count = base_cfg.get("count_per_object", 5)
    base_aspect = "16:9"
    base_version = base_cfg.get("model_version", "reve-create@20250915")
    extra_prompt = None

    api_key = args.api_key or os.environ.get("REVE_API_KEY")
    if not api_key:
        raise SystemExit("REVE_API_KEY not set. Provide --api-key or export REVE_API_KEY.")

    base_output_root = output_root / "base"
    attack_output_root = output_root / "attacks"
    max_parallel = max(1, min(args.max_parallel, MAX_PARALLEL_DEFAULT))
    print(f"[EXECUTOR] Using up to {max_parallel} parallel threads.")
    failure_tracker = FailureTracker()

    with ThreadPoolExecutor(max_workers=max_parallel, thread_name_prefix="GenWorker") as executor:
        base_futures = []
        for obj in objects:
            obj_base_dir = base_output_root / obj
            # Check if any attack is text_replacement to determine attack_type for base generation
            attack_type_for_base = None
            for attack in attacks:
                if attack["type"] == "text_replacement":
                    attack_type_for_base = "text_replacement"
                    break
            base_futures.extend(
                generate_base_images(
                    obj=obj,
                    count=base_count,
                    aspect_ratio=base_aspect,
                    version=base_version,
                    extra_prompt=extra_prompt,
                    output_dir=obj_base_dir,
                    api_key=api_key,
                    timeout=args.timeout,
                    max_retries=args.max_retries,
                    retry_backoff=args.retry_backoff,
                    redo_existing=args.redo_existing,
                    executor=executor,
                    failure_tracker=failure_tracker,
                    task_attempts=MAX_TASK_ATTEMPTS,
                    attack_type=attack_type_for_base,
                )
            )
        for future in base_futures:
            future.result()
        print("[STAGE] Base image generation completed.")

        attack_futures = []
        for obj in objects:
            obj_base_dir = base_output_root / obj
            attack_futures.extend(
                run_attacks_for_object(
                    obj=obj,
                    base_images_dir=obj_base_dir,
                    attacks_config=attacks,
                    output_root=attack_output_root,
                    api_key=api_key,
                    redo_existing=args.redo_existing,
                    executor=executor,
                    failure_tracker=failure_tracker,
                    task_attempts=MAX_TASK_ATTEMPTS,
                )
            )
        for future in attack_futures:
            future.result()
        print("[STAGE] Attack generation completed.")

    failures = failure_tracker.get_failures()
    if failures:
        print(f"[SUMMARY] {len(failures)} tasks failed after retries:")
        for failure in failures:
            print(f"  - {failure}")
    else:
        print("[SUMMARY] All tasks completed successfully after retries.")


if __name__ == "__main__":
    main()
