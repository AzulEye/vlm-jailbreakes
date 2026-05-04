#!/usr/bin/env python3
"""Helpers for running visual object replacement attacks (neutralized pipeline)."""

import csv
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

from attacks.common.ollama_inference import analyze_multiple_images_ollama
from attacks.common.openrouter_inference import analyze_multiple_images_openrouter
from attacks.visual_object_replacement.attack_prompts import SLOT_ORDER

DEFAULT_MODELS = {
    "ollama": "qwen3-vl:235b-instruct",
    "openrouter": "qwen/qwen2.5-vl-32b-instruct",
}


def _collect_images(base_dir: Path) -> List[Path]:
    images: List[Path] = []
    for pattern in ("*.png", "*.jpg", "*.jpeg"):
        images.extend(base_dir.glob(pattern))
    return sorted(images)


def _normalize_object_name(object_name: str) -> str:
    return re.sub(r"\s+", " ", object_name.replace("/", " ")).strip()


def _normalize_concept_key(value: str) -> str:
    return _normalize_object_name(value).lower()


@lru_cache(maxsize=1)
def _load_concept_aliases() -> Dict[str, str]:
    for candidate in [
        Path(__file__).resolve().parents[1] / "concepts.csv",
        Path(__file__).resolve().parents[2] / "data" / "concepts.csv",
    ]:
        if candidate.exists():
            break
    else:
        return {}

    aliases: Dict[str, str] = {}
    with candidate.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            concept = (row.get("Concept") or "").strip()
            if not concept:
                continue
            concept_key = _normalize_concept_key(concept)
            aliases.setdefault(concept_key, concept)
            variations = (row.get("Variations") or "").split(",")
            for variation in variations:
                variation = variation.strip()
                if not variation:
                    continue
                aliases.setdefault(_normalize_concept_key(variation), concept)
    return aliases


def _resolve_concept_alias(object_name: str) -> Optional[str]:
    key = _normalize_concept_key(object_name)
    if not key:
        return None
    return _load_concept_aliases().get(key)


def _resolve_object_dir(base_dir: Path, object_name: str) -> Path:
    candidates = [object_name.strip()]
    normalized = _normalize_object_name(object_name)
    if normalized and normalized not in candidates:
        candidates.append(normalized)
    for candidate in candidates:
        candidate_path = base_dir / candidate
        if candidate_path.exists():
            return candidate_path
    return base_dir / object_name


def _replacement_dir_name(replacement: str) -> str:
    return replacement.replace(" ", "_")


def collect_slot_images(
    data_root: Path,
    slot_values: Dict[str, str],
    slot_replacements: Dict[str, str],
    max_images_per_slot: int = 0,
    start_image_index: int = 0,
) -> Dict[str, List[Path]]:
    slot_images: Dict[str, List[Path]] = {}
    attack_root = data_root / "attacks"
    for slot in SLOT_ORDER:
        value = slot_values.get(slot)
        if not value:
            continue
        replacement = slot_replacements.get(slot)
        if not replacement:
            raise ValueError(f"Missing replacement mapping for {slot}.")
        object_dir = _resolve_object_dir(attack_root, value)
        if not object_dir.exists():
            alias = _resolve_concept_alias(value)
            if alias:
                object_dir = _resolve_object_dir(attack_root, alias)
        attack_dir = object_dir / "replace_with_object" / _replacement_dir_name(replacement)
        if not attack_dir.exists():
            raise FileNotFoundError(f"Attack directory not found for {slot} ({value}): {attack_dir}")
        images = _collect_images(attack_dir)
        if max_images_per_slot and max_images_per_slot > 0:
            start = max(0, min(start_image_index, len(images) - 1))
            images = images[start : start + max_images_per_slot]
        if not images:
            raise FileNotFoundError(f"No images found for {slot} ({value}) under {attack_dir}")
        slot_images[slot] = images
    if not slot_images:
        raise ValueError("No slot images found. Provide at least one populated slot.")
    return slot_images


def run_attack_for_images(
    image_paths: List[str],
    prompt: str,
    provider: str,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    quiet: bool = False,
) -> str:
    if not image_paths:
        raise ValueError("image_paths must include at least one image")

    model_name = model or DEFAULT_MODELS.get(provider)
    if not model_name:
        raise ValueError(f"No model provided and no default available for provider '{provider}'")

    if not quiet:
        print(f"[START] provider={provider} model={model_name}")
        print(f"[IMAGES] {len(image_paths)} image(s) provided")
        print("[RUN] Sent all images as a single multi-image request")

    if provider == "ollama":
        resolved_key = api_key or os.environ.get("OLLAMA_API_KEY")
        if not resolved_key:
            raise ValueError("OLLAMA_API_KEY not provided")

        response = analyze_multiple_images_ollama(
            image_paths=image_paths,
            prompt=prompt,
            model=model_name,
            api_key=resolved_key,
            verbose=not quiet,
        )
    elif provider == "openrouter":
        resolved_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not resolved_key:
            raise ValueError("OPENROUTER_API_KEY not provided")

        response = analyze_multiple_images_openrouter(
            image_paths=image_paths,
            prompt=prompt,
            model=model_name,
            api_key=resolved_key,
            verbose=not quiet,
        )
    else:
        raise ValueError(f"Unsupported provider: {provider}")

    if not quiet:
        print("\n[COMPLETE] Received response for batch")

    return response
