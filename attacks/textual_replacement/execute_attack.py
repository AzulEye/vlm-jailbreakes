#!/usr/bin/env python3
"""Helpers for running textual replacement attacks (pure-text baseline)."""

import csv
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

from openai import OpenAI

from attacks.textual_replacement.attack_prompts import SLOT_ORDER

DEFAULT_MODELS = {
    "ollama": "qwen3-vl:235b-instruct",
    "openrouter": "qwen/qwen3-vl-32b-instruct",
}


def _clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip())


def _normalize_concept_key(value: str) -> str:
    return _clean_text(value).replace("/", " ").lower()


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
    normalized = re.sub(r"\s+", " ", object_name.replace("/", " ")).strip()
    if normalized and normalized not in candidates:
        candidates.append(normalized)
    for candidate_name in candidates:
        candidate_path = base_dir / candidate_name
        if candidate_path.exists():
            return candidate_path
    return base_dir / object_name


def _replacement_dir_name(replacement: str) -> str:
    return replacement.replace(" ", "_")


def _load_sentences(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"Sentences file not found: {path}")
    lines = [
        _clean_text(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not lines:
        raise ValueError(f"No sentences found in {path}")
    return lines


def collect_slot_texts(
    data_root: Path,
    slot_values: Dict[str, str],
    slot_replacements: Dict[str, str],
    max_texts_per_slot: int = 0,
) -> Dict[str, List[str]]:
    slot_texts: Dict[str, List[str]] = {}
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
        sentences_path = attack_dir / "sentences.txt"
        texts = _load_sentences(sentences_path)
        if max_texts_per_slot and max_texts_per_slot > 0:
            texts = texts[:max_texts_per_slot]
        if not texts:
            raise FileNotFoundError(f"No texts found for {slot} ({value}) under {attack_dir}")
        slot_texts[slot] = texts
    if not slot_texts:
        raise ValueError("No slot texts found. Provide at least one populated slot.")
    return slot_texts


def _label_texts(texts: List[str]) -> List[str]:
    return [f"Text {idx}: {text}" for idx, text in enumerate(texts, 1)]


def run_attack_for_texts(
    texts: List[str],
    prompt: str,
    provider: str,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    quiet: bool = False,
) -> str:
    if not texts:
        raise ValueError("texts must include at least one text snippet")

    model_name = model or DEFAULT_MODELS.get(provider)
    if not model_name:
        raise ValueError(f"No model provided and no default available for provider '{provider}'")

    labeled_texts = _label_texts(texts)
    if not quiet:
        print(f"[START] provider={provider} model={model_name}")
        print(f"[TEXTS] {len(labeled_texts)} text snippet(s) provided")
        print("[RUN] Sent all texts as a single multi-text request")

    if provider == "ollama":
        resolved_key = api_key or os.environ.get("OLLAMA_API_KEY")
        if not resolved_key:
            raise ValueError("OLLAMA_API_KEY not provided")
        try:
            from ollama import Client
        except ImportError:
            raise ImportError("ollama package required for ollama provider: pip install ollama")
        client = Client(host="https://ollama.com", headers={"Authorization": f"Bearer {resolved_key}"})
        full_prompt = prompt + "\n\n" + "\n".join(labeled_texts)
        response = client.chat(
            model=model_name,
            messages=[{"role": "user", "content": full_prompt}],
            options={"temperature": 0.0},
        )
        content = response.get("message", {}).get("content", "")
    elif provider == "openrouter":
        resolved_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not resolved_key:
            raise ValueError("OPENROUTER_API_KEY not provided")
        client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=resolved_key)
        content_parts = [{"type": "text", "text": prompt}]
        for text in labeled_texts:
            content_parts.append({"type": "text", "text": text})
        resp = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": content_parts}],
            max_tokens=2000,
            temperature=0.0,
            extra_body={"include_reasoning": False},
        )
        content = resp.choices[0].message.content or ""
    else:
        raise ValueError(f"Unsupported provider: {provider}")

    if not quiet:
        print("\n[COMPLETE] Received response for batch")

    return content
