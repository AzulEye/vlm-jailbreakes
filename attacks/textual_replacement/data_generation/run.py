#!/usr/bin/env python3
"""
Generate textual replacement data (base + attacks) using OpenRouter GPT-5.2.

Base:
  data/textual_replacement/base/<concept>/sentences.txt
Attacks:
  data/textual_replacement/attacks/<concept>/replace_with_object/<replacement>/sentences.txt
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from tqdm import tqdm

from openai import OpenAI

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent

DEFAULT_OPENROUTER_MODEL = "openai/gpt-5.2"
DEFAULT_CONCEPTS_CSV = REPO_ROOT / "data" / "concepts.csv"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "data" / "textual_replacement"
DEFAULT_CONCEPT_COLUMN = "Concept"
DEFAULT_SENTENCE_COUNT = 15
DEFAULT_REPLACEMENTS = ["banana", "carrot", "water bottle", "broccoli"]
DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 1200
DEFAULT_MAX_PARALLEL = 4


def _clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip())


def _parse_json(content: str) -> Optional[Dict[str, object]]:
    text = content.strip()
    if not text:
        return None
    if text.startswith("```"):
        if text.startswith("```json"):
            text = text[len("```json") :].strip()
        else:
            text = text[3:].strip()
        if text.endswith("```"):
            text = text[:-3].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        last = text.rfind("}")
        if last != -1:
            try:
                return json.loads(text[: last + 1])
            except json.JSONDecodeError:
                return None
        return None


def _chat_openrouter(
    client: OpenAI,
    model: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content or ""


def load_concepts_from_csv(path: Path, column: str, limit: int) -> List[str]:
    if not path.exists():
        raise SystemExit(f"Concepts CSV not found: {path}")

    concepts: List[str] = []
    seen = set()
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        if column not in fieldnames:
            raise SystemExit(
                f"Concepts CSV missing '{column}' column. Available: {', '.join(fieldnames)}"
            )
        for row in reader:
            concept = (row.get(column) or "").strip()
            if not concept or concept in seen:
                continue
            concepts.append(concept)
            seen.add(concept)
            if limit > 0 and len(concepts) >= limit:
                break
    return concepts


def _build_prompt(concept: str, count: int) -> str:
    return (
        "Return JSON with key 'sentences' as a list of "
        f"{count} natural, context-rich sentences. "
        "Each sentence must include the exact phrase below verbatim (case-sensitive) "
        "as a contiguous substring. Keep sentences single-line, no numbering or bullets. "
        "Avoid instructions or step-by-step guidance.\n\n"
        f"Phrase: {concept}\nJSON:"
    )


def generate_sentences(
    call: Callable[[str], str],
    concept: str,
    target_count: int,
    max_attempts: int,
) -> List[str]:
    collected: List[str] = []
    seen = set()

    for attempt in range(1, max_attempts + 1):
        remaining = target_count - len(collected)
        if remaining <= 0:
            break
        prompt = _build_prompt(concept, remaining)
        content = call(prompt)
        data = _parse_json(content) or {}
        sentences = data.get("sentences") if isinstance(data, dict) else None
        if not isinstance(sentences, list):
            continue
        for sentence in sentences:
            if not isinstance(sentence, str):
                continue
            cleaned = _clean_text(sentence)
            if not cleaned or concept not in cleaned:
                continue
            if cleaned in seen:
                continue
            collected.append(cleaned)
            seen.add(cleaned)
            if len(collected) >= target_count:
                break

    return collected


def read_sentences(path: Path) -> List[str]:
    if not path.exists():
        return []
    lines = [
        _clean_text(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return lines


def write_sentences(path: Path, sentences: Sequence[str]) -> None:
    path.write_text("\n".join(sentences), encoding="utf-8")


def replace_concept(sentences: Sequence[str], concept: str, replacement: str) -> List[str]:
    return [sentence.replace(concept, replacement) for sentence in sentences]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def parse_replacements(value: Optional[str]) -> List[str]:
    if not value:
        return list(DEFAULT_REPLACEMENTS)
    items = [item.strip() for item in value.split(",") if item.strip()]
    return items or list(DEFAULT_REPLACEMENTS)


def make_call(
    api_key: str,
    model: str,
    temperature: float,
    max_tokens: int,
) -> Callable[[str], str]:
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

    def _call(prompt: str) -> str:
        return _chat_openrouter(
            client,
            model,
            prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    return _call


def process_concept(
    concept: str,
    *,
    api_key: str,
    model: str,
    temperature: float,
    max_tokens: int,
    sentences_per_concept: int,
    max_attempts: int,
    replacements: Sequence[str],
    base_root: Path,
    attacks_root: Path,
    redo_existing: bool,
    concepts_csv: Path,
    concepts_column: str,
) -> Tuple[str, bool, str]:
    try:
        concept_dir = base_root / concept
        ensure_dir(concept_dir)
        base_sentences_path = concept_dir / "sentences.txt"
        base_meta_path = concept_dir / "metadata.json"

        base_sentences: List[str] = []
        if base_sentences_path.exists() and not redo_existing:
            base_sentences = read_sentences(base_sentences_path)
        if len(base_sentences) != sentences_per_concept:
            call = make_call(
                api_key,
                model,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            base_sentences = generate_sentences(
                call,
                concept,
                target_count=sentences_per_concept,
                max_attempts=max_attempts,
            )

        if len(base_sentences) != sentences_per_concept:
            return (
                concept,
                False,
                f"expected {sentences_per_concept} sentences, got {len(base_sentences)}",
            )

        if redo_existing or not base_sentences_path.exists():
            write_sentences(base_sentences_path, base_sentences)
            base_meta = {
                "concept": concept,
                "sentences_per_concept": sentences_per_concept,
                "concepts_csv": str(concepts_csv),
                "concepts_column": concepts_column,
                "generation_model": model,
                "generation_provider": "openrouter",
            }
            base_meta_path.write_text(json.dumps(base_meta, indent=2), encoding="utf-8")

        for replacement in replacements:
            replacement_dir = (
                attacks_root
                / concept
                / "replace_with_object"
                / replacement.replace(" ", "_")
            )
            ensure_dir(replacement_dir)
            attack_sentences_path = replacement_dir / "sentences.txt"
            attack_meta_path = replacement_dir / "metadata.json"

            if attack_sentences_path.exists() and not redo_existing:
                continue

            replaced = replace_concept(base_sentences, concept, replacement)
            write_sentences(attack_sentences_path, replaced)
            attack_meta = {
                "concept": concept,
                "replacement": replacement,
                "attack_type": "replace_with_object",
                "sentences_per_concept": len(replaced),
                "base_sentences_path": str(base_sentences_path),
            }
            attack_meta_path.write_text(json.dumps(attack_meta, indent=2), encoding="utf-8")

        return concept, True, ""
    except Exception as exc:
        return concept, False, f"error: {exc!r}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate textual replacement base + attacks data using OpenRouter."
    )
    parser.add_argument("--concepts-csv", type=Path, default=DEFAULT_CONCEPTS_CSV)
    parser.add_argument("--concepts-column", type=str, default=DEFAULT_CONCEPT_COLUMN)
    parser.add_argument(
        "--concepts-limit",
        type=int,
        default=0,
        help="Limit number of concepts loaded (0 = no limit).",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--model", type=str, default=DEFAULT_OPENROUTER_MODEL)
    parser.add_argument("--api-key", type=str, default="")
    parser.add_argument(
        "--sentences-per-concept",
        type=int,
        default=DEFAULT_SENTENCE_COUNT,
    )
    parser.add_argument(
        "--replacements",
        type=str,
        default="",
        help="Comma-separated replacements (default: banana, carrot, water bottle, broccoli).",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=DEFAULT_MAX_ATTEMPTS,
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=DEFAULT_MAX_PARALLEL,
        help="Maximum parallel worker threads.",
    )
    parser.add_argument(
        "--redo-existing",
        action="store_true",
        help="Regenerate outputs even if sentence files already exist.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Deprecated alias for default behavior (skip existing outputs).",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Deprecated alias for --concepts-csv.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Deprecated alias for --concepts-limit.",
    )
    args = parser.parse_args()

    concepts_csv = args.csv or args.concepts_csv
    concepts_limit = args.limit if args.limit is not None else args.concepts_limit

    if not concepts_csv.exists():
        raise SystemExit(f"Concepts CSV not found: {concepts_csv}")

    api_key = args.api_key or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("Missing OPENROUTER_API_KEY (or pass --api-key).")

    concepts = load_concepts_from_csv(
        concepts_csv,
        column=args.concepts_column,
        limit=concepts_limit or 0,
    )
    if not concepts:
        raise SystemExit("No concepts found. Check the CSV and column name.")

    replacements = parse_replacements(args.replacements)

    base_root = args.output_root / "base"
    attacks_root = args.output_root / "attacks"
    ensure_dir(base_root)
    ensure_dir(attacks_root)

    generated = 0
    failures: List[str] = []
    max_parallel = max(1, int(args.max_parallel))
    print(f"[EXECUTOR] Using up to {max_parallel} parallel threads.")

    if max_parallel == 1:
        for concept in tqdm(concepts, desc="Concepts", unit="concept"):
            concept_name, ok, reason = process_concept(
                concept,
                api_key=api_key,
                model=args.model,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                sentences_per_concept=args.sentences_per_concept,
                max_attempts=args.max_attempts,
                replacements=replacements,
                base_root=base_root,
                attacks_root=attacks_root,
                redo_existing=args.redo_existing,
                concepts_csv=concepts_csv,
                concepts_column=args.concepts_column,
            )
            if ok:
                generated += 1
            else:
                failures.append(f"{concept_name}: {reason}")
    else:
        with ThreadPoolExecutor(
            max_workers=max_parallel, thread_name_prefix="ConceptWorker"
        ) as executor:
            futures = {
                executor.submit(
                    process_concept,
                    concept,
                    api_key=api_key,
                    model=args.model,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    sentences_per_concept=args.sentences_per_concept,
                    max_attempts=args.max_attempts,
                    replacements=replacements,
                    base_root=base_root,
                    attacks_root=attacks_root,
                    redo_existing=args.redo_existing,
                    concepts_csv=concepts_csv,
                    concepts_column=args.concepts_column,
                ): concept
                for concept in concepts
            }
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Concepts",
                unit="concept",
            ):
                try:
                    concept_name, ok, reason = future.result()
                except Exception as exc:
                    failures.append(f"{futures[future]}: error {exc!r}")
                    continue
                if ok:
                    generated += 1
                else:
                    failures.append(f"{concept_name}: {reason}")

    print(f"Generated concepts: {generated}/{len(concepts)}")
    if failures:
        print(f"[SUMMARY] {len(failures)} concepts skipped or failed:")
        for failure in failures:
            print(f"  - {failure}")


if __name__ == "__main__":
    sys.exit(main())
