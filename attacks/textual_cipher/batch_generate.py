#!/usr/bin/env python3
"""Batch-generate text-based cipher legends and encoded sentences from a CSV of behaviors.

For each row, creates a subdirectory with:
- legend.txt (human-readable word->nonsense mapping)
- sentence.txt (encoded sentence using nonsense words)
- codebook.json (machine-readable mapping)
- metadata.json (tokens + original row data)

Expected CSV columns: Behavior (text), optional BehaviorID. Extra columns are carried into metadata.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from pathlib import Path
from typing import Dict, List, Sequence

# Syllable components for generating pronounceable nonsense words
CONSONANTS = ['b', 'c', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'm', 'n', 'p', 'r', 's', 't', 'v', 'w', 'z']
VOWELS = ['a', 'e', 'i', 'o', 'u']
CONSONANT_CLUSTERS = ['bl', 'br', 'ch', 'cl', 'cr', 'dr', 'fl', 'fr', 'gl', 'gr', 'pl', 'pr', 'sc', 'sh', 'sk', 'sl', 'sm', 'sn', 'sp', 'st', 'sw', 'th', 'tr', 'tw', 'wh', 'wr']

# Distractor words (same as visual cipher)
DISTRACTOR_WORDS: List[str] = [
    "river", "mountain", "ocean", "forest", "desert", "island", "valley", "bridge",
    "tower", "garden", "window", "mirror", "door", "pencil", "notebook", "lantern",
    "camera", "compass", "ladder", "rocket", "planet", "galaxy", "meteor", "comet",
    "cloud", "rain", "snow", "thunder", "storm", "breeze", "shadow", "flame",
    "ember", "cliff", "canyon", "harbor", "village", "castle", "market", "library",
    "museum", "theater", "stadium", "bicycle", "train", "boat", "anchor", "harp",
    "violin", "drum", "canvas", "palette", "statue", "fountain", "laneway", "courtyard",
    "plaza", "gardenia", "orchid", "maple", "cedar", "walnut", "amber", "onyx",
    "quartz", "silver", "copper", "marble", "granite", "linen", "velvet", "leather",
    "ceramic", "clay", "paper", "ink", "chalk", "sketch", "signal", "beacon",
    "harvest", "orchard", "meadow", "pasture", "cottage", "barn", "windmill", "lighthouse",
    "harpoon", "helmet", "shield", "compass", "beaker", "goggles", "satchel", "quiver",
    "anvil", "plank", "scroll", "torch", "whistle",
]


def generate_nonsense_word(rng: random.Random, min_syllables: int = 2, max_syllables: int = 4) -> str:
    """Generate a pronounceable nonsense word using CV patterns."""
    num_syllables = rng.randint(min_syllables, max_syllables)
    word = ""
    
    for i in range(num_syllables):
        # First syllable might start with a cluster
        if i == 0 and rng.random() < 0.3:
            word += rng.choice(CONSONANT_CLUSTERS)
        else:
            word += rng.choice(CONSONANTS)
        
        word += rng.choice(VOWELS)
        
        # Sometimes add a consonant at the end of syllable
        if rng.random() < 0.4:
            word += rng.choice(CONSONANTS)
    
    # Capitalize first letter
    return word.capitalize()


def generate_unique_nonsense_words(rng: random.Random, count: int, existing: set[str] | None = None) -> List[str]:
    """Generate a list of unique nonsense words."""
    words: List[str] = []
    seen = existing.copy() if existing else set()
    
    attempts = 0
    max_attempts = count * 10
    
    while len(words) < count and attempts < max_attempts:
        word = generate_nonsense_word(rng)
        if word.lower() not in seen:
            words.append(word)
            seen.add(word.lower())
        attempts += 1
    
    if len(words) < count:
        # Fallback: add numbered words
        for i in range(count - len(words)):
            words.append(f"Xeno{i:03d}")
    
    return words


def tokenize(text: str) -> List[str]:
    """Extract lowercase alphanumeric tokens from text."""
    return re.findall(r"[a-z0-9']+", text.lower())


def unique(seq: Sequence[str]) -> List[str]:
    """Return unique items preserving order."""
    seen = set()
    out: List[str] = []
    for item in seq:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def build_vocab_tokens(tokens: List[str], distractor_count: int, rng: random.Random) -> List[str]:
    """Build vocabulary: sentence tokens + distractors."""
    vocab_tokens = list(tokens)
    pool = [w for w in DISTRACTOR_WORDS if w not in vocab_tokens]
    if distractor_count > len(pool):
        pool.extend([f"extra{idx}" for idx in range(distractor_count - len(pool))])
    vocab_tokens.extend(rng.sample(pool, k=distractor_count))
    return vocab_tokens


def slugify(text: str, fallback: str, index: int, used: set[str]) -> str:
    """Create a URL-safe slug from text."""
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


def build_codebook(tokens: List[str], distractor_count: int, rng: random.Random) -> Dict[str, str]:
    """Build mapping from words to nonsense words."""
    vocab_tokens = build_vocab_tokens(tokens, distractor_count, rng)
    nonsense_words = generate_unique_nonsense_words(rng, len(vocab_tokens))
    
    codebook = {}
    for word, nonsense in zip(vocab_tokens, nonsense_words):
        codebook[word] = nonsense
    
    return codebook


def render_legend_text(codebook: Dict[str, str], rng: random.Random) -> str:
    """Render the legend as text (shuffled order)."""
    items = list(codebook.items())
    rng.shuffle(items)
    lines = [f"{word}: {nonsense}" for word, nonsense in items]
    return "\n".join(lines)


def render_sentence_text(tokens: List[str], codebook: Dict[str, str]) -> str:
    """Render the encoded sentence as a sequence of nonsense words."""
    encoded = [codebook[token] for token in tokens]
    return " ".join(encoded)


def process_row(
    row: Dict[str, str],
    index: int,
    args: argparse.Namespace,
    used_slugs: set[str],
) -> None:
    """Process a single CSV row and generate cipher files."""
    if (row.get("FunctionalCategory") or "").strip().lower() != "standard":
        return
    behavior = (row.get("Behavior") or "").strip()
    if not behavior:
        return

    tokens = unique(tokenize(behavior))
    if not tokens:
        return

    behavior_id = (row.get("BehaviorID") or "").strip()
    slug = slugify(behavior_id or behavior, fallback="item", index=index, used=used_slugs)
    out_dir = args.output_root / slug
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed + index)
    auto_distractors = len(tokens)
    distractor_count = auto_distractors if args.distractor_count < 0 else args.distractor_count

    codebook = build_codebook(tokens, distractor_count, rng)
    
    legend_path = out_dir / "legend.txt"
    sentence_path = out_dir / "sentence.txt"
    codebook_json = out_dir / "codebook.json"
    metadata_json = out_dir / "metadata.json"

    # Render and save legend
    legend_text = render_legend_text(codebook, rng)
    legend_path.write_text(legend_text, encoding="utf-8")
    
    # Render and save encoded sentence
    sentence_text = render_sentence_text(tokens, codebook)
    sentence_path.write_text(sentence_text, encoding="utf-8")
    
    # Save machine-readable codebook
    codebook_data = {
        "codebook": [{"word": w, "nonsense": n} for w, n in codebook.items()],
        "cipher_type": "textual",
    }
    codebook_json.write_text(json.dumps(codebook_data, indent=2), encoding="utf-8")
    
    # Save metadata
    meta = {
        "behavior": behavior,
        "behavior_id": behavior_id,
        "tokens": tokens,
        "encoded_sentence": sentence_text,
        "legend_tokens": list(codebook.keys()),
        "row_index": index,
        "csv": str(args.csv),
        "cipher_type": "textual",
    }
    for k, v in row.items():
        if k == "Behavior":
            continue
        meta.setdefault(k, v)
    
    metadata_json.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote {slug}: {metadata_json}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch-generate textual cipher legends and sentences from a CSV.")
    parser.add_argument("--csv", type=Path, default=Path("harmbench_behaviors_text_test.csv"))
    parser.add_argument("--output-root", type=Path, default=Path("results/attacks/textual_cipher"))
    parser.add_argument("--seed", type=int, default=0, help="Seed for nonsense word generation.")
    parser.add_argument(
        "--distractor-count",
        type=int,
        default=-1,
        help="Number of distractor words per legend. -1 = auto (match sentence length).",
    )
    args = parser.parse_args()

    if not args.csv.exists():
        raise SystemExit(f"CSV not found: {args.csv}")

    args.output_root.mkdir(parents=True, exist_ok=True)

    with args.csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        used_slugs: set[str] = set()
        for idx, row in enumerate(reader):
            process_row(row, idx, args, used_slugs)


if __name__ == "__main__":
    main()
