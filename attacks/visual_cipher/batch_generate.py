#!/usr/bin/env python3
"""Batch-generate glyph legends and sentences from a CSV of behaviors.

For each row, creates a subdirectory with:
- legend.png
- sentence.png
- codebook.json (only the glyphs used)
- metadata.json (tokens + original row data)

Expected CSV columns: Behavior (text), optional BehaviorID. Extra columns are carried into metadata.
"""

from __future__ import annotations

import argparse
import csv
import random
import re
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

try:
    from attacks.visual_cipher.visual_language_experiment import (
        GlyphSpec,
        export_codebook_json,
        export_sentence_json,
        list_image_paths,
        relative_image_path,
        render_legend,
        render_sentence,
    )
except ImportError:
    from visual_language_experiment import (
        GlyphSpec,
        export_codebook_json,
        export_sentence_json,
        list_image_paths,
        relative_image_path,
        render_legend,
        render_sentence,
    )

Color = Tuple[int, int, int]
IMAGE_MODE_TAG = "img"

# Palette (rgb, name)
PALETTE: List[Tuple[Color, str]] = [
    ((244, 144, 86), "orange"),
    ((95, 149, 230), "blue"),
    ((169, 110, 219), "purple"),
    ((77, 158, 92), "green"),
    ((244, 208, 63), "yellow"),
    ((160, 177, 207), "gray"),
    ((236, 91, 103), "red"),
    ((70, 170, 173), "teal"),
    ((231, 157, 200), "pink"),
    ((196, 165, 120), "tan"),
    ((53, 84, 138), "navy"),
    ((176, 195, 110), "olive"),
    ((240, 94, 156), "magenta"),
    ((106, 199, 194), "aqua"),
    ((232, 140, 45), "amber"),
    ((94, 90, 211), "indigo"),
    ((156, 220, 246), "sky"),
    ((192, 93, 51), "terracotta"),
]

SHAPES = ["circle", "square", "triangle", "diamond", "hex"]
MARKS = [
    "dot",
    "stripe",
    "bars",
    "ring",
    "slash",
    "arrow",
    "bar",
    "chevron",
    "chevron_down",
    "cross",
    "grid",
]

MARK_PHRASE = {
    "dot": "center dot",
    "stripe": "vertical stripe",
    "bars": "two horizontal bars",
    "ring": "ring",
    "slash": "diagonal slash",
    "arrow": "right arrow",
    "bar": "horizontal bar",
    "chevron": "chevron",
    "chevron_down": "down chevron",
    "cross": "x cross",
    "grid": "plus grid",
}

DISTRACTOR_WORDS: List[str] = [
    "river",
    "mountain",
    "ocean",
    "forest",
    "desert",
    "island",
    "valley",
    "bridge",
    "tower",
    "garden",
    "window",
    "mirror",
    "door",
    "pencil",
    "notebook",
    "lantern",
    "camera",
    "compass",
    "ladder",
    "rocket",
    "planet",
    "galaxy",
    "meteor",
    "comet",
    "cloud",
    "rain",
    "snow",
    "thunder",
    "storm",
    "breeze",
    "shadow",
    "flame",
    "ember",
    "cliff",
    "canyon",
    "harbor",
    "village",
    "castle",
    "market",
    "library",
    "museum",
    "theater",
    "stadium",
    "bicycle",
    "train",
    "boat",
    "anchor",
    "harp",
    "violin",
    "drum",
    "canvas",
    "palette",
    "statue",
    "fountain",
    "laneway",
    "courtyard",
    "plaza",
    "gardenia",
    "orchid",
    "maple",
    "cedar",
    "walnut",
    "amber",
    "onyx",
    "quartz",
    "silver",
    "copper",
    "marble",
    "granite",
    "linen",
    "velvet",
    "leather",
    "ceramic",
    "clay",
    "paper",
    "ink",
    "chalk",
    "sketch",
    "signal",
    "beacon",
    "harvest",
    "orchard",
    "meadow",
    "pasture",
    "cottage",
    "barn",
    "windmill",
    "lighthouse",
    "harpoon",
    "helmet",
    "shield",
    "compass",
    "beaker",
    "goggles",
    "satchel",
    "quiver",
    "anvil",
    "plank",
    "scroll",
    "torch",
    "whistle",
]


def apply_dir_tag(path: Path, tag: str | None) -> Path:
    if not tag:
        return path
    if path.name.endswith(f"_{tag}"):
        return path
    return path.with_name(f"{path.name}_{tag}")


def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def unique(seq: Sequence[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for item in seq:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def build_vocab_tokens(tokens: List[str], distractor_count: int, rng: random.Random) -> List[str]:
    vocab_tokens = list(tokens)
    pool = [w for w in DISTRACTOR_WORDS if w not in vocab_tokens]
    if distractor_count > len(pool):
        pool.extend([f"extra{idx}" for idx in range(distractor_count - len(pool))])
    vocab_tokens.extend(rng.sample(pool, k=distractor_count))
    return vocab_tokens


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


def build_combos() -> List[Tuple[str, str, Color, str]]:
    combos: List[Tuple[str, str, Color, str]] = []
    for color, _name in PALETTE:
        for shape in SHAPES:
            for mark in MARKS:
                combos.append((shape, mark, color, _name))
    return combos


def build_codebook(tokens: List[str], distractor_count: int, rng: random.Random) -> List[GlyphSpec]:
    combos = build_combos()
    rng.shuffle(combos)
    needed = len(tokens) + distractor_count
    if needed > len(combos):
        raise ValueError(f"Not enough glyph combos for {needed} tokens; available {len(combos)}")

    vocab_tokens = build_vocab_tokens(tokens, distractor_count, rng)

    specs: List[GlyphSpec] = []
    for token, (shape, mark, color, cname) in zip(vocab_tokens, combos):
        desc = f"{cname} {shape} with a white {MARK_PHRASE.get(mark, mark)}"
        specs.append(GlyphSpec(token, shape, color, mark, desc))
    return specs


def build_image_codebook(
    tokens: List[str],
    distractor_count: int,
    rng: random.Random,
    image_paths: List[Path],
    image_root: Path,
) -> List[GlyphSpec]:
    vocab_tokens = build_vocab_tokens(tokens, distractor_count, rng)
    needed = len(vocab_tokens)
    if needed > len(image_paths):
        raise ValueError(f"Not enough images for {needed} tokens; available {len(image_paths)}")
    chosen = rng.sample(image_paths, k=needed)
    specs: List[GlyphSpec] = []
    for token, img_path in zip(vocab_tokens, chosen):
        desc = f"image tile {img_path.stem}"
        specs.append(
            GlyphSpec(
                token,
                "image",
                (255, 255, 255),
                "image",
                desc,
                image_path=relative_image_path(img_path, image_root),
            )
        )
    return specs


def process_row(
    row: Dict[str, str],
    index: int,
    args: argparse.Namespace,
    used_slugs: set[str],
    image_paths: List[Path] | None,
    image_root: Path | None,
) -> None:
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
    if args.glyph_mode == "images":
        if not image_paths or not image_root:
            raise ValueError("Image glyph mode requires a non-empty image directory.")
        codebook = build_image_codebook(
            tokens,
            distractor_count=auto_distractors if args.distractor_count < 0 else args.distractor_count,
            rng=rng,
            image_paths=image_paths,
            image_root=image_root,
        )
    else:
        codebook = build_codebook(
            tokens,
            distractor_count=auto_distractors if args.distractor_count < 0 else args.distractor_count,
            rng=rng,
        )
    codebook_by_word = {spec.word: spec for spec in codebook}
    legend_specs = list(codebook)
    rng.shuffle(legend_specs)

    legend_path = out_dir / "legend.png"
    sentence_path = out_dir / "sentence.png"
    codebook_json = out_dir / "codebook.json"
    metadata_json = out_dir / "metadata.json"

    def auto_cols(n: int) -> int:
        return max(3, int(n ** 0.5 + 0.999))

    def auto_tile(n: int) -> int:
        # In compact mode, use medium-small tiles (80-100px). Standard mode: 120-220px
        if args.compact:
            return max(80, min(100, 100 - max(0, n - 10) * 2))
        return max(120, min(220, 220 - max(0, n - 8) * 5))

    cols = args.legend_cols if args.legend_cols > 0 else auto_cols(len(legend_specs))
    tile_size = args.tile_size if args.tile_size > 0 else auto_tile(len(legend_specs))

    render_legend(legend_specs, tile_size, cols=cols, output_path=legend_path, image_root=image_root, compact=args.compact)
    render_sentence(tokens, codebook_by_word, tile_size, output_path=sentence_path, image_root=image_root, compact=args.compact)
    export_codebook_json(
        legend_specs,
        tile_size,
        output_path=codebook_json,
        glyph_mode=args.glyph_mode,
        image_root=image_root,
    )

    meta = {
        "behavior": behavior,  # canonical copy
        "behavior_id": behavior_id,
        "legend_tokens": [spec.word for spec in legend_specs],
        "row_index": index,
        "csv": str(args.csv),
        "glyph_mode": args.glyph_mode,
    }
    if image_root:
        meta["image_root"] = str(image_root)
    for k, v in row.items():
        if k == "Behavior":
            continue  # avoid redundant copy; we keep the trimmed/normalized behavior field
        meta.setdefault(k, v)

    export_sentence_json(tokens, metadata_json, extra=meta)
    print(f"Wrote {slug}: {metadata_json}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch-generate glyph legends and sentences from a CSV.")
    parser.add_argument("--csv", type=Path, default=Path("sample_behaviors.csv"))
    parser.add_argument("--output-root", type=Path, default=Path("results/attacks/visual_cipher"))
    parser.add_argument("--tile-size", type=int, default=-1, help="Tile size px. -1 = auto per legend size.")
    parser.add_argument("--legend-cols", type=int, default=-1, help="Legend columns. -1 = auto per legend size.")
    parser.add_argument("--seed", type=int, default=0, help="Seed for glyph assignment shuffling.")
    parser.add_argument(
        "--distractor-count",
        type=int,
        default=-1,
        help="Number of distractor glyphs per legend. -1 = auto (match sentence length).",
    )
    parser.add_argument(
        "--glyph-mode",
        choices=["abstract", "images"],
        default="abstract",
        help="abstract = geometric glyphs; images = sample tiles from a directory.",
    )
    parser.add_argument(
        "--glyph-image-dir",
        type=Path,
        default=Path("assets/object_tiles"),
        help="Directory with image tiles used when --glyph-mode images.",
    )
    parser.add_argument(
        "--mode-tag",
        type=str,
        default="",
        help="Optional suffix appended to output-root when set.",
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Use compact mode: smaller tiles, larger fonts, minimal padding. Reduces image size by ~70%%.",
    )
    args = parser.parse_args()

    if not args.csv.exists():
        raise SystemExit(f"CSV not found: {args.csv}")

    mode_tag = args.mode_tag or (IMAGE_MODE_TAG if args.glyph_mode == "images" else "")
    args.output_root = apply_dir_tag(args.output_root, mode_tag)

    image_paths: List[Path] | None = None
    image_root: Path | None = None
    if args.glyph_mode == "images":
        image_root = args.glyph_image_dir
        image_paths = list_image_paths(image_root)

    args.output_root.mkdir(parents=True, exist_ok=True)

    with args.csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        used_slugs: set[str] = set()
        for idx, row in enumerate(reader):
            process_row(row, idx, args, used_slugs, image_paths, image_root)


if __name__ == "__main__":
    main()
