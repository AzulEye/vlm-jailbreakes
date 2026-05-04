#!/usr/bin/env python3
"""
Visualize generated images in a grid.

Columns: each base image for a concept (object).
Rows: base (original) followed by configured attacks (including per-replacement variants).

Assumes directory structure produced by data_generation/visual_replacement/run.py:
  <root>/base/<object>/<object>_base_<i>.png
  <root>/attacks/<object>/<attack_type>/<replacement?>/<object>_attack_<attack_type>_<replacement?>_<i>.png

Saves a grid image to ./figures/data_generated.png by default.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image


def load_config(config_path: Path) -> Dict:
    with config_path.open("r") as f:
        return json.load(f)


def expected_attack_rows(attacks_cfg: List[Dict]) -> List[Tuple[str, str]]:
    """
    Build a list of (attack_type, replacement_or_empty) for row ordering.
    replacement_or_empty is "" for non-replacement attacks.
    """
    rows = [("base", "")]
    for attack in attacks_cfg:
        atype = attack["type"]
        if atype == "replace_with_object":
            for repl in attack.get("replacements", []):
                rows.append((atype, repl))
        else:
            rows.append((atype, ""))
    return rows


def find_image(root: Path, obj: str, row: Tuple[str, str], idx: int) -> Path:
    atype, repl = row
    if atype == "base":
        return root / "base" / obj / f"{obj}_base_{idx}.png"

    if atype == "replace_with_object":
        dir_name = repl.replace(" ", "_")
        return (
            root
            / "attacks"
            / obj
            / atype
            / dir_name
            / f"{obj}_attack_{atype}_{repl}_{idx}.png"
        )

    return root / "attacks" / obj / atype / f"{obj}_attack_{atype}_{idx}.png"


def load_images_matrix(
    root: Path,
    objects: List[str],
    attacks_cfg: List[Dict],
    count_per_object: int,
) -> List[List[Image.Image]]:
    rows_def = expected_attack_rows(attacks_cfg)
    grid: List[List[Image.Image]] = []
    missing = []

    for row in rows_def:
        row_imgs = []
        for obj in objects:
            # one column per base image index
            for idx in range(1, count_per_object + 1):
                path = find_image(root, obj, row, idx)
                if path.exists():
                    row_imgs.append(Image.open(path).convert("RGB"))
                else:
                    row_imgs.append(None)
                    missing.append(str(path))
        grid.append(row_imgs)
    return grid, rows_def, missing


def compose_grid(
    grid: List[List[Image.Image]],
    rows_def: List[Tuple[str, str]],
    objects: List[str],
    count_per_object: int,
    output_path: Path,
    max_bytes: int = 10 * 1024 * 1024,
):
    # Determine cell width/height from first available image
    sample = next((img for row in grid for img in row if img is not None), None)
    if sample is None:
        raise SystemExit("No images found to visualize.")

    cols = len(objects) * count_per_object
    rows = len(rows_def)

    # Compute a scale factor so the final canvas is ~<= max_bytes assuming 3 bytes per pixel.
    cell_w, cell_h = sample.size
    target_pixels = max_bytes // 3
    target_cell_area = max(target_pixels // max(cols * rows, 1), 1)
    current_cell_area = cell_w * cell_h
    scale = min(1.0, (target_cell_area / current_cell_area) ** 0.5)
    new_cell_w = max(int(cell_w * scale), 1)
    new_cell_h = max(int(cell_h * scale), 1)

    canvas = Image.new(
        "RGB", (new_cell_w * cols, new_cell_h * rows), color=(255, 255, 255)
    )

    for r, row_imgs in enumerate(grid):
        for c, img in enumerate(row_imgs):
            if img is None:
                continue
            if scale < 1.0:
                img = img.resize((new_cell_w, new_cell_h), Image.LANCZOS)
            canvas.paste(img, (c * new_cell_w, r * new_cell_h))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    return cols, rows, (new_cell_w, new_cell_h)


def main():
    parser = argparse.ArgumentParser(description="Visualize generated data grid.")
    parser.add_argument(
        "--config",
        type=str,
        default="data_generation/visual_replacement/config.json",
        help="Path to pipeline config (to determine objects/attacks/count).",
    )
    parser.add_argument(
        "--root",
        type=str,
        default="data",
        help="Root directory containing base/ and attacks/ folders.",
    )
    parser.add_argument(
        "--objects",
        nargs="+",
        default=["molotov cocktail"],
        help="Objects to include. Defaults to ['molotov cocktail'] to avoid plotting all objects.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="figures/data_generated.png",
        help="Output path for the grid image.",
    )
    parser.add_argument(
        "--max-bytes",
        type=int,
        default=10 * 1024 * 1024,
        help="Maximum approximate output size in bytes (uncompressed) to scale grid within.",
    )
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    configured_objects = cfg.get("objects", [])
    if args.objects:
        # Only keep requested objects that are present in the config
        objects = [obj for obj in args.objects if obj in configured_objects]
        if not objects:
            raise SystemExit("None of the requested objects are present in the config.")
    else:
        objects = configured_objects
    attacks_cfg = cfg["attacks"]
    count = cfg["base_generation"].get("count_per_object", 1)

    root = Path(args.root)
    grid, rows_def, missing = load_images_matrix(root, objects, attacks_cfg, count)
    cols, rows, cell_size = compose_grid(
        grid,
        rows_def,
        objects,
        count,
        Path(args.output),
        max_bytes=args.max_bytes,
    )

    if missing:
        print(f"Warning: {len(missing)} image(s) missing, left blank in grid.")
    print(f"Grid saved: {args.output} | rows={rows} cols={cols} | cell={cell_size[0]}x{cell_size[1]}")


if __name__ == "__main__":
    main()
