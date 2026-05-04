#!/usr/bin/env python3
"""
Collect/copy generated image riddles into a single new folder, grouped by target and option.

Source layout (default):
  <src_root>/<category>/<target>/option_###/images/riddle_*.png

Destination layout:
  <dst_root>/<category>/<target>/
    option_000_riddle_000.png
    option_000_riddle_001.png   (if exists)
    option_001_riddle_000.png
    ...

Also writes:
  <dst_root>/index.json  (provenance + missing list)

Example:
  python attacks/analogy/collect_riddle_images.py \
    --src-root results_imageriddle_v2/attacks/analogy \
    --dst-root results_imageriddle_v2/analogy_riddle_images_by_target
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description="Collect image riddles into a single folder (offline file copy).")
    p.add_argument("--src-root", type=str, required=True, help="Source root (contains <category>/<target>/option_###/)")
    p.add_argument("--dst-root", type=str, required=True, help="Destination root to create/populate")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing files in destination")
    args = p.parse_args()

    src_root = Path(args.src_root).resolve()
    dst_root = Path(args.dst_root).resolve()
    if not src_root.exists():
        raise SystemExit(f"src root not found: {src_root}")
    dst_root.mkdir(parents=True, exist_ok=True)

    copied: List[Dict[str, str]] = []
    missing: List[Dict[str, str]] = []

    for cat_dir in sorted([p for p in src_root.iterdir() if p.is_dir()]):
        category = cat_dir.name
        for tgt_dir in sorted([p for p in cat_dir.iterdir() if p.is_dir()]):
            target = tgt_dir.name
            opt_dirs = sorted([p for p in tgt_dir.iterdir() if p.is_dir() and p.name.startswith("option_")])
            if not opt_dirs:
                missing.append({"category": category, "target": target, "reason": "no_option_dirs", "path": str(tgt_dir)})
                continue
            out_dir = dst_root / category / target
            out_dir.mkdir(parents=True, exist_ok=True)
            for opt_dir in opt_dirs:
                images_dir = opt_dir / "images"
                if not images_dir.exists():
                    missing.append(
                        {"category": category, "target": target, "option": opt_dir.name, "reason": "no_images_dir", "path": str(opt_dir)}
                    )
                    continue
                imgs = sorted(images_dir.glob("riddle_*.png"))
                if not imgs:
                    missing.append(
                        {"category": category, "target": target, "option": opt_dir.name, "reason": "no_riddle_png", "path": str(images_dir)}
                    )
                    continue
                for img in imgs:
                    dst_name = f"{opt_dir.name}_{img.name}"
                    dst_path = out_dir / dst_name
                    if dst_path.exists() and not args.overwrite:
                        continue
                    shutil.copy2(img, dst_path)
                    copied.append(
                        {
                            "category": category,
                            "target": target,
                            "option": opt_dir.name,
                            "src": str(img),
                            "dst": str(dst_path),
                        }
                    )

    _write_json(dst_root / "index.json", {"src_root": str(src_root), "dst_root": str(dst_root), "copied": copied, "missing": missing})
    print(f"[collect] src_root={src_root}")
    print(f"[collect] dst_root={dst_root}")
    print(f"[collect] copied_files={len(copied)} missing_records={len(missing)}")


if __name__ == "__main__":
    main()

