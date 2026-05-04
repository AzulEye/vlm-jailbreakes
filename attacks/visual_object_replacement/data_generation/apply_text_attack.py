#!/usr/bin/env python3
"""
Apply text replacement attack to downloaded reference images.
Reads from manifest.json and applies REVE Edit API to replace text.

Usage:
    python apply_text_attack.py --base-dir ./data/visual_replacement_search/base \
        --output-dir ./data/visual_replacement_search/attacks \
        --replacement banana
"""

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Dict, Optional

from .generate_attack_image import (
    request_reve_edit,
    decode_base64_image,
    save_image,
)
from .prompts import EDIT_TEXT_TEMPLATE


def sanitize_filename(name: str) -> str:
    """Convert string to safe filename."""
    import re
    name = re.sub(r'[^\w\s-]', '', name)
    name = re.sub(r'[-\s]+', '_', name)
    return name[:100].strip('_')


def apply_text_replacement(
    source_image: Path,
    original_text: str,
    replacement_text: str,
    output_path: Path,
    api_key: str,
    aspect_ratio: str = "16:9",
    version: str = "reve-edit-fast@20251030",
) -> bool:
    """Apply text replacement attack to a single image."""
    
    # Build the edit prompt
    prompt = EDIT_TEXT_TEMPLATE.format(
        original_text=original_text,
        replacement_text=replacement_text,
    )
    
    try:
        result = request_reve_edit(
            image_path=source_image,
            edit_instruction=prompt,
            api_key=api_key,
            aspect_ratio=aspect_ratio,
            version=version,
        )
        
        edited_b64 = result.get("image")
        if not edited_b64:
            print(f"    ✗ No image returned for {source_image.name}")
            return False
        
        edited_image = decode_base64_image(edited_b64)
        save_image(edited_image, output_path)
        
        print(f"    ✓ Saved: {output_path.name}")
        return True
        
    except Exception as e:
        print(f"    ✗ Error: {e}")
        return False


def process_object(
    object_name: str,
    image_paths: List[str],
    base_dir: Path,
    output_dir: Path,
    replacement: str,
    api_key: str,
    redo_existing: bool = False,
) -> Dict[str, List[str]]:
    """Process all images for a single object."""
    
    print(f"\n=== {object_name} ===")
    
    if not image_paths:
        print("  No images to process")
        return {"object": object_name, "attacked": [], "failed": []}
    
    # Create output directory
    attack_dir = output_dir / sanitize_filename(object_name) / "text_replacement" / replacement
    attack_dir.mkdir(parents=True, exist_ok=True)
    
    attacked = []
    failed = []
    
    for img_path_str in image_paths:
        # Handle both relative and absolute paths
        img_path = Path(img_path_str)
        if not img_path.is_absolute():
            # Try relative to base_dir parent
            img_path = base_dir.parent.parent / img_path_str
        
        if not img_path.exists():
            # Try relative to current working directory
            img_path = Path(img_path_str)
            if not img_path.exists():
                print(f"  ✗ Image not found: {img_path_str}")
                failed.append(img_path_str)
                continue
        
        # Output filename
        output_name = f"{img_path.stem}_attack_{replacement}.png"
        output_path = attack_dir / output_name
        
        if output_path.exists() and not redo_existing:
            print(f"  Skipping existing: {output_name}")
            attacked.append(str(output_path))
            continue
        
        print(f"  Processing: {img_path.name}")
        
        if apply_text_replacement(
            source_image=img_path,
            original_text=object_name,
            replacement_text=replacement,
            output_path=output_path,
            api_key=api_key,
        ):
            attacked.append(str(output_path))
        else:
            failed.append(img_path_str)
    
    return {"object": object_name, "attacked": attacked, "failed": failed}


def main():
    parser = argparse.ArgumentParser(
        description="Apply text replacement attack to reference images"
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        required=True,
        help="Directory containing base images and manifest.json"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for attacked images"
    )
    parser.add_argument(
        "--replacement",
        type=str,
        default="banana",
        help="Replacement text (default: banana)"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.environ.get("REVE_API_KEY"),
        help="REVE API key"
    )
    parser.add_argument(
        "--redo-existing",
        action="store_true",
        help="Reprocess existing attacked images"
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=1,
        help="Max parallel workers (default: 1 for rate limiting)"
    )
    parser.add_argument(
        "--objects",
        type=str,
        nargs="+",
        help="Specific objects to process (default: all)"
    )
    
    args = parser.parse_args()
    
    base_dir = Path(args.base_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    
    if not args.api_key:
        print("Error: REVE_API_KEY not set")
        sys.exit(1)
    
    # Load manifest
    manifest_path = base_dir / "manifest.json"
    if not manifest_path.exists():
        print(f"Error: manifest.json not found in {base_dir}")
        sys.exit(1)
    
    with open(manifest_path) as f:
        manifest = json.load(f)
    
    # Filter objects if specified
    if args.objects:
        manifest = {k: v for k, v in manifest.items() if k in args.objects}
    
    print(f"Processing {len(manifest)} objects")
    print(f"Replacement text: {args.replacement}")
    print(f"Output directory: {output_dir}")
    
    all_results = []
    
    for object_name, image_paths in manifest.items():
        result = process_object(
            object_name=object_name,
            image_paths=image_paths,
            base_dir=base_dir,
            output_dir=output_dir,
            replacement=args.replacement,
            api_key=args.api_key,
            redo_existing=args.redo_existing,
        )
        all_results.append(result)
    
    # Save attack manifest
    attack_manifest = {r["object"]: r["attacked"] for r in all_results}
    attack_manifest_path = output_dir / "manifest.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(attack_manifest_path, "w") as f:
        json.dump(attack_manifest, f, indent=2)
    
    # Summary
    total_attacked = sum(len(r["attacked"]) for r in all_results)
    total_failed = sum(len(r["failed"]) for r in all_results)
    
    print(f"\n=== Summary ===")
    print(f"Successfully attacked: {total_attacked}")
    print(f"Failed: {total_failed}")
    print(f"Attack manifest saved to: {attack_manifest_path}")


if __name__ == "__main__":
    main()
