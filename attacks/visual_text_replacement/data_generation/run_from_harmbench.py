#!/usr/bin/env python3
"""
Visual Text Replacement Pipeline using Harmbench Neutralized CSV

This script reads harmbench_neutralized.csv and:
1. Extracts all unique objects from X1, X2, X3, X4 columns
2. Searches for real-world references (books, movies, etc.) containing each object
3. Downloads reference images
4. Generates missing images using REVE Create API
5. Creates attack images replacing the object text with X1, X2, X3, or X4

Output structure:
    ./data/visual_text_replacement_final/
        base/{object}/              # Base images
        attacks/{object}/text_replacement/X1/   # Attacked images (X1, X2, X3, or X4)
        references.json             # Search results
        manifest.json               # Object -> image paths mapping

Usage:
    # Full pipeline
    python run_from_harmbench.py --csv harmbench_neutralized.csv

    # Step-by-step
    python run_from_harmbench.py --csv harmbench_neutralized.csv --step search
    python run_from_harmbench.py --csv harmbench_neutralized.csv --step download
    python run_from_harmbench.py --csv harmbench_neutralized.csv --step generate
    python run_from_harmbench.py --csv harmbench_neutralized.csv --step attack
"""

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# Import local modules
from .search_references import process_config as search_references
from .download_reference_images import process_references
from .generate_base_image import generate_missing_images, sanitize_filename
from .apply_text_attack import process_object


def load_harmbench_csv(csv_path: Path) -> List[Dict[str, str]]:
    """Load harmbench_neutralized CSV into a list of dictionaries."""
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def extract_objects_with_xn(rows: List[Dict[str, str]]) -> Dict[str, Set[str]]:
    """
    Extract all unique objects from X1-X4 columns and track which Xn they belong to.
    
    Returns:
        Dict mapping object name -> set of Xn keys (e.g., {"parents": {"X1"}, "honey": {"X3"}})
    """
    objects_to_xn: Dict[str, Set[str]] = {}
    
    for row in rows:
        for xn in ["X1", "X2", "X3", "X4"]:
            val = row.get(xn, "").strip()
            if val:
                if val not in objects_to_xn:
                    objects_to_xn[val] = set()
                objects_to_xn[val].add(xn)
    
    return objects_to_xn


def run_search_step(objects: List[str], output_dir: Path) -> Path:
    """Step 1: Search for references containing target text."""
    print("\n" + "="*60)
    print("STEP 1: Searching for Real-World References")
    print("="*60)
    print(f"Objects to search: {len(objects)}")
    
    references_path = output_dir / "references.json"
    
    # Check for existing references
    existing_refs = {}
    if references_path.exists():
        with open(references_path) as f:
            existing_refs = json.load(f)
        print(f"Loaded {len(existing_refs)} existing references")
    
    # Create temp config file for search
    temp_config = {"objects": objects}
    temp_config_path = output_dir / "_temp_config.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(temp_config_path, "w") as f:
        json.dump(temp_config, f)
    
    # Run search
    search_references(
        config_path=temp_config_path,
        output_path=references_path,
        tmdb_key=os.environ.get("TMDB_API_KEY"),
        omdb_key=os.environ.get("OMDB_API_KEY"),
        google_books_key=os.environ.get("GOOGLE_BOOKS_API_KEY"),
    )
    
    # Clean up temp file
    try:
        temp_config_path.unlink()
    except FileNotFoundError:
        pass
    
    return references_path


def run_download_step(
    references_path: Path,
    output_dir: Path,
    num_images: int = 3,
) -> Path:
    """Step 2: Download reference images."""
    print("\n" + "="*60)
    print("STEP 2: Downloading Reference Images")
    print("="*60)
    
    base_dir = output_dir / "base"
    
    results = process_references(
        references_path=references_path,
        output_dir=base_dir,
        serpapi_key=os.environ.get("SERPAPI_KEY"),
        bing_key=os.environ.get("BING_SEARCH_KEY"),
        num_images=num_images,
        prefer_direct=True,
    )
    
    total_downloaded = sum(len(v) for v in results.values())
    print(f"\nTotal images downloaded: {total_downloaded}")
    
    # Save manifest
    manifest_path = base_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Manifest saved to: {manifest_path}")
    
    return manifest_path


def run_generate_step(
    manifest_path: Path,
    output_dir: Path,
    objects: List[str],
    api_key: str,
    num_images: int = 3,
) -> Path:
    """Step 2b: Generate missing images using REVE Create API."""
    print("\n" + "="*60)
    print("STEP 2b: Generating Missing Images")
    print("="*60)
    
    base_dir = output_dir / "base"
    
    # Load existing manifest
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
    else:
        manifest = {}
    
    # Generate missing images
    updated_manifest = generate_missing_images(
        manifest=manifest,
        objects=objects,
        base_dir=base_dir,
        api_key=api_key,
        num_images=num_images,
    )
    
    # Save updated manifest
    with open(manifest_path, "w") as f:
        json.dump(updated_manifest, f, indent=2)
    
    total = sum(len(v) for v in updated_manifest.values())
    complete = sum(1 for v in updated_manifest.values() if len(v) >= num_images)
    print(f"\nTotal images: {total}")
    print(f"Objects with {num_images}+ images: {complete}/{len(objects)}")
    print(f"Manifest saved to: {manifest_path}")
    
    return manifest_path


def run_attack_step(
    manifest_path: Path,
    output_dir: Path,
    objects_to_xn: Dict[str, Set[str]],
    api_key: str,
) -> Dict[str, Dict[str, List[str]]]:
    """
    Step 3: Apply text replacement attack to images.
    
    Each object is replaced with its corresponding Xn (X1, X2, X3, or X4).
    If an object appears in multiple Xn columns, create attacks for each.
    """
    print("\n" + "="*60)
    print("STEP 3: Applying Text Replacement Attack")
    print("="*60)
    
    if not manifest_path.exists():
        print(f"Error: Manifest not found at {manifest_path}")
        return {}
    
    with open(manifest_path) as f:
        manifest = json.load(f)
    
    base_dir = manifest_path.parent
    attack_dir = output_dir / "attacks"
    
    all_results = {}
    
    # Process each object with its Xn replacement(s)
    for object_name, xn_set in objects_to_xn.items():
        if object_name not in manifest:
            print(f"\n=== {object_name} === (no base images, skipping)")
            continue
        
        image_paths = manifest[object_name]
        if not image_paths:
            print(f"\n=== {object_name} === (empty image list, skipping)")
            continue
        
        for xn in sorted(xn_set):
            print(f"\n=== {object_name} -> {xn} ===")
            
            result = process_object(
                object_name=object_name,
                image_paths=image_paths,
                base_dir=base_dir,
                output_dir=attack_dir,
                replacement=xn,
                api_key=api_key,
                redo_existing=False,
            )
            
            if xn not in all_results:
                all_results[xn] = {}
            all_results[xn][object_name] = result["attacked"]
    
    # Save combined attack manifest
    attack_manifest_path = attack_dir / "manifest.json"
    attack_dir.mkdir(parents=True, exist_ok=True)
    with open(attack_manifest_path, "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nAttack manifest saved to: {attack_manifest_path}")
    
    return all_results


def run_full_pipeline(
    csv_path: Path,
    output_dir: Path,
    api_key: Optional[str] = None,
    num_images: int = 3,
):
    """Run the complete pipeline from harmbench CSV."""
    
    # Load CSV and extract objects
    rows = load_harmbench_csv(csv_path)
    objects_to_xn = extract_objects_with_xn(rows)
    objects = list(objects_to_xn.keys())
    
    print(f"Loaded {len(rows)} behaviors from {csv_path}")
    print(f"Extracted {len(objects)} unique objects")
    print(f"Output directory: {output_dir}")
    
    # Show object -> Xn mapping summary
    print("\nObject -> Xn mapping:")
    for obj, xns in sorted(objects_to_xn.items()):
        print(f"  {obj}: {', '.join(sorted(xns))}")
    
    # Get API key
    if api_key is None:
        api_key = os.environ.get("REVE_API_KEY")
    
    if not api_key:
        print("Error: REVE_API_KEY not set")
        sys.exit(1)
    
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Search
    references_path = run_search_step(objects, output_dir)
    
    # Step 2: Download
    manifest_path = run_download_step(references_path, output_dir, num_images)
    
    # Step 2b: Generate missing
    manifest_path = run_generate_step(manifest_path, output_dir, objects, api_key, num_images)
    
    # Step 3: Attack with Xn replacements
    results = run_attack_step(manifest_path, output_dir, objects_to_xn, api_key)
    
    # Final summary
    print("\n" + "="*60)
    print("PIPELINE COMPLETE")
    print("="*60)
    print(f"Output directory: {output_dir}")
    print(f"Base images: {output_dir}/base/")
    print(f"Attacked images: {output_dir}/attacks/")
    
    for xn, obj_results in sorted(results.items()):
        total = sum(len(v) for v in obj_results.values())
        print(f"  {xn}: {total} attacked images across {len(obj_results)} objects")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visual Text Replacement Pipeline from Harmbench CSV"
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="harmbench_neutralized.csv",
        help="Path to harmbench_neutralized CSV file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data/visual_text_replacement_final",
        help="Output directory (default: ./data/visual_text_replacement_final)"
    )
    parser.add_argument(
        "--step",
        type=str,
        choices=["search", "download", "generate", "attack", "full"],
        default="full",
        help="Which step to run (default: full pipeline)"
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=3,
        help="Number of images per object (default: 3)"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.environ.get("REVE_API_KEY"),
        help="REVE API key"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    csv_path = Path(args.csv)
    if not csv_path.exists():
        # Try looking in the same directory as this script
        script_dir = Path(__file__).parent
        csv_path = script_dir / args.csv
        if not csv_path.exists():
            print(f"Error: CSV file not found: {args.csv}")
            sys.exit(1)
    
    output_dir = Path(args.output_dir).resolve()
    
    # Load CSV and extract objects
    rows = load_harmbench_csv(csv_path)
    objects_to_xn = extract_objects_with_xn(rows)
    objects = list(objects_to_xn.keys())
    
    print(f"Loaded {len(rows)} behaviors from {csv_path}")
    print(f"Extracted {len(objects)} unique objects")
    
    if args.step == "search":
        run_search_step(objects, output_dir)
        
    elif args.step == "download":
        references_path = output_dir / "references.json"
        if not references_path.exists():
            print(f"Error: References file not found: {references_path}")
            print("Run the 'search' step first.")
            sys.exit(1)
        run_download_step(references_path, output_dir, args.num_images)
        
    elif args.step == "generate":
        manifest_path = output_dir / "base" / "manifest.json"
        if not manifest_path.exists():
            print(f"Error: Manifest not found: {manifest_path}")
            print("Run the 'download' step first.")
            sys.exit(1)
        
        if not args.api_key:
            print("Error: REVE_API_KEY not set")
            sys.exit(1)
        
        run_generate_step(manifest_path, output_dir, objects, args.api_key, args.num_images)
        
    elif args.step == "attack":
        manifest_path = output_dir / "base" / "manifest.json"
        if not manifest_path.exists():
            print(f"Error: Manifest not found: {manifest_path}")
            print("Run the 'download' step first.")
            sys.exit(1)
        
        if not args.api_key:
            print("Error: REVE_API_KEY not set")
            sys.exit(1)
        
        run_attack_step(manifest_path, output_dir, objects_to_xn, args.api_key)
        
    else:  # full pipeline
        run_full_pipeline(
            csv_path=csv_path,
            output_dir=output_dir,
            api_key=args.api_key,
            num_images=args.num_images,
        )


if __name__ == "__main__":
    main()
