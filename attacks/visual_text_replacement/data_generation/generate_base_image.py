#!/usr/bin/env python3
"""
Generate images from a text prompt using the REVE create API.
Builds prompts from CREATE_TEXT_TEMPLATE for text replacement attacks.
"""

import argparse
import base64
import json
import os
import sys
import time
from io import BytesIO
from pathlib import Path
from typing import Optional, Tuple, Dict, List

import requests
from PIL import Image

from .prompts import CREATE_TEXT_TEMPLATE


def decode_base64_image(data: str) -> Image.Image:
    """Decode a base64 image string (data URLs supported)."""
    if "," in data and data.strip().startswith("data:"):
        data = data.split(",", 1)[1]
    return Image.open(BytesIO(base64.b64decode(data))).convert("RGB")


def sanitize_filename(name: str) -> str:
    """Convert a string to a safe filename."""
    # Replace problematic characters
    replacements = {
        "/": "_",
        "\\": "_",
        ":": "_",
        "*": "_",
        "?": "_",
        '"': "_",
        "<": "_",
        ">": "_",
        "|": "_",
        " ": "_",
        "'": "",
        ",": "",
        ".": "_",
    }
    result = name
    for old, new in replacements.items():
        result = result.replace(old, new)
    # Remove consecutive underscores and trim
    while "__" in result:
        result = result.replace("__", "_")
    return result.strip("_")[:100]


def request_reve_create(
    prompt: str,
    api_key: str,
    aspect_ratio: str = "3:2",
    version: str = "latest",
    timeout: int = 90,
    max_retries: int = 3,
    backoff: int = 2,
) -> dict:
    """Call REVE create API and return parsed JSON response."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    payload = {"prompt": prompt, "aspect_ratio": aspect_ratio, "version": version}

    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(
                "https://api.reve.com/v1/image/create",
                headers=headers,
                json=payload,
                timeout=timeout,
            )
            response.raise_for_status()
            try:
                return response.json()
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Failed to parse response: {exc}\nRaw: {response.text}") from exc
        except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError) as exc:
            last_error = exc
            if attempt < max_retries:
                wait = backoff ** (attempt - 1)
                print(f"    Request timed out/failed (attempt {attempt}/{max_retries}). Retrying in {wait}s...", flush=True)
                time.sleep(wait)
                continue
            raise SystemExit(f"Request failed after {max_retries} attempts: {exc}") from exc
        except requests.HTTPError as exc:
            detail = None
            try:
                detail = response.json()
            except Exception:
                detail = response.text
            raise SystemExit(f"Request failed: {exc}\nDetails: {detail}") from exc

    raise SystemExit(f"Request failed: {last_error}")


def build_prompt(target_text: str, extra: Optional[str] = None) -> str:
    """Construct the generation prompt using CREATE_TEXT_TEMPLATE."""
    base_prompt = CREATE_TEXT_TEMPLATE.format(target_text=target_text)
    if extra:
        return f"{base_prompt} {extra}"
    return base_prompt


def save_image(image_b64: str, output_path: Path) -> Tuple[Path, Image.Image]:
    """Decode base64 image and save to disk."""
    image = decode_base64_image(image_b64)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    return output_path, image


def generate_image(
    target_text: str,
    api_key: str,
    output_path: Path,
    aspect_ratio: str = "3:2",
    version: str = "latest",
    timeout: int = 90,
    max_retries: int = 3,
) -> Optional[Path]:
    """Generate a single image for the target text."""
    prompt = build_prompt(target_text)
    
    try:
        result = request_reve_create(
            prompt=prompt,
            api_key=api_key,
            aspect_ratio=aspect_ratio,
            version=version,
            timeout=timeout,
            max_retries=max_retries,
        )
        
        image_b64 = result.get("image")
        if not image_b64:
            print(f"    ✗ No image returned for '{target_text}'")
            return None
        
        saved_path, _ = save_image(image_b64, output_path)
        
        if result.get("content_violation"):
            print(f"    ⚠ Content policy warning for '{target_text}'")
        
        return saved_path
        
    except Exception as e:
        print(f"    ✗ Error generating image for '{target_text}': {e}")
        return None


def generate_missing_images(
    manifest: Dict[str, List[str]],
    objects: List[str],
    base_dir: Path,
    api_key: str,
    num_images: int = 3,
    aspect_ratio: str = "3:2",
) -> Dict[str, List[str]]:
    """
    Generate images for objects that have fewer than num_images.
    Saves progress incrementally so interrupted runs can be resumed.
    
    Args:
        manifest: Current manifest mapping object -> list of image paths
        objects: List of all objects to ensure coverage
        base_dir: Base directory for saving images
        api_key: REVE API key
        num_images: Target number of images per object
        aspect_ratio: Aspect ratio for generated images
    
    Returns:
        Updated manifest with new image paths
    """
    updated_manifest = dict(manifest)
    total_generated = 0
    manifest_path = base_dir / "manifest.json"
    
    print("\n" + "="*60)
    print("STEP 2b: Generating Missing Images with REVE Create API")
    print("="*60)
    
    # Count how many need generation
    need_gen_count = sum(1 for obj in objects if len(updated_manifest.get(obj, [])) < num_images)
    print(f"Objects needing generation: {need_gen_count}")
    
    for obj in objects:
        current_images = updated_manifest.get(obj, [])
        # Verify existing images still exist
        current_images = [p for p in current_images if Path(p).exists()]
        needed = num_images - len(current_images)
        
        if needed <= 0:
            updated_manifest[obj] = current_images
            continue
        
        print(f"\n=== {obj} === (has {len(current_images)}, need {needed} more)")
        
        obj_dir = base_dir / sanitize_filename(obj)
        obj_dir.mkdir(parents=True, exist_ok=True)
        
        for i in range(needed):
            idx = len(current_images) + i + 1
            filename = f"{sanitize_filename(obj)}_generated_{idx}.jpg"
            output_path = obj_dir / filename
            
            # Skip if file already exists
            if output_path.exists():
                current_images.append(str(output_path))
                print(f"  Image {idx}/{num_images} already exists, skipping")
                continue
            
            print(f"  Generating image {idx}/{num_images}...", end=" ", flush=True)
            
            result = generate_image(
                target_text=obj,
                api_key=api_key,
                output_path=output_path,
                aspect_ratio=aspect_ratio,
            )
            
            if result:
                current_images.append(str(result))
                total_generated += 1
                print(f"✓ Saved: {output_path.name}")
            else:
                print("✗ Failed")
            
            # Rate limiting
            time.sleep(1)
        
        updated_manifest[obj] = current_images
        
        # Save manifest incrementally after each object
        try:
            with open(manifest_path, "w") as f:
                json.dump(updated_manifest, f, indent=2)
        except IOError as e:
            print(f"Warning: Failed to save manifest: {e}")
    
    print(f"\nTotal images generated: {total_generated}")
    
    return updated_manifest


def main():
    parser = argparse.ArgumentParser(
        description="Generate images with REVE create API using CREATE_TEXT_TEMPLATE."
    )
    parser.add_argument(
        "target_text",
        type=str,
        nargs="?",
        default="bomb",
        help="Target text to generate image for (default: bomb).",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output path for image (default: ./tmp/generated_<target_text>.jpg).",
    )
    parser.add_argument("--api-key", type=str, default=None, help="REVE API key.")
    parser.add_argument(
        "--aspect-ratio",
        type=str,
        default="3:2",
        help="Aspect ratio (default: 3:2). Valid: 16:9, 9:16, 3:2, 2:3, 4:3, 3:4, 1:1.",
    )
    parser.add_argument(
        "--version",
        type=str,
        default="latest",
        help="REVE model version (default: latest).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=90,
        help="Request timeout in seconds (default: 90).",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Max retries on timeout/connection errors (default: 3).",
    )

    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("REVE_API_KEY")
    if not api_key:
        raise ValueError("REVE_API_KEY not set. Provide --api-key or export REVE_API_KEY.")

    output_path = Path(args.output) if args.output else Path(f"./tmp/generated_{sanitize_filename(args.target_text)}.jpg")
    
    print(f"Generating image for: {args.target_text}")
    print(f"Aspect ratio: {args.aspect_ratio}")
    print(f"Output: {output_path}")

    result = generate_image(
        target_text=args.target_text,
        api_key=api_key,
        output_path=output_path,
        aspect_ratio=args.aspect_ratio,
        version=args.version,
        timeout=args.timeout,
        max_retries=args.max_retries,
    )

    if result:
        print(f"\n✓ Image saved to: {result}")
    else:
        print("\n✗ Failed to generate image")
        sys.exit(1)


if __name__ == "__main__":
    main()
