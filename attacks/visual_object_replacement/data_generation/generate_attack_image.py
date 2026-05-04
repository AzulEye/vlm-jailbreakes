#!/usr/bin/env python3
"""
Generate an edited image via the REVE API and save a side-by-side comparison.
"""

import argparse
import base64
import json
import os
import sys
from io import BytesIO
from pathlib import Path

import requests
from PIL import Image

from .prompts import (
    EDIT_BBBLORK_TEMPLATE,
    EDIT_TEMPLATE,
    EDIT_TEXT_TEMPLATE,
    REMOVE_OBJECT_TEMPLATE,
)


def image_to_base64(image_path: Path) -> str:
    """Read an image file and return base64 string."""
    with image_path.open("rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def decode_base64_image(data: str) -> Image.Image:
    """Decode a base64 image string (data URLs supported)."""
    if "," in data and data.strip().startswith("data:"):
        data = data.split(",", 1)[1]
    return Image.open(BytesIO(base64.b64decode(data))).convert("RGB")


def request_reve_edit(
    image_path: Path,
    edit_instruction: str,
    api_key: str,
    aspect_ratio: str = None,
    version: str = "latest",
) -> dict:
    """Call REVE edit API and return parsed JSON response."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }

    payload = {
        "edit_instruction": edit_instruction,
        "reference_image": image_to_base64(image_path),
        "version": version,
    }
    if aspect_ratio:
        payload["aspect_ratio"] = aspect_ratio

    response = requests.post(
        "https://api.reve.com/v1/image/edit", headers=headers, json=payload, timeout=60
    )
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        detail = None
        try:
            detail = response.json()
        except Exception:
            detail = response.text
        raise SystemExit(f"Request failed: {exc}\nDetails: {detail}") from exc

    try:
        return response.json()
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Failed to parse response: {exc}\nRaw: {response.text}") from exc


def save_image(edited_image: Image.Image, output_path: Path):
    """Save edited image to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    edited_image.save(output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Edit an image with REVE and save a side-by-side comparison."
    )
    parser.add_argument("image_path", type=str, help="Path to the source image to edit.")
    parser.add_argument(
        "prompt",
        type=str,
        nargs="?",
        help="Optional extra instructions appended to the built prompt.",
    )
    parser.add_argument(
        "--attack-type",
        type=str,
        choices=["replace_with_object", "replace_with_BBBLORK", "replace_with_nothing", "text_replacement"],
        default="replace_with_object",
        help="Type of edit to perform.",
    )
    parser.add_argument(
        "--original-object",
        type=str,
        default="bomb",
        help="Name of the object to replace/remove in the image (default: bomb).",
    )
    parser.add_argument(
        "--replacement-object",
        type=str,
        default="banana",
        help="Name of the object that should replace the original object (default: banana).",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output path for side-by-side image (default: ./tmp/<input>_reve_edit.jpg).",
    )
    parser.add_argument("--api-key", type=str, default=None, help="REVE API key.")
    parser.add_argument(
        "--aspect-ratio",
        type=str,
        default="16:9",
        help="Aspect ratio for output image (default: 16:9). Valid: 16:9, 3:2, 4:3, 1:1, 3:4, 2:3, 9:16.",
    )
    parser.add_argument(
        "--version",
        type=str,
        default="reve-edit-fast@20251030",
        help="REVE model version to use (default: reve-edit-fast@20251030).",
    )

    args = parser.parse_args()

    source_path = Path(args.image_path)
    if not source_path.exists():
        raise FileNotFoundError(f"Image not found: {source_path}")

    api_key = args.api_key or os.environ.get("REVE_API_KEY")
    if not api_key:
        raise ValueError("REVE_API_KEY not set. Provide --api-key or export REVE_API_KEY.")

    # Build prompt based on attack type
    attack_type = args.attack_type
    if not args.original_object:
        raise ValueError("Provide --original-object for all attack types.")

    if attack_type == "replace_with_object":
        if not args.replacement_object:
            raise ValueError("Provide --replacement-object for replace_with_object.")
        base_prompt = EDIT_TEMPLATE.format(
            original_object=args.original_object,
            replacement_object=args.replacement_object,
        )
    elif attack_type == "replace_with_BBBLORK":
        base_prompt = EDIT_BBBLORK_TEMPLATE.format(
            original_object=args.original_object,
        )
    elif attack_type == "replace_with_nothing":
        base_prompt = REMOVE_OBJECT_TEMPLATE.format(
            original_object=args.original_object,
        )
    elif attack_type == "text_replacement":
        if not args.replacement_object:
            raise ValueError("Provide --replacement-object for text_replacement.")
        base_prompt = EDIT_TEXT_TEMPLATE.format(
            original_text=args.original_object,
            replacement_text=args.replacement_object,
        )
    else:
        raise ValueError(f"Unsupported attack type: {attack_type}")

    prompt = f"{base_prompt} {args.prompt}" if args.prompt else base_prompt

    # Aspect ratio passed through (default 16:9; must be one of allowed ratios per API)
    aspect_ratio = args.aspect_ratio

    try:
        result = request_reve_edit(
            image_path=source_path,
            edit_instruction=prompt,
            api_key=api_key,
            aspect_ratio=aspect_ratio,
            version=args.version,
        )
    except requests.RequestException as exc:
        raise SystemExit(f"Request failed: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Failed to parse response: {exc}") from exc

    edited_b64 = result.get("image")
    if not edited_b64:
        raise SystemExit("No edited image returned from REVE response.")

    edited_image = decode_base64_image(edited_b64)
    default_out = source_path.with_name(f"{source_path.stem}_reve_edit{source_path.suffix or '.jpg'}")
    output_path = Path(args.output) if args.output else Path("./tmp") / default_out.name

    save_image(edited_image, output_path)

    print(f"Request ID: {result.get('request_id')}")
    print(f"Credits used: {result.get('credits_used')}")
    print(f"Credits remaining: {result.get('credits_remaining')}")
    if result.get("content_violation"):
        print("Warning: Content policy violation detected.")
    print(f"Side-by-side saved to: {output_path}")


if __name__ == "__main__":
    main()
