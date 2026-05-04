#!/usr/bin/env python3
"""
Generate images from a text prompt using the REVE create API.
Builds prompts from a template that implies the target object via context.
"""

import argparse
import base64
import json
import os
import sys
import time
from io import BytesIO
from pathlib import Path
from typing import Optional, Tuple

import requests
from PIL import Image

from .prompts import CREATE_TEMPLATE, CREATE_TEXT_TEMPLATE


def decode_base64_image(data: str) -> Image.Image:
    """Decode a base64 image string (data URLs supported)."""
    if "," in data and data.strip().startswith("data:"):
        data = data.split(",", 1)[1]
    return Image.open(BytesIO(base64.b64decode(data))).convert("RGB")


def request_reve_create(
    prompt: str,
    api_key: str,
    aspect_ratio: str = "3:2",
    version: str = "reve-create@20250915",
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
                print(f"Request timed out/failed (attempt {attempt}/{max_retries}). Retrying in {wait}s...", flush=True)
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


def build_prompt(object_name: str, extra: Optional[str], attack_type: Optional[str] = None) -> str:
    """Construct the generation prompt using the create template constant."""
    if attack_type == "text_replacement":
        base_prompt = CREATE_TEXT_TEMPLATE.format(target_text=object_name)
    else:
        base_prompt = CREATE_TEMPLATE.format(object=object_name)
    if extra:
        return f"{base_prompt} {extra}"
    return base_prompt


def save_image(image_b64: str, output_path: Path) -> Tuple[Path, Image.Image]:
    """Decode base64 image and save to disk."""
    image = decode_base64_image(image_b64)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    return output_path, image


def main():
    parser = argparse.ArgumentParser(
        description="Generate an image with REVE create API using a template-based prompt."
    )
    parser.add_argument(
        "object",
        type=str,
        nargs="?",
        default="bomb",
        help="Object to imply via context (default: bomb).",
    )
    parser.add_argument(
        "extra",
        type=str,
        nargs="?",
        help="Optional extra instructions to append to the template prompt.",
    )
    parser.add_argument(
        "--attack-type",
        type=str,
        default=None,
        help="Attack type for determining prompt template (e.g., 'text_replacement').",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output path for image (default: ./tmp/generated_<object>.png).",
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
        default="reve-create@20250915",
        help="REVE model version (default: reve-create@20250915).",
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
    parser.add_argument(
        "--retry-backoff",
        type=int,
        default=2,
        help="Exponential backoff base in seconds (default: 2).",
    )

    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("REVE_API_KEY")
    if not api_key:
        raise ValueError("REVE_API_KEY not set. Provide --api-key or export REVE_API_KEY.")

    prompt = build_prompt(args.object, args.extra, args.attack_type)

    print(
        f"Requesting REVE create API with aspect_ratio={args.aspect_ratio}, version={args.version}...",
        flush=True,
    )

    result = request_reve_create(
        prompt=prompt,
        api_key=api_key,
        aspect_ratio=args.aspect_ratio,
        version=args.version,
        timeout=args.timeout,
        max_retries=args.max_retries,
        backoff=args.retry_backoff,
    )

    image_b64 = result.get("image")
    if not image_b64:
        raise SystemExit("No image returned from REVE response.")

    output_path = Path(args.output) if args.output else Path(f"./tmp/generated_{args.object}.png")
    save_image(image_b64, output_path)

    print(f"Prompt used: {prompt}")
    print(f"Request ID: {result.get('request_id')}")
    print(f"Credits used: {result.get('credits_used')}")
    print(f"Credits remaining: {result.get('credits_remaining')}")
    if result.get("content_violation"):
        print("Warning: Content policy violation detected.")
    print(f"Image saved to: {output_path}")


if __name__ == "__main__":
    main()
