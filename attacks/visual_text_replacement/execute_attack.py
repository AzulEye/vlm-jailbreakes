#!/usr/bin/env python3
"""
Execute a single attack run for one object using pre-generated images.

This module handles the visual text replacement attack execution:
- Load attacked images from data directory
- Build prompt using harmbench questions
- Send images and prompt to VLM for evaluation

Usage:
    python execute_attack.py --object murder --replacement banana --provider ollama
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Import shared inference modules
from attacks.common.ollama_inference import analyze_multiple_images_ollama
from attacks.common.openrouter_inference import analyze_multiple_images_openrouter
from attacks.visual_text_replacement.attack_prompts import get_attack_prompt

DEFAULT_MODELS = {
    "ollama": "qwen3-vl:235b-instruct",
    "openrouter": "qwen/qwen2.5-vl-32b-instruct",
}


def _collect_images(base_dir: Path) -> List[Path]:
    """Collect all image files from a directory."""
    images: List[Path] = []
    for pattern in ("*.png", "*.jpg", "*.jpeg"):
        images.extend(base_dir.glob(pattern))
    return sorted(images)


def _sanitize_name(name: str) -> str:
    """Convert a string to a safe directory name."""
    import re
    name = re.sub(r'[^\w\s-]', '', name)
    name = re.sub(r'[-\s]+', '_', name)
    return name[:100].strip('_')


def _attack_image_dir(
    data_root: Path,
    object_name: str,
    replacement: str,
) -> Path:
    """Get the directory containing attacked images for an object/replacement pair."""
    return data_root / "attacks" / _sanitize_name(object_name) / "text_replacement" / replacement


def run_attack_for_object(
    object_name: str,
    replacement: str,
    provider: str,
    model: Optional[str] = None,
    data_root: Path = Path("./data/visual_text_replacement"),
    api_key: Optional[str] = None,
    quiet: bool = False,
) -> List[dict]:
    """
    Run the text replacement attack for a single object.
    
    Args:
        object_name: Target object (e.g., "murder", "bomb")
        replacement: Replacement text used in attack (e.g., "banana")
        provider: VLM provider ("ollama" or "openrouter")
        model: Model name (uses default if not specified)
        data_root: Root directory containing attack images
        api_key: API key for the provider
        quiet: Suppress verbose output
    
    Returns:
        List of response dictionaries
    """
    attack_dir = _attack_image_dir(data_root, object_name, replacement)

    if not attack_dir.exists():
        raise FileNotFoundError(f"Attack directory not found: {attack_dir}")

    images = _collect_images(attack_dir)
    if not images:
        raise FileNotFoundError(f"No images found under {attack_dir}")

    prompt = get_attack_prompt(object_name, replacement)
    model_name = model or DEFAULT_MODELS.get(provider)
    if not model_name:
        raise ValueError(f"No model provided and no default available for provider '{provider}'")

    image_paths = [str(p) for p in images]

    if not quiet:
        print(f"[START] attack=text_replacement object={object_name} provider={provider} model={model_name}")
        print(f"[IMAGES] {len(images)} image(s) from {attack_dir}")
        print("[RUN] Sent all images as a single multi-image request")

    if provider == "ollama":
        resolved_key = api_key or os.environ.get("OLLAMA_API_KEY")
        if not resolved_key:
            raise ValueError("OLLAMA_API_KEY not provided")

        response = analyze_multiple_images_ollama(
            image_paths=image_paths,
            prompt=prompt,
            model=model_name,
            api_key=resolved_key,
            verbose=not quiet,
        )
    elif provider == "openrouter":
        resolved_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not resolved_key:
            raise ValueError("OPENROUTER_API_KEY not provided")

        response = analyze_multiple_images_openrouter(
            image_paths=image_paths,
            prompt=prompt,
            model=model_name,
            api_key=resolved_key,
            verbose=not quiet,
        )
    else:
        raise ValueError(f"Unsupported provider: {provider}")

    if not quiet:
        print("\n[COMPLETE] Received response for batch")

    return [{"images": image_paths, "response": response}]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Execute a visual text replacement attack for one object.")
    parser.add_argument("--object", required=True, help="Target object name (e.g., 'murder', 'bomb').")
    parser.add_argument(
        "--replacement",
        required=True,
        help="Replacement text used in the attack (e.g., 'banana').",
    )
    parser.add_argument(
        "--provider",
        choices=["ollama", "openrouter"],
        default="ollama",
        help="Provider to run the attack with.",
    )
    parser.add_argument("--model", help="Model to use. Falls back to provider default when omitted.")
    parser.add_argument(
        "--data-root",
        default=str(REPO_ROOT / "data" / "visual_text_replacement"),
        help="Root directory containing attacked images.",
    )
    parser.add_argument("--api-key", help="API key for the chosen provider (overrides env).")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose inference output.")
    return parser.parse_args()


def main():
    args = parse_args()
    data_root = Path(args.data_root).resolve()

    try:
        results = run_attack_for_object(
            object_name=args.object,
            replacement=args.replacement,
            provider=args.provider,
            model=args.model,
            data_root=data_root,
            api_key=args.api_key,
            quiet=args.quiet,
        )
        
        # Print response preview
        if results and results[0].get("response"):
            resp = results[0]["response"]
            print("\n[RESPONSE PREVIEW]")
            print(resp[:500] + "..." if len(resp) > 500 else resp)
            
    except Exception as exc:
        print(f"Error: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
