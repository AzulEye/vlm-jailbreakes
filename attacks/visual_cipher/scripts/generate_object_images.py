#!/usr/bin/env python3
"""Generate a set of square images for everyday objects using OpenRouter."""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import time
import urllib.request
import sys
import subprocess
from pathlib import Path
from typing import List


PROMPTS: List[str] = [
    "A smiling red apple with a tiny heart sticker",
    "A sunny bunch of ripe bananas tied with a ribbon",
    "A bright orange with a happy face drawn in marker",
    "A bowl of fresh strawberries with a small smiley tag",
    "A cluster of grapes with a cheerful pastel label",
    "A juicy pear with a cute polka dot bow",
    "A peach with a soft pink glow and a smiley note",
    "A pineapple wearing a tiny party hat",
    "A slice of watermelon with a friendly face",
    "A lemon with a big cheerful grin sticker",
    "A lime with a tiny green smile button",
    "A cherry pair with a smiling ribbon",
    "A handful of blueberries in a happy paper cup",
    "A raspberry with a tiny heart tag",
    "A mango with a cheerful sun sticker",
    "A kiwi cut in half like a smiling face",
    "A tomato with a happy kitchen doodle",
    "A carrot with a friendly smile sticker",
    "A broccoli floret shaped like a little tree with a smile",
    "A cucumber with a cheerful striped band",
    "A bell pepper with a bright smiley magnet",
    "A friendly eggplant with a tiny bow tie",
    "A potato with a smiling face drawn on it",
    "An onion with a happy label and a bow",
    "A garlic bulb with a cute grin sticker",
    "A mushroom with a cheerful polka dot cap",
    "A corn cob with a happy picnic tag",
    "A warm bread loaf with a smiling bakery sticker",
    "A crisp baguette wrapped in a cheerful paper sleeve",
    "A golden croissant with a tiny heart flag",
    "A colorful donut with sprinkles and a smiley face",
    "A cupcake with rainbow frosting and a smile topper",
    "A cookie with a friendly smile and chocolate chips",
    "A chocolate bar with a cheerful wrapper",
    "A coffee mug with a sunny face and steam hearts",
    "A teacup with a floral pattern and a smiley tag",
    "A water bottle with a bright cheerful label",
    "A soda can with a happy retro design",
    "A milk carton with a smiling cow illustration",
    "A cereal bowl with a happy spoon resting beside it",
    "A plate with a cute smiley doodle on the rim",
    "A fork with a tiny bow and a cheerful sticker",
    "A spoon with a smiley face on the handle",
    "A knife with a colorful happy handle",
    "A frying pan with a sunny yellow handle",
    "A saucepan with a cheerful lid knob",
    "A kettle with a smiley magnet",
    "A toaster with a happy sticker and toast hearts",
    "A blender with a bright smiley label",
    "A cutting board with a cheerful gingham pattern",
    "A wooden spoon with a cute painted face",
    "A rolling pin with a pastel ribbon",
    "A kitchen timer shaped like a smiling tomato",
    "A salt shaker with a friendly face",
    "A pepper shaker with a cheerful label",
    "A candle with a warm glow and a heart tag",
    "A light bulb glowing softly with a smile sticker",
    "A flashlight with a bright happy sticker",
    "A battery with a cheerful green label",
    "An alarm clock with a smiling face",
    "A wristwatch with a sunny yellow band",
    "Eyeglasses with a cute heart case",
    "Sunglasses with a bright rainbow frame",
    "A wallet with a happy sticker",
    "A key with a tiny smiling keychain",
    "A keychain shaped like a happy star",
    "A backpack with a cheerful patch",
    "A handbag with a bright floral charm",
    "A cheerful bright yellow umbrella with a cute smiling face",
    "A cozy knit hat with a smiley tag",
    "A soft scarf with pastel stripes and a happy pin",
    "A pair of gloves with a tiny heart patch",
    "A cozy sock with a smiling face pattern",
    "A comfy shoe with a cheerful lace charm",
    "A sneaker with colorful happy accents",
    "A book with a friendly cover illustration",
    "A notebook with a smiling sun on the cover",
    "A pencil with a happy eraser face",
    "A pen with a cheerful clip",
    "An eraser shaped like a smiling cloud",
    "A ruler with a rainbow stripe and a smiley sticker",
    "A stapler with a bright cheerful sticker",
    "A paperclip shaped like a heart",
    "A tape dispenser with a sunny pattern",
    "A pair of scissors with colorful handles and a smiley tag",
    "A calculator with a bright cheerful sticker",
    "A smartphone with a happy wallpaper",
    "A remote control with a cute smiley decal",
    "Headphones with a colorful joyful sticker",
    "A camera with a cheerful strap and a smile tag",
    "A plant pot with a smiling face",
    "A flower vase with fresh daisies and a smiley note",
    "A picture frame with a heart border",
    "A pillow with a happy face embroidery",
    "A blanket with warm pastel stripes",
    "A chair with a cheerful cushion",
    "A small table with a sunny yellow top",
    "A lamp with a warm smiling shade",
    "A bicycle with a cheerful bell and a flower basket",
    "A soccer ball with a friendly smiley logo",
]


def slugify(text: str) -> str:
    base = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return base or "object"


def build_prompt(obj: str) -> str:
    text = obj.strip()
    if not text:
        return "a photo of a cheerful object on a plain background, centered, no text"
    lower = text.lower()
    if lower.startswith(("a ", "an ", "the ")):
        base = text
    else:
        base = f"a photo of a {text}"
    return f"{base} on a plain background, centered, no text"


def load_prompts(path: Path | None) -> List[str]:
    if not path:
        return PROMPTS.copy()
    prompts = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            prompts.append(line)
    return prompts


def decode_image_url(url: str) -> bytes:
    if url.startswith("data:image/"):
        header, b64 = url.split(",", 1)
        return base64.b64decode(b64)
    with urllib.request.urlopen(url, timeout=60) as resp:
        return resp.read()


def extract_images(payload: dict) -> List[bytes]:
    images: List[bytes] = []
    for choice in payload.get("choices", []):
        msg = choice.get("message", {}) or {}
        msg_images = msg.get("images")
        if isinstance(msg_images, list):
            for item in msg_images:
                if not isinstance(item, dict):
                    continue
                if item.get("type") == "image_url":
                    url = (item.get("image_url") or {}).get("url") or item.get("url")
                    if url:
                        images.append(decode_image_url(url))
                else:
                    data = item.get("data") or item.get("b64_json")
                    if data:
                        images.append(base64.b64decode(data))
        content = msg.get("content")
        if isinstance(content, list):
            for part in content:
                if not isinstance(part, dict):
                    continue
                if part.get("type") == "image_url":
                    url = (part.get("image_url") or {}).get("url") or part.get("url")
                    if url:
                        images.append(decode_image_url(url))
                elif part.get("type") in ("image", "output_image"):
                    data = part.get("image") or part.get("data") or part.get("b64_json")
                    if data:
                        images.append(base64.b64decode(data))
        elif isinstance(content, str):
            match = re.search(r"data:image/[^;]+;base64,([A-Za-z0-9+/=]+)", content)
            if match:
                images.append(base64.b64decode(match.group(1)))
        for key in ("image", "data", "b64_json"):
            if isinstance(msg.get(key), str):
                images.append(base64.b64decode(msg[key]))
    return images


def ensure_deps() -> None:
    try:
        import openai  # noqa: F401
        import PIL  # noqa: F401
    except ImportError:
        print("Installing dependencies...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "openai", "pillow"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate object image tiles with OpenRouter.")
    parser.add_argument("--output-dir", type=Path, default=Path("assets/object_tiles"))
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--gen-size", type=int, default=1024, help="Requested generation size (e.g., 1024).")
    parser.add_argument("--output-size", type=int, default=128, help="Final saved size (e.g., 128).")
    parser.add_argument("--model", type=str, default="google/gemini-2.5-flash-image")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--sleep", type=float, default=0.0, help="Seconds to sleep between requests.")
    parser.add_argument("--timeout", type=float, default=180.0, help="Request timeout in seconds.")
    parser.add_argument("--retries", type=int, default=3, help="Retries per prompt on failure.")
    parser.add_argument("--response-format", type=str, default="image", help="Response format type.")
    parser.add_argument("--image-config", type=str, default="", help="Optional JSON dict for image_config.")
    parser.add_argument("--prompts-file", type=Path, default=None, help="Optional text file with one object per line.")
    parser.add_argument("--api-key", type=str, default="", help="OpenRouter API key (or set OPENROUTER_API_KEY).")
    parser.add_argument("--app-name", type=str, default="", help="Optional OpenRouter app name.")
    parser.add_argument("--app-url", type=str, default="", help="Optional OpenRouter app URL.")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate images even if files already exist.")
    args = parser.parse_args()

    ensure_deps()

    from openai import OpenAI
    from PIL import Image

    prompts = load_prompts(args.prompts_file)
    if args.count > len(prompts):
        raise SystemExit(f"Requested {args.count} images, but only {len(prompts)} prompts available.")

    api_key = args.api_key or os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("Missing OPENROUTER_API_KEY (or pass --api-key).")

    headers = {}
    if args.app_name:
        headers["X-Title"] = args.app_name
    if args.app_url:
        headers["HTTP-Referer"] = args.app_url
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        default_headers=headers or None,
        timeout=args.timeout,
    )

    if args.image_config:
        image_config = json.loads(args.image_config)
    else:
        image_config = {"size": f"{args.gen_size}x{args.gen_size}"}

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "model": args.model,
        "gen_size": args.gen_size,
        "output_size": args.output_size,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "image_config": image_config,
        "seed_base": args.seed,
        "images": [],
    }

    for idx, obj in enumerate(prompts[: args.count]):
        prompt = build_prompt(obj)
        filename = f"{idx:03d}_{slugify(obj)}.png"
        out_path = args.output_dir / filename
        if out_path.exists() and not args.overwrite:
            manifest["images"].append(
                {"index": idx, "object": obj, "prompt": prompt, "file": filename, "seed": args.seed + idx}
            )
            continue

        print(f"[{idx+1}/{args.count}] Requesting image for: {obj}")
        last_err = None
        images = []
        for attempt in range(args.retries + 1):
            try:
                response = client.chat.completions.create(
                    model=args.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    seed=args.seed + idx,
                    response_format={"type": args.response_format} if args.response_format else None,
                    extra_body={"image_config": image_config} if image_config else None,
                )
                payload = response.model_dump()
                images = extract_images(payload)
                if images:
                    break
                last_err = RuntimeError("No image returned in response.")
            except Exception as exc:  # noqa: BLE001
                last_err = exc
            wait_s = min(10.0, 1.0 + attempt * 2)
            print(f"  retry {attempt + 1}/{args.retries} after error: {last_err}")
            time.sleep(wait_s)
        if not images:
            print(f"Warning: failed to get image for {obj}: {last_err}")
            continue

        image = Image.open(io.BytesIO(images[0])).convert("RGB")
        if args.output_size and image.size != (args.output_size, args.output_size):
            image = image.resize((args.output_size, args.output_size), Image.LANCZOS)
        image.save(out_path)
        manifest["images"].append(
            {"index": idx, "object": obj, "prompt": prompt, "file": filename, "seed": args.seed + idx}
        )
        print(f"Wrote {out_path}")
        if args.sleep > 0:
            time.sleep(args.sleep)

    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Manifest written to {args.output_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
