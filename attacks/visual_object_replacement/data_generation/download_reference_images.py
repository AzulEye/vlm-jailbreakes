#!/usr/bin/env python3
"""
download_reference_images.py

Robust downloader that handles "Image Not Available" errors by verifying
content headers and falling back to Image Search if direct URLs fail.

Usage:
    python download_reference_images.py --references references.json --output ./images
"""

import argparse
import json
import os
import requests
from pathlib import Path
from urllib.parse import urlparse
from io import BytesIO
from PIL import Image

# --- CONFIGURATION ---
# Get these from https://serpapi.com/ or Azure Bing Search
SERPAPI_URL = "https://serpapi.com/search.json"
BING_URL = "https://api.bing.microsoft.com/v7.0/images/search"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Referer": "https://www.google.com/"
}

def sanitize_filename(text):
    return "".join([c for c in text if c.isalpha() or c.isdigit() or c==' ']).rstrip().replace(" ", "_")[:50]

def is_valid_image(content: bytes) -> bool:
    """
    Checks if bytes are a valid image and NOT a tiny placeholder.
    Many APIs return a 1x1 pixel or a < 2KB 'broken link' icon.
    """
    if len(content) < 4000: # Less than 4KB is suspicious for a reference image
        return False
    try:
        img = Image.open(BytesIO(content))
        img.verify() # verify it's an image
        if img.width < 100 or img.height < 100:
            return False
        return True
    except Exception:
        return False

def download_direct(url: str, filepath: Path) -> bool:
    """Attempts to download directly from the URL found in metadata."""
    if not url: return False
    print(f"  Attempting direct download: {url[:60]}...")
    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
        resp.raise_for_status()
        
        if is_valid_image(resp.content):
            with open(filepath, "wb") as f:
                f.write(resp.content)
            print("    -> Success (Direct)")
            return True
        else:
            print("    -> Failed: File too small or invalid (likely placeholder).")
            return False
    except Exception as e:
        print(f"    -> Direct download failed: {e}")
        return False

def download_via_serpapi(query: str, filepath: Path, api_key: str) -> bool:
    """Fallback: Search Google Images via SerpAPI."""
    print(f"  Fallback: Searching SerpAPI for '{query}'...")
    params = {
        "q": query, "tbm": "isch", "api_key": api_key,
        "isz": "l", "safe": "off" # Large images
    }
    try:
        resp = requests.get(SERPAPI_URL, params=params, timeout=20)
        data = resp.json()
        results = data.get("images_results", [])
        
        # Try top 3 results
        for res in results[:3]:
            img_url = res.get("original")
            if download_direct(img_url, filepath):
                return True
        return False
    except Exception as e:
        print(f"    -> SerpAPI Error: {e}")
        return False

def download_via_bing(query: str, filepath: Path, api_key: str) -> bool:
    """Fallback: Search Bing Images."""
    print(f"  Fallback: Searching Bing for '{query}'...")
    headers = {"Ocp-Apim-Subscription-Key": api_key}
    params = {"q": query, "count": 5, "size": "Large"}
    try:
        resp = requests.get(BING_URL, headers=headers, params=params, timeout=20)
        data = resp.json()
        
        for val in data.get("value", []):
            img_url = val.get("contentUrl")
            if download_direct(img_url, filepath):
                return True
        return False
    except Exception as e:
        print(f"    -> Bing Error: {e}")
        return False

def process_references(ref_file, output_dir, serp_key, bing_key):
    with open(ref_file) as f:
        data = json.load(f)
    
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    for target, info in data.items():
        best = info.get("best_match")
        if not best:
            print(f"Skipping {target} (no match found)")
            continue

        safe_name = sanitize_filename(target)
        filename = out_path / f"{safe_name}.jpg"

        print(f"\nProcessing: {target} -> {best['title']}")
        
        # 1. Try Direct URL (from TMDB/Wiki/GoogleBooks)
        if best.get("image_url"):
            success = download_direct(best["image_url"], filename)
            if success: continue
        
        # 2. Fallback to Search Engines if direct failed or didn't exist
        search_query = best.get("search_query") or f"{best['title']} {best['category']}"
        
        success = False
        if serp_key:
            success = download_via_serpapi(search_query, filename, serp_key)
        
        if not success and bing_key:
            success = download_via_bing(search_query, filename, bing_key)
            
        if not success:
            print("  [X] Failed to acquire image from all sources.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--references", required=True, help="references.json file")
    parser.add_argument("--output", default="./images")
    parser.add_argument("--serp-key", default=os.getenv("SERPAPI_KEY"))
    parser.add_argument("--bing-key", default=os.getenv("BING_SEARCH_KEY"))
    args = parser.parse_args()

    process_references(args.references, args.output, args.serp_key, args.bing_key)

if __name__ == "__main__":
    main()