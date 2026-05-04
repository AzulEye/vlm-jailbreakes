#!/usr/bin/env python3
"""
Download reference images from URLs or search engines.
Uses multiple sources: SerpAPI, DuckDuckGo, Wikimedia Commons, and more.
Includes OCR verification to ensure target text appears in images.

Usage:
    # Download from reference.json (uses image_url or search_query)
    python download_reference_images.py --references references.json --output images/

    # Search Google Images for specific query
    python download_reference_images.py --query "Murder on the Orient Express book cover" --output images/
"""

import argparse
import hashlib
import json
import os
import re
import sys
import time
import random
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
from urllib.parse import urlparse, quote_plus, urlencode
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from PIL import Image
from io import BytesIO

# API endpoints
SERPAPI_ENDPOINT = "https://serpapi.com/search.json"
BING_SEARCH_URL = "https://api.bing.microsoft.com/v7.0/images/search"
WIKIMEDIA_COMMONS_API = "https://commons.wikimedia.org/w/api.php"
ARCHIVE_ORG_API = "https://archive.org/advancedsearch.php"
OPEN_LIBRARY_COVERS = "https://covers.openlibrary.org"

# User agents for rotation
USER_AGENTS = [
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
]

# OCR engine (will be initialized on first use)
_ocr_engine = None
_ocr_available = None


def get_ocr_engine():
    """Get or initialize the OCR engine. Uses easyocr (no system dependencies)."""
    global _ocr_engine, _ocr_available
    
    if _ocr_available is False:
        return None
    
    if _ocr_engine is not None:
        return _ocr_engine
    
    # Try easyocr first (easier to install, no system dependencies)
    try:
        # Fix SSL certificate issues for model download
        import ssl
        ssl._create_default_https_context = ssl._create_unverified_context
        
        import easyocr
        _ocr_engine = easyocr.Reader(['en'], gpu=False, verbose=False)
        _ocr_available = True
        print("    [OCR] Using easyocr")
        return _ocr_engine
    except ImportError:
        pass
    except Exception as e:
        print(f"    [OCR] easyocr error: {e}")
    
    # Try pytesseract as fallback
    try:
        import pytesseract
        # Test if tesseract is installed
        pytesseract.get_tesseract_version()
        _ocr_engine = "pytesseract"
        _ocr_available = True
        print("    [OCR] Using pytesseract")
        return _ocr_engine
    except Exception:
        pass
    
    _ocr_available = False
    print("    [OCR] No OCR engine available. Install easyocr: pip install easyocr")
    return None


def extract_text_from_image(img: Image.Image) -> str:
    """Extract text from an image using OCR."""
    ocr = get_ocr_engine()
    
    if ocr is None:
        return ""
    
    try:
        # Convert PIL Image to numpy array for easyocr
        img_array = np.array(img.convert('RGB'))
        
        if ocr == "pytesseract":
            import pytesseract
            text = pytesseract.image_to_string(img)
        else:
            # easyocr
            results = ocr.readtext(img_array, detail=0)
            text = " ".join(results)
        
        return text.lower()
    except Exception as e:
        print(f"    [OCR] Error: {e}")
        return ""


def is_placeholder_image(img: Image.Image) -> bool:
    """
    Detect if an image is a placeholder (e.g., "Image not available").
    Uses multiple heuristics:
    1. Check for uniform/low variance colors (gray placeholder)
    2. Check image dimensions (many placeholders are square or very small)
    3. Use OCR to detect "not available", "no image", etc.
    """
    try:
        # Convert to RGB
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        img_array = np.array(img)
        
        # Check 1: Very low color variance (uniform gray image)
        std_dev = np.std(img_array)
        if std_dev < 20:  # Very uniform image
            print("    ✗ Placeholder detected: uniform color")
            return True
        
        # Check 2: Predominantly gray (common for placeholders)
        # Calculate if R, G, B channels are very similar (grayscale)
        r, g, b = img_array[:,:,0], img_array[:,:,1], img_array[:,:,2]
        gray_diff = np.mean(np.abs(r.astype(float) - g.astype(float)) + 
                           np.abs(g.astype(float) - b.astype(float)))
        mean_brightness = np.mean(img_array)
        
        # Gray images with low color difference and medium brightness are likely placeholders
        if gray_diff < 5 and 100 < mean_brightness < 200:
            # Additional check: small dimensions often indicate placeholder
            if img.size[0] < 300 or img.size[1] < 300:
                print("    ✗ Placeholder detected: gray and small")
                return True
        
        # Check 3: OCR for placeholder text
        text = extract_text_from_image(img)
        placeholder_phrases = [
            "not available", "no image", "image unavailable", 
            "no cover", "cover not", "preview not",
            "thumbnail", "placeholder", "coming soon"
        ]
        for phrase in placeholder_phrases:
            if phrase in text:
                print(f"    ✗ Placeholder detected: contains '{phrase}'")
                return True
        
        return False
        
    except Exception as e:
        print(f"    [Placeholder check] Error: {e}")
        return False


def verify_text_in_image(img: Image.Image, target_text: str, min_match_ratio: float = 0.5) -> bool:
    """
    Verify that the target text (or significant part of it) appears in the image.
    
    Args:
        img: PIL Image to check
        target_text: The text that should appear in the image
        min_match_ratio: Minimum fraction of words that must match (default 0.5)
    
    Returns:
        True if target text is found, False otherwise
    """
    if not target_text:
        return True  # No text to verify
    
    ocr_text = extract_text_from_image(img)
    
    if not ocr_text:
        # If OCR returns nothing, we can't verify - be lenient
        print("    [OCR] No text detected in image")
        return True  # Allow it through if we can't verify
    
    # Normalize both texts
    target_words = set(re.findall(r'\b\w+\b', target_text.lower()))
    ocr_words = set(re.findall(r'\b\w+\b', ocr_text.lower()))
    
    if not target_words:
        return True
    
    # Check how many target words appear in OCR text
    matches = target_words & ocr_words
    match_ratio = len(matches) / len(target_words)
    
    # For single words, require exact match
    if len(target_words) == 1:
        target_word = list(target_words)[0]
        if target_word in ocr_text:
            print(f"    [OCR] ✓ Found '{target_word}' in image")
            return True
        else:
            # Check for partial match (at least 4 chars)
            if len(target_word) >= 4:
                for word in ocr_words:
                    if target_word in word or word in target_word:
                        print(f"    [OCR] ✓ Partial match: '{word}' contains/in '{target_word}'")
                        return True
            print(f"    [OCR] ✗ '{target_word}' not found. OCR text: {ocr_text[:100]}...")
            return False
    
    # For multi-word targets, require minimum match ratio
    if match_ratio >= min_match_ratio:
        print(f"    [OCR] ✓ Found {len(matches)}/{len(target_words)} words: {matches}")
        return True
    else:
        print(f"    [OCR] ✗ Only {len(matches)}/{len(target_words)} words found. OCR: {ocr_text[:100]}...")
        return False


def get_random_headers() -> Dict[str, str]:
    """Get headers with a random user agent."""
    return {
        "User-Agent": random.choice(USER_AGENTS),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
    }


def sanitize_filename(name: str) -> str:
    """Convert a string to a safe filename."""
    name = re.sub(r'[^\w\s-]', '', name)
    name = re.sub(r'[-\s]+', '_', name)
    return name[:100].strip('_')


# Minimum image dimensions for quality filtering
MIN_IMAGE_WIDTH = 200
MIN_IMAGE_HEIGHT = 200


def download_image(
    url: str, 
    output_path: Path, 
    timeout: int = 30, 
    min_width: int = MIN_IMAGE_WIDTH, 
    min_height: int = MIN_IMAGE_HEIGHT,
    verify_text: Optional[str] = None,
    check_placeholder: bool = True,
) -> bool:
    """
    Download an image from URL and save to disk.
    
    Args:
        url: Image URL to download
        output_path: Path to save the image
        timeout: Request timeout in seconds
        min_width: Minimum image width
        min_height: Minimum image height
        verify_text: If provided, use OCR to verify this text appears in image
        check_placeholder: If True, check if image is a placeholder
    
    Returns:
        True if image was downloaded and passed all checks, False otherwise
    """
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        resp = requests.get(url, headers=headers, timeout=timeout, stream=True)
        resp.raise_for_status()
        
        # Verify it's actually an image
        content_type = resp.headers.get("content-type", "")
        if "image" not in content_type.lower() and not url.endswith(('.jpg', '.jpeg', '.png', '.webp')):
            print(f"    Not an image: {content_type}")
            return False
        
        # Load and validate with PIL
        img = Image.open(BytesIO(resp.content))
        
        # Check minimum dimensions for quality
        if img.size[0] < min_width or img.size[1] < min_height:
            print(f"    ✗ Image too small: {img.size[0]}x{img.size[1]} (min: {min_width}x{min_height})")
            return False
        
        # Check for placeholder images
        if check_placeholder and is_placeholder_image(img):
            return False
        
        # Verify target text appears in image (OCR)
        if verify_text and not verify_text_in_image(img, verify_text):
            return False
        
        # Convert to RGB if necessary
        if img.mode in ('RGBA', 'P'):
            img = img.convert('RGB')
        
        # Save at high quality
        output_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(output_path, quality=95)
        
        print(f"    ✓ Saved: {output_path} ({img.size[0]}x{img.size[1]})")
        return True
        
    except Exception as e:
        print(f"    ✗ Failed to download {url[:60]}...: {e}")
        return False


def search_serpapi(query: str, api_key: str, num_results: int = 5, min_width: int = 400, min_height: int = 400) -> List[Dict[str, str]]:
    """
    Search Google Images via SerpAPI.
    Returns list of {url, title, source, width, height} dicts.
    Filters for large images only.
    """
    params = {
        "q": query,
        "tbm": "isch",  # Image search
        "api_key": api_key,
        "num": num_results * 3,  # Request more to filter by size
        "safe": "off",
        "tbs": "isz:l",  # Large images only (isz:l = large, isz:m = medium)
    }
    
    try:
        resp = requests.get(SERPAPI_ENDPOINT, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        
        results = []
        for img in data.get("images_results", []):
            # Get image dimensions from SerpAPI response
            width = img.get("original_width", 0) or 0
            height = img.get("original_height", 0) or 0
            
            # Filter by minimum size if dimensions are available
            if width > 0 and height > 0:
                if width < min_width or height < min_height:
                    continue
            
            results.append({
                "url": img.get("original"),
                "thumbnail": img.get("thumbnail"),
                "title": img.get("title", ""),
                "source": img.get("source", ""),
                "width": width,
                "height": height,
            })
            
            if len(results) >= num_results:
                break
        
        # Sort by resolution (largest first)
        results.sort(key=lambda x: (x.get("width", 0) * x.get("height", 0)), reverse=True)
        return results[:num_results]
        
    except Exception as e:
        print(f"    SerpAPI error: {e}")
        return []


def search_bing_images(query: str, api_key: str, num_results: int = 5, min_width: int = 400, min_height: int = 400) -> List[Dict[str, str]]:
    """
    Search Bing Images API.
    Requires Bing Search API key from Azure.
    Filters for large, high-quality images.
    """
    headers = {"Ocp-Apim-Subscription-Key": api_key}
    params = {
        "q": query,
        "count": num_results * 3,  # Request more to filter
        "safeSearch": "Off",
        "size": "Large",  # Filter for large images
        "imageType": "Photo",  # Prefer photos over clipart
    }
    
    try:
        resp = requests.get(BING_SEARCH_URL, headers=headers, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        
        results = []
        for img in data.get("value", []):
            width = img.get("width", 0) or 0
            height = img.get("height", 0) or 0
            
            # Filter by minimum size
            if width > 0 and height > 0:
                if width < min_width or height < min_height:
                    continue
            
            results.append({
                "url": img.get("contentUrl"),
                "thumbnail": img.get("thumbnailUrl"),
                "title": img.get("name", ""),
                "source": urlparse(img.get("hostPageUrl", "")).netloc,
                "width": width,
                "height": height,
            })
            
            if len(results) >= num_results:
                break
        
        # Sort by resolution (largest first)
        results.sort(key=lambda x: (x.get("width", 0) * x.get("height", 0)), reverse=True)
        return results[:num_results]
        
    except Exception as e:
        print(f"    Bing API error: {e}")
        return []


def search_duckduckgo_images(query: str, num_results: int = 10) -> List[Dict[str, str]]:
    """
    Search DuckDuckGo Images - NO API KEY REQUIRED.
    Uses the instant answer API and HTML scraping as fallback.
    """
    results = []
    
    try:
        # Try using the duckduckgo_search library if available
        try:
            from duckduckgo_search import DDGS
            with DDGS() as ddgs:
                for r in ddgs.images(query, max_results=num_results * 2):
                    width = r.get("width", 0) or 0
                    height = r.get("height", 0) or 0
                    
                    if width < MIN_IMAGE_WIDTH or height < MIN_IMAGE_HEIGHT:
                        continue
                    
                    results.append({
                        "url": r.get("image"),
                        "thumbnail": r.get("thumbnail"),
                        "title": r.get("title", ""),
                        "source": r.get("source", "duckduckgo"),
                        "width": width,
                        "height": height,
                    })
                    
                    if len(results) >= num_results:
                        break
            
            results.sort(key=lambda x: (x.get("width", 0) * x.get("height", 0)), reverse=True)
            return results
            
        except ImportError:
            pass  # Library not installed, try alternative method
        
        # Alternative: Use DuckDuckGo's lite version with HTML parsing
        headers = get_random_headers()
        headers["Accept"] = "text/html"
        
        search_url = f"https://lite.duckduckgo.com/lite/?q={quote_plus(query + ' high resolution')}"
        resp = requests.get(search_url, headers=headers, timeout=15)
        
        # This is a basic fallback - won't get images directly but can find image URLs in results
        # For now, return empty and rely on other sources
        
    except Exception as e:
        print(f"    DuckDuckGo error: {e}")
    
    return results


def search_goodreads_books(query: str, num_results: int = 10) -> List[Dict[str, str]]:
    """
    Search Goodreads for book covers - NO API KEY REQUIRED.
    Uses the search page to find books with covers.
    """
    results = []
    
    try:
        headers = get_random_headers()
        search_url = f"https://www.goodreads.com/search?q={quote_plus(query)}&search_type=books"
        
        resp = requests.get(search_url, headers=headers, timeout=15)
        resp.raise_for_status()
        
        # Parse book cover URLs from HTML
        # Goodreads uses pattern like: src="https://images-na.ssl-images-amazon.com/images/..."
        # or: src="https://i.gr-assets.com/images/..."
        
        cover_pattern = r'<img[^>]+class="[^"]*bookCover[^"]*"[^>]+src="([^"]+)"'
        title_pattern = r'<a[^>]+class="[^"]*bookTitle[^"]*"[^>]*>([^<]+)</a>'
        
        covers = re.findall(cover_pattern, resp.text)
        titles = re.findall(title_pattern, resp.text)
        
        for i, (cover_url, title) in enumerate(zip(covers, titles)):
            if i >= num_results:
                break
            
            # Upgrade to larger image size
            # Goodreads uses _SX50_ or _SY75_ for thumbnails, can try _SX400_ or remove size
            cover_url = re.sub(r'_S[XY]\d+_', '_SX400_', cover_url)
            
            if query.lower() not in title.lower():
                continue
            
            results.append({
                "url": cover_url,
                "title": title.strip(),
                "source": "goodreads",
                "width": 0,
                "height": 0,
            })
        
    except Exception as e:
        print(f"    Goodreads error: {e}")
    
    return results


def search_wikimedia_commons(query: str, num_results: int = 10) -> List[Dict[str, str]]:
    """
    Search Wikimedia Commons for images - NO API KEY REQUIRED.
    Great for book covers, movie posters, album art, etc.
    """
    results = []
    
    params = {
        "action": "query",
        "format": "json",
        "generator": "search",
        "gsrnamespace": "6",  # File namespace
        "gsrsearch": f'"{query}"',
        "gsrlimit": num_results * 2,
        "prop": "imageinfo",
        "iiprop": "url|size|mime",
        "iiurlwidth": 1000,  # Request reasonably sized images
    }
    
    try:
        resp = requests.get(WIKIMEDIA_COMMONS_API, params=params, headers=get_random_headers(), timeout=15)
        resp.raise_for_status()
        data = resp.json()
        
        pages = data.get("query", {}).get("pages", {})
        
        for page_id, page in pages.items():
            if int(page_id) < 0:
                continue
                
            imageinfo = page.get("imageinfo", [{}])[0]
            
            # Check if it's an image
            mime = imageinfo.get("mime", "")
            if not mime.startswith("image/"):
                continue
            
            width = imageinfo.get("width", 0) or 0
            height = imageinfo.get("height", 0) or 0
            
            if width < MIN_IMAGE_WIDTH or height < MIN_IMAGE_HEIGHT:
                continue
            
            # Use thumburl for reasonable size, or url for original
            url = imageinfo.get("thumburl") or imageinfo.get("url")
            
            results.append({
                "url": url,
                "thumbnail": imageinfo.get("thumburl"),
                "title": page.get("title", "").replace("File:", ""),
                "source": "wikimedia",
                "width": width,
                "height": height,
            })
            
            if len(results) >= num_results:
                break
        
        results.sort(key=lambda x: (x.get("width", 0) * x.get("height", 0)), reverse=True)
        
    except Exception as e:
        print(f"    Wikimedia Commons error: {e}")
    
    return results


def search_archive_org(query: str, media_type: str = "texts", num_results: int = 10) -> List[Dict[str, str]]:
    """
    Search Archive.org (Internet Archive) - NO API KEY REQUIRED.
    Excellent for book covers, vintage media, posters.
    media_type: 'texts' for books, 'movies' for videos/posters, 'image' for images
    """
    results = []
    
    # Build query for books/movies with this text in title
    search_query = f'title:"{query}" AND mediatype:{media_type}'
    
    params = {
        "q": search_query,
        "fl[]": ["identifier", "title", "mediatype", "description"],
        "rows": num_results * 2,
        "page": 1,
        "output": "json",
    }
    
    try:
        resp = requests.get(ARCHIVE_ORG_API, params=params, headers=get_random_headers(), timeout=15)
        resp.raise_for_status()
        data = resp.json()
        
        for doc in data.get("response", {}).get("docs", []):
            identifier = doc.get("identifier")
            if not identifier:
                continue
            
            title = doc.get("title", "")
            if isinstance(title, list):
                title = title[0]
            
            # Archive.org provides cover images via a consistent URL pattern
            # For books: https://archive.org/services/img/{identifier}
            image_url = f"https://archive.org/services/img/{identifier}"
            
            results.append({
                "url": image_url,
                "title": title,
                "source": "archive.org",
                "identifier": identifier,
                "width": 0,  # Unknown, will be checked on download
                "height": 0,
            })
            
            if len(results) >= num_results:
                break
        
    except Exception as e:
        print(f"    Archive.org error: {e}")
    
    return results


def search_open_library_covers(query: str, num_results: int = 10) -> List[Dict[str, str]]:
    """
    Search Open Library for book covers by title - NO API KEY REQUIRED.
    Returns high-quality book cover images.
    """
    results = []
    
    params = {
        "title": query,
        "limit": num_results * 2,
        "fields": "key,title,cover_i,author_name,first_publish_year",
    }
    
    try:
        resp = requests.get("https://openlibrary.org/search.json", params=params, 
                           headers=get_random_headers(), timeout=15)
        resp.raise_for_status()
        data = resp.json()
        
        for doc in data.get("docs", []):
            title = doc.get("title", "")
            
            # Require query to appear in title
            if query.lower() not in title.lower():
                continue
            
            cover_id = doc.get("cover_i")
            if not cover_id:
                continue
            
            # Open Library cover sizes: S, M, L (L is largest, typically 300-500px)
            # Use -L suffix for largest
            image_url = f"https://covers.openlibrary.org/b/id/{cover_id}-L.jpg"
            
            authors = doc.get("author_name", [])
            author = authors[0] if authors else ""
            year = doc.get("first_publish_year", "")
            
            results.append({
                "url": image_url,
                "title": f"{title} by {author}" if author else title,
                "source": "openlibrary",
                "year": str(year) if year else "",
                "width": 0,  # Unknown
                "height": 0,
            })
            
            if len(results) >= num_results:
                break
        
    except Exception as e:
        print(f"    Open Library error: {e}")
    
    return results


def search_google_books_covers(query: str, num_results: int = 10) -> List[Dict[str, str]]:
    """
    Search Google Books for book covers - NO API KEY REQUIRED for basic search.
    Constructs higher-resolution image URLs.
    """
    results = []
    
    params = {
        "q": f"intitle:{query}",
        "maxResults": min(num_results * 2, 40),
        "printType": "books",
    }
    
    try:
        resp = requests.get("https://www.googleapis.com/books/v1/volumes", 
                           params=params, headers=get_random_headers(), timeout=15)
        resp.raise_for_status()
        data = resp.json()
        
        for item in data.get("items", []):
            vol = item.get("volumeInfo", {})
            title = vol.get("title", "")
            
            if query.lower() not in title.lower():
                continue
            
            # Try to construct a higher-resolution URL
            image_links = vol.get("imageLinks", {})
            image_url = (
                image_links.get("extraLarge") or
                image_links.get("large") or
                image_links.get("medium") or
                image_links.get("thumbnail")
            )
            
            if not image_url:
                continue
            
            # Modify URL for better quality
            image_url = re.sub(r'zoom=\d', 'zoom=0', image_url)
            image_url = image_url.replace("&edge=curl", "")
            
            authors = vol.get("authors", [])
            author = authors[0] if authors else ""
            
            results.append({
                "url": image_url,
                "title": f"{title} by {author}" if author else title,
                "source": "google_books",
                "year": vol.get("publishedDate", "")[:4] if vol.get("publishedDate") else "",
                "width": 0,
                "height": 0,
            })
            
            if len(results) >= num_results:
                break
        
    except Exception as e:
        print(f"    Google Books error: {e}")
    
    return results


def search_itunes(query: str, media: str = "movie", num_results: int = 10) -> List[Dict[str, str]]:
    """
    Search iTunes/Apple for movie/album artwork - NO API KEY REQUIRED.
    media: 'movie', 'music', 'ebook', 'tvShow'
    Returns high-quality artwork (up to 600x600 or larger).
    """
    results = []
    
    params = {
        "term": query,
        "media": media,
        "limit": num_results * 2,
    }
    
    try:
        resp = requests.get("https://itunes.apple.com/search", params=params, 
                           headers=get_random_headers(), timeout=15)
        resp.raise_for_status()
        data = resp.json()
        
        for item in data.get("results", []):
            # Get title based on media type
            if media == "movie":
                title = item.get("trackName", "")
            elif media == "music":
                title = item.get("collectionName", "") or item.get("trackName", "")
            elif media == "ebook":
                title = item.get("trackName", "")
            elif media == "tvShow":
                title = item.get("collectionName", "") or item.get("trackName", "")
            else:
                title = item.get("trackName", "")
            
            if query.lower() not in title.lower():
                continue
            
            # Get artwork URL and upscale to larger size
            artwork_url = item.get("artworkUrl100", "")
            if artwork_url:
                # Replace 100x100 with larger size (600x600 usually available)
                artwork_url = artwork_url.replace("100x100", "600x600")
            
            if not artwork_url:
                continue
            
            results.append({
                "url": artwork_url,
                "title": title,
                "source": "itunes",
                "year": item.get("releaseDate", "")[:4] if item.get("releaseDate") else "",
                "width": 600,
                "height": 600,
            })
            
            if len(results) >= num_results:
                break
        
    except Exception as e:
        print(f"    iTunes error: {e}")
    
    return results


def search_all_free_sources(query: str, category: str = "all", num_results: int = 5, fast_mode: bool = True) -> List[Dict[str, str]]:
    """
    Search all free image sources.
    category: 'book', 'movie', 'album', 'all'
    fast_mode: If True, only use the most reliable/fast sources (iTunes, Google Books)
    """
    all_results = []
    
    # Determine which sources to search based on category
    # Prioritize faster, more reliable sources first
    searches = []
    
    # Priority 1: Fast and reliable sources (usually no rate limiting)
    if category in ["movie", "tv_show", "all"]:
        searches.append(("iTunes Movies", lambda: search_itunes(query, "movie", num_results)))
        searches.append(("iTunes TV", lambda: search_itunes(query, "tvShow", num_results)))
    
    if category in ["album", "music", "all"]:
        searches.append(("iTunes Music", lambda: search_itunes(query, "music", num_results)))
    
    if category in ["book", "all"]:
        searches.append(("Google Books", lambda: search_google_books_covers(query, num_results)))
    
    # Priority 2: Slower but still reliable (if fast_mode is off)
    if not fast_mode:
        if category in ["book", "all"]:
            searches.append(("Open Library", lambda: search_open_library_covers(query, num_results)))
            searches.append(("Archive.org Books", lambda: search_archive_org(query, "texts", num_results)))
            searches.append(("Goodreads", lambda: search_goodreads_books(query, num_results)))
        
        if category in ["movie", "tv_show", "all"]:
            searches.append(("Archive.org Movies", lambda: search_archive_org(query, "movies", num_results)))
        
        # General image sources
        searches.append(("DuckDuckGo", lambda: search_duckduckgo_images(f"{query} high resolution", num_results)))
        searches.append(("Wikimedia", lambda: search_wikimedia_commons(query, num_results)))
    
    # Execute searches with early stopping if we have enough results
    for source_name, search_fn in searches:
        if len(all_results) >= num_results * 2:
            break  # We have enough
            
        try:
            results = search_fn()
            if results:
                print(f"      {source_name}: found {len(results)}")
                all_results.extend(results)
            time.sleep(0.3)  # More conservative rate limiting
        except Exception as e:
            print(f"      {source_name}: error - {str(e)[:50]}")
    
    # Deduplicate by URL
    seen_urls = set()
    unique_results = []
    for r in all_results:
        url = r.get("url", "")
        if url and url not in seen_urls:
            seen_urls.add(url)
            unique_results.append(r)
    
    # Sort by resolution
    unique_results.sort(key=lambda x: (x.get("width", 0) * x.get("height", 0)), reverse=True)
    
    return unique_results


def search_and_download(
    query: str,
    output_dir: Path,
    filename_prefix: str,
    serpapi_key: Optional[str] = None,
    bing_key: Optional[str] = None,
    num_images: int = 3,
    verify_text: Optional[str] = None,
) -> List[Path]:
    """
    Search for images and download the best results.
    
    Args:
        verify_text: If provided, use OCR to verify this text appears in the image
    """
    print(f"  Searching: {query}")
    
    downloaded = []
    
    # Try SerpAPI first (Google Images)
    if serpapi_key:
        results = search_serpapi(query, serpapi_key, num_images * 2)
        if results:
            print(f"    Found {len(results)} results via SerpAPI")
            for i, result in enumerate(results):
                if len(downloaded) >= num_images:
                    break
                url = result.get("url")
                if not url:
                    continue
                
                filename = f"{filename_prefix}_{i+1}.jpg"
                output_path = output_dir / filename
                
                if download_image(url, output_path, verify_text=verify_text, check_placeholder=True):
                    downloaded.append(output_path)
    
    # Fall back to Bing if needed
    if len(downloaded) < num_images and bing_key:
        results = search_bing_images(query, bing_key, num_images * 2)
        if results:
            print(f"    Found {len(results)} results via Bing")
            for i, result in enumerate(results):
                if len(downloaded) >= num_images:
                    break
                url = result.get("url")
                if not url:
                    continue
                
                filename = f"{filename_prefix}_bing_{i+1}.jpg"
                output_path = output_dir / filename
                
                if download_image(url, output_path, verify_text=verify_text, check_placeholder=True):
                    downloaded.append(output_path)
    
    # Fall back to FREE sources (no API key required)
    if len(downloaded) < num_images:
        # Determine category from query
        category = "all"
        query_lower = query.lower()
        if "book cover" in query_lower:
            category = "book"
        elif "movie poster" in query_lower or "film poster" in query_lower:
            category = "movie"
        elif "album cover" in query_lower:
            category = "album"
        elif "tv show" in query_lower or "tv series" in query_lower:
            category = "tv_show"
        
        results = search_all_free_sources(query, category, num_images * 2)
        if results:
            print(f"    Found {len(results)} results via free sources")
            for i, result in enumerate(results):
                if len(downloaded) >= num_images:
                    break
                url = result.get("url")
                if not url:
                    continue
                
                source = result.get("source", "free")
                filename = f"{filename_prefix}_{source}_{i+1}.jpg"
                output_path = output_dir / filename
                
                if download_image(url, output_path, verify_text=verify_text, check_placeholder=True):
                    downloaded.append(output_path)
    
    return downloaded


def search_and_download_free(
    query: str,
    output_dir: Path,
    filename_prefix: str,
    category: str = "all",
    num_images: int = 3,
) -> List[Path]:
    """
    Search for images using ONLY free sources (no API keys).
    """
    print(f"  Searching (free): {query}")
    
    downloaded = []
    results = search_all_free_sources(query, category, num_images * 3)
    
    if results:
        print(f"    Found {len(results)} results via free sources")
        for i, result in enumerate(results):
            if len(downloaded) >= num_images:
                break
            url = result.get("url")
            if not url:
                continue
            
            source = result.get("source", "free")
            filename = f"{filename_prefix}_{source}_{i+1}.jpg"
            output_path = output_dir / filename
            
            if download_image(url, output_path):
                downloaded.append(output_path)
    
    return downloaded


def process_references(
    references_path: Path,
    output_dir: Path,
    serpapi_key: Optional[str] = None,
    bing_key: Optional[str] = None,
    num_images: int = 3,
    prefer_direct: bool = True,
    prefer_search: bool = True,
    fast_mode: bool = True,  # Use fast_mode for image search
    verify_ocr: bool = True,  # Use OCR to verify target text in images
) -> Dict[str, List[str]]:
    """
    Process a references.json file and download images for each object.
    Saves progress incrementally so interrupted runs can be resumed.
    
    Strategy order:
    1. Direct URLs from TMDB/Open Library (already have from search, high quality)
    2. iTunes search (fast, no rate limits)
    3. Google Books search (fast)
    4. Other sources if fast_mode=False
    
    Args:
        verify_ocr: If True, use OCR to verify target text appears in downloaded images
    """
    with open(references_path) as f:
        references = json.load(f)
    
    # Load existing manifest if it exists (for resume capability)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        try:
            with open(manifest_path) as f:
                results = json.load(f)
            print(f"Loaded {len(results)} existing results from manifest")
        except (json.JSONDecodeError, IOError):
            results = {}
    else:
        results = {}
    
    # Count how many we can skip (already have enough images)
    skip_count = sum(1 for obj in references if obj in results and len(results.get(obj, [])) >= num_images)
    if skip_count > 0:
        print(f"Skipping {skip_count} objects that already have {num_images}+ images")
    
    for obj, data in references.items():
        # Skip if already have enough images
        existing_images = results.get(obj, [])
        # Verify existing images still exist on disk
        existing_images = [p for p in existing_images if Path(p).exists()]
        if len(existing_images) >= num_images:
            results[obj] = existing_images  # Update with verified paths
            continue
        
        print(f"\n=== {obj} === (have {len(existing_images)}, need {num_images})")
        obj_dir = output_dir / sanitize_filename(obj)
        downloaded = list(existing_images)  # Start with existing
        refs = data.get("references", [])
        
        # Target text to verify with OCR
        target_text = obj if verify_ocr else None
        
        # Strategy 1: Direct URLs from TMDB/Open Library (already obtained in search)
        # These are high quality and don't require additional API calls
        if prefer_direct:
            print("  Trying direct URLs from references...")
            for ref in refs[:10]:
                if len(downloaded) >= num_images:
                    break
                
                image_url = ref.get("image_url")
                source = ref.get("source", "")
                
                # Skip Google Books (often low quality or placeholders) and sources without URLs
                if source == "google_books" or not image_url:
                    continue
                
                title = ref.get("title", "unknown")
                cat = ref.get("category", "ref")
                filename = f"{sanitize_filename(title)}_{cat}.jpg"
                output_path = obj_dir / filename
                
                # Skip if file already exists
                if output_path.exists() and str(output_path) in downloaded:
                    continue
                
                print(f"    [{source}] {title[:40]}...")
                if download_image(image_url, output_path, verify_text=target_text, check_placeholder=True):
                    downloaded.append(str(output_path))
        
        # Strategy 2: Fast free sources (iTunes, Google Books API)
        if len(downloaded) < num_images and prefer_search:
            best_ref = data.get("best_reference")
            category = best_ref.get("category", "all") if best_ref else "all"
            
            # Search using best reference title
            if best_ref and best_ref.get("title"):
                title = best_ref.get("title", obj)
                print(f"  Fast search for: {title[:40]}...")
                
                search_results = search_all_free_sources(title, category, num_images * 2, fast_mode=True)
                
                for i, result in enumerate(search_results):
                    if len(downloaded) >= num_images:
                        break
                    url = result.get("url")
                    if not url:
                        continue
                    
                    src = result.get("source", "free")
                    filename = f"{sanitize_filename(title)}_{src}_{i+1}.jpg"
                    output_path = obj_dir / filename
                    
                    if download_image(url, output_path, verify_text=target_text, check_placeholder=True):
                        downloaded.append(str(output_path))
        
        # Strategy 3: Search for the object name directly
        if len(downloaded) < num_images:
            print(f"  Searching for: {obj}...")
            
            search_results = search_all_free_sources(obj, "all", num_images * 2, fast_mode=True)
            
            for i, result in enumerate(search_results):
                if len(downloaded) >= num_images:
                    break
                url = result.get("url")
                if not url:
                    continue
                
                src = result.get("source", "free")
                filename = f"{sanitize_filename(obj)}_{src}_{i+1}.jpg"
                output_path = obj_dir / filename
                
                if download_image(url, output_path, verify_text=target_text, check_placeholder=True):
                    downloaded.append(str(output_path))
        
        # Strategy 4: Use paid API sources if available (SerpAPI, Bing)
        if len(downloaded) < num_images and (serpapi_key or bing_key):
            best_ref = data.get("best_reference")
            if best_ref and best_ref.get("search_query"):
                print(f"  Paid API search: {best_ref.get('title', obj)}")
                more = search_and_download(
                    best_ref["search_query"],
                    obj_dir,
                    sanitize_filename(best_ref.get("title", obj)),
                    serpapi_key,
                    bing_key,
                    num_images - len(downloaded),
                    verify_text=target_text,
                )
                downloaded.extend([str(p) for p in more])
        
        results[obj] = downloaded
        print(f"  Total downloaded: {len(downloaded)}")
        
        # Save manifest incrementally after each object
        try:
            with open(manifest_path, "w") as f:
                json.dump(results, f, indent=2)
        except IOError as e:
            print(f"Warning: Failed to save manifest: {e}")
        
        time.sleep(0.5)  # Rate limiting
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Download reference images for text replacement attack"
    )
    parser.add_argument(
        "--references",
        type=str,
        help="Path to references.json from search_references.py"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./base",
        help="Output directory for downloaded images"
    )
    parser.add_argument(
        "--query",
        type=str,
        help="Single search query (instead of references file)"
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=3,
        help="Number of images to download per object"
    )
    parser.add_argument(
        "--serpapi-key",
        type=str,
        default=os.environ.get("SERPAPI_KEY"),
        help="SerpAPI key for Google Image Search"
    )
    parser.add_argument(
        "--bing-key",
        type=str,
        default=os.environ.get("BING_SEARCH_KEY"),
        help="Bing Search API key"
    )
    parser.add_argument(
        "--no-direct",
        action="store_true",
        help="Skip direct URL downloads, only use search"
    )
    
    args = parser.parse_args()
    output_dir = Path(args.output)
    
    if args.query:
        # Single query mode
        downloaded = search_and_download(
            args.query,
            output_dir,
            "query_result",
            args.serpapi_key,
            args.bing_key,
            args.num_images,
        )
        print(f"\nDownloaded {len(downloaded)} images to {output_dir}")
        
    elif args.references:
        # Process references file
        results = process_references(
            Path(args.references),
            output_dir,
            args.serpapi_key,
            args.bing_key,
            args.num_images,
            prefer_direct=not args.no_direct,
        )
        
        # Save manifest
        manifest_path = output_dir / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(results, f, indent=2)
        
        total = sum(len(v) for v in results.values())
        print(f"\n=== Summary ===")
        print(f"Total images downloaded: {total}")
        print(f"Manifest saved to: {manifest_path}")
        
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
