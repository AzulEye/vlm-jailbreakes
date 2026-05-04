#!/usr/bin/env python3
"""
search_references.py

Searches for canonical cultural artifacts (Books, Movies, Wikipedia Entities) 
that contain a target string verbatim.

Usage:
    python search_references.py --query "The Great Gatsby"
    python search_references.py --config targets.json --output references.json
"""

import argparse
import json
import os
import time
import requests
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, asdict

# --- CONFIGURATION ---
GOOGLE_BOOKS_API = "https://www.googleapis.com/books/v1/volumes"
TMDB_SEARCH_MOVIE = "https://api.themoviedb.org/3/search/movie"
WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"

# Standard headers to prevent being blocked as a bot
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json, text/html, */*"
}

@dataclass
class Reference:
    target_text: str
    category: str       # book, movie, wiki_entity
    title: str
    year: Optional[str]
    description: Optional[str]
    image_url: Optional[str]    # Direct URL if available
    source: str
    search_query: str           # Optimized query for image downloading later
    relevance: float = 0.0

def search_google_books(query: str, api_key: Optional[str] = None) -> List[Reference]:
    """Search Google Books. Handles the 'Image Not Available' URL issue."""
    params = {"q": f'intitle:"{query}"', "maxResults": 5, "printType": "books", "orderBy": "relevance"}
    if api_key: params["key"] = api_key

    try:
        resp = requests.get(GOOGLE_BOOKS_API, params=params, headers=HEADERS, timeout=10)
        if resp.status_code != 200: return []
        
        results = []
        for item in resp.json().get("items", []):
            info = item.get("volumeInfo", {})
            title = info.get("title", "")
            
            # Strict filtering: Target must be in title
            if query.lower() not in title.lower(): continue

            # Construct Image URL
            # Note: We prioritize the search query later, as these URLs often expire
            img_links = info.get("imageLinks", {})
            img_url = img_links.get("thumbnail", "").replace("http://", "https://")
            
            # Remove edge=curl to try and flatten the page, but keep zoom=1 (thumbnail)
            # because zoom=0 often requires auth. The downloader will upgrade this via search.
            if img_url:
                img_url = img_url.replace("&edge=curl", "")

            results.append(Reference(
                target_text=query,
                category="book",
                title=title,
                year=info.get("publishedDate", "")[:4],
                description=info.get("description", "")[:150],
                image_url=img_url,
                source="google_books",
                search_query=f'"{title}" book cover high resolution',
                relevance=10 if query.lower() == title.lower() else 5
            ))
        return results
    except Exception as e:
        print(f"Error Google Books: {e}")
        return []

def search_tmdb(query: str, api_key: str) -> List[Reference]:
    """Search TMDB. High reliability for posters."""
    if not api_key: return []
    
    try:
        params = {"query": query, "api_key": api_key}
        resp = requests.get(TMDB_SEARCH_MOVIE, params=params, headers=HEADERS, timeout=10)
        if resp.status_code != 200: return []

        results = []
        for item in resp.json().get("results", []):
            title = item.get("title", "")
            if query.lower() not in title.lower(): continue

            poster = item.get("poster_path")
            img_url = f"https://image.tmdb.org/t/p/original{poster}" if poster else None

            results.append(Reference(
                target_text=query,
                category="movie",
                title=title,
                year=item.get("release_date", "")[:4],
                description=item.get("overview", "")[:150],
                image_url=img_url,
                source="tmdb",
                search_query=f'"{title}" movie poster official',
                relevance=10 if query.lower() == title.lower() else 5
            ))
        return results
    except Exception:
        return []

def search_wikipedia(query: str) -> List[Reference]:
    """Search Wikipedia with PageImages. The 'Canonical' Source."""
    params = {
        "action": "query", "format": "json", "generator": "search",
        "gsrsearch": query, "gsrlimit": 5,
        "prop": "pageimages|description", "pithumbsize": 1000  # Request high res
    }

    try:
        resp = requests.get(WIKIPEDIA_API, params=params, headers=HEADERS, timeout=10)
        data = resp.json()
        pages = data.get("query", {}).get("pages", {})

        results = []
        for _, page in pages.items():
            title = page.get("title", "")
            if query.lower() not in title.lower(): continue

            img_url = page.get("thumbnail", {}).get("source")
            
            results.append(Reference(
                target_text=query,
                category="wikipedia_entity",
                title=title,
                year=None,
                description=page.get("description", ""),
                image_url=img_url,
                source="wikipedia",
                search_query=f'"{title}" wikipedia image',
                relevance=8
            ))
        return results
    except Exception:
        return []

def process_target(target: str, tmdb_key: str = None, gbooks_key: str = None) -> Dict:
    print(f"Searching for: {target}...")
    refs = []
    
    # Run Searches
    refs.extend(search_google_books(target, gbooks_key))
    refs.extend(search_tmdb(target, tmdb_key))
    refs.extend(search_wikipedia(target))

    # Deduplicate and Sort
    unique_refs = {}
    for r in refs:
        # Create a key based on title to dedup
        key = r.title.lower()
        if key not in unique_refs or r.image_url: # Prefer entries with images
            unique_refs[key] = r
            
    sorted_refs = sorted(unique_refs.values(), key=lambda x: x.relevance, reverse=True)
    
    return {
        "target": target,
        "best_match": asdict(sorted_refs[0]) if sorted_refs else None,
        "alternatives": [asdict(r) for r in sorted_refs[1:6]]
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, help="Single target string")
    parser.add_argument("--config", type=str, help="JSON file with list of strings")
    parser.add_argument("--output", type=str, default="references.json")
    parser.add_argument("--tmdb-key", default=os.getenv("TMDB_API_KEY"))
    parser.add_argument("--gbooks-key", default=os.getenv("GOOGLE_BOOKS_API_KEY"))
    args = parser.parse_args()

    results = {}

    targets = []
    if args.query: targets.append(args.query)
    if args.config:
        with open(args.config) as f:
            data = json.load(f)
            targets.extend(data if isinstance(data, list) else data.get("objects", []))

    for t in targets:
        data = process_target(t, args.tmdb_key, args.gbooks_key)
        results[t] = data
        time.sleep(0.5) # Be nice to APIs

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {len(results)} references to {args.output}")

if __name__ == "__main__":
    main()