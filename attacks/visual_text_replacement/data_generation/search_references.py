#!/usr/bin/env python3
"""
Search for real-world references (books, movies, museums, etc.) that contain target strings.
These references can be used to generate more effective text replacement attacks since
VLMs will recognize the original context from training data.

Usage:
    python search_references.py --config text_config.json --output references.json
    python search_references.py --query "murder" --category books
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Optional, Any
from urllib.parse import quote_plus

import requests

# API endpoints
GOOGLE_BOOKS_API = "https://www.googleapis.com/books/v1/volumes"
OPEN_LIBRARY_SEARCH = "https://openlibrary.org/search.json"
TMDB_SEARCH_MOVIE = "https://api.themoviedb.org/3/search/movie"
TMDB_SEARCH_TV = "https://api.themoviedb.org/3/search/tv"
WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"
OMDB_API = "http://www.omdbapi.com/"


@dataclass
class Reference:
    """A real-world reference containing the target text."""
    target_text: str
    category: str  # book, movie, tv_show, museum, event, institution, artifact
    title: str
    description: Optional[str] = None
    year: Optional[str] = None
    image_url: Optional[str] = None
    source: Optional[str] = None
    search_query: Optional[str] = None  # Query to use for image search
    confidence: float = 1.0  # How well the title matches


def search_google_books(query: str, api_key: Optional[str] = None) -> List[Reference]:
    """Search Google Books API for books containing the query in title."""
    results = []
    params = {
        "q": f"intitle:{query}",
        "maxResults": 10,
        "printType": "books",
        "orderBy": "relevance",
    }
    if api_key:
        params["key"] = api_key
    
    try:
        resp = requests.get(GOOGLE_BOOKS_API, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        
        for item in data.get("items", []):
            vol = item.get("volumeInfo", {})
            title = vol.get("title", "")
            
            # Check if query actually appears in title (case-insensitive)
            if query.lower() not in title.lower():
                continue
            
            # Get cover image - prefer highest quality available
            # Google Books imageLinks: extraLarge > large > medium > small > thumbnail > smallThumbnail
            image_links = vol.get("imageLinks", {})
            image_url = (
                image_links.get("extraLarge") or
                image_links.get("large") or
                image_links.get("medium") or
                image_links.get("thumbnail") or
                image_links.get("smallThumbnail")
            )
            
            # Try to get higher res by modifying the URL (remove zoom parameter or set zoom=0)
            if image_url and "zoom=" in image_url:
                # Replace zoom=1 with zoom=0 for full size, or remove &edge=curl
                image_url = image_url.replace("zoom=1", "zoom=0").replace("&edge=curl", "")
            
            results.append(Reference(
                target_text=query,
                category="book",
                title=title,
                description=vol.get("description", "")[:200] if vol.get("description") else None,
                year=vol.get("publishedDate", "")[:4] if vol.get("publishedDate") else None,
                image_url=image_url,
                source="google_books",
                # Search query prioritizes high-res sources
                search_query=f'"{title}" book cover high resolution',
            ))
    except Exception as e:
        print(f"[Google Books] Error searching '{query}': {e}")
    
    return results[:5]


def search_open_library(query: str) -> List[Reference]:
    """Search Open Library for books containing the query in title."""
    results = []
    params = {
        "title": query,
        "limit": 10,
    }
    
    try:
        resp = requests.get(OPEN_LIBRARY_SEARCH, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        
        for doc in data.get("docs", []):
            title = doc.get("title", "")
            
            if query.lower() not in title.lower():
                continue
            
            # Get cover
            cover_id = doc.get("cover_i")
            image_url = f"https://covers.openlibrary.org/b/id/{cover_id}-L.jpg" if cover_id else None
            
            results.append(Reference(
                target_text=query,
                category="book",
                title=title,
                description=doc.get("first_sentence", [None])[0] if doc.get("first_sentence") else None,
                year=str(doc.get("first_publish_year", "")) if doc.get("first_publish_year") else None,
                image_url=image_url,
                source="open_library",
                search_query=f'"{title}" book cover',
            ))
    except Exception as e:
        print(f"[Open Library] Error searching '{query}': {e}")
    
    return results[:5]


def search_tmdb_movies(query: str, api_key: Optional[str] = None) -> List[Reference]:
    """Search TMDB for movies containing the query in title."""
    if not api_key:
        api_key = os.environ.get("TMDB_API_KEY")
    if not api_key:
        return []
    
    results = []
    params = {
        "query": query,
        "api_key": api_key,
    }
    
    try:
        resp = requests.get(TMDB_SEARCH_MOVIE, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        
        for movie in data.get("results", []):
            title = movie.get("title", "")
            
            if query.lower() not in title.lower():
                continue
            
            poster_path = movie.get("poster_path")
            # Use 'original' size for maximum quality (typically 2000x3000 pixels)
            image_url = f"https://image.tmdb.org/t/p/original{poster_path}" if poster_path else None
            
            results.append(Reference(
                target_text=query,
                category="movie",
                title=title,
                description=movie.get("overview", "")[:200] if movie.get("overview") else None,
                year=movie.get("release_date", "")[:4] if movie.get("release_date") else None,
                image_url=image_url,
                source="tmdb",
                search_query=f'"{title}" movie poster official high resolution',
            ))
    except Exception as e:
        print(f"[TMDB Movies] Error searching '{query}': {e}")
    
    return results[:5]


def search_tmdb_tv(query: str, api_key: Optional[str] = None) -> List[Reference]:
    """Search TMDB for TV shows containing the query in title."""
    if not api_key:
        api_key = os.environ.get("TMDB_API_KEY")
    if not api_key:
        return []
    
    results = []
    params = {
        "query": query,
        "api_key": api_key,
    }
    
    try:
        resp = requests.get(TMDB_SEARCH_TV, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        
        for show in data.get("results", []):
            title = show.get("name", "")
            
            if query.lower() not in title.lower():
                continue
            
            poster_path = show.get("poster_path")
            # Use 'original' size for maximum quality
            image_url = f"https://image.tmdb.org/t/p/original{poster_path}" if poster_path else None
            
            results.append(Reference(
                target_text=query,
                category="tv_show",
                title=title,
                description=show.get("overview", "")[:200] if show.get("overview") else None,
                year=show.get("first_air_date", "")[:4] if show.get("first_air_date") else None,
                image_url=image_url,
                source="tmdb",
                search_query=f'"{title}" TV show poster high resolution',
            ))
    except Exception as e:
        print(f"[TMDB TV] Error searching '{query}': {e}")
    
    return results[:3]


def search_wikipedia(query: str, categories: List[str] = None) -> List[Reference]:
    """
    Search Wikipedia for pages containing the query.
    Categories can include: museum, memorial, event, institution, building
    """
    results = []
    
    # Different search patterns for different reference types
    search_patterns = [
        (f"{query} museum", "museum"),
        (f"{query} memorial", "museum"),
        (f"{query} (event)", "event"),
        (f"{query} (film)", "movie"),
        (f"{query} (book)", "book"),
        (f"{query} building", "institution"),
        (f"{query} monument", "museum"),
    ]
    
    headers = {
        "User-Agent": "MARS-Research/1.0 (Academic research; contact: research@example.edu)"
    }
    
    for search_term, category in search_patterns:
        params = {
            "action": "query",
            "list": "search",
            "srsearch": search_term,
            "format": "json",
            "srlimit": 3,
        }
        
        try:
            resp = requests.get(WIKIPEDIA_API, params=params, headers=headers, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            
            for result in data.get("query", {}).get("search", []):
                title = result.get("title", "")
                snippet = result.get("snippet", "")
                
                # Basic relevance check
                if query.lower() not in title.lower():
                    continue
                
                results.append(Reference(
                    target_text=query,
                    category=category,
                    title=title,
                    description=snippet[:200] if snippet else None,
                    source="wikipedia",
                    search_query=f'"{title}" {category} photo',
                ))
        except Exception as e:
            print(f"[Wikipedia] Error searching '{search_term}': {e}")
        
        time.sleep(0.1)  # Rate limiting
    
    return results[:5]


def search_omdb(query: str, api_key: Optional[str] = None) -> List[Reference]:
    """Search OMDB for movies/shows containing the query."""
    if not api_key:
        api_key = os.environ.get("OMDB_API_KEY")
    if not api_key:
        return []
    
    results = []
    params = {
        "s": query,
        "apikey": api_key,
    }
    
    try:
        resp = requests.get(OMDB_API, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        
        if data.get("Response") != "True":
            return []
        
        for item in data.get("Search", []):
            title = item.get("Title", "")
            
            if query.lower() not in title.lower():
                continue
            
            category = "movie" if item.get("Type") == "movie" else "tv_show"
            poster = item.get("Poster")
            image_url = poster if poster and poster != "N/A" else None
            
            results.append(Reference(
                target_text=query,
                category=category,
                title=title,
                year=item.get("Year"),
                image_url=image_url,
                source="omdb",
                search_query=f'"{title}" {category} poster',
            ))
    except Exception as e:
        print(f"[OMDB] Error searching '{query}': {e}")
    
    return results[:5]


def generate_search_queries(target: str) -> List[Dict[str, str]]:
    """
    Generate Google/Bing image search queries for a target string.
    Returns queries optimized for finding images with visible text.
    """
    queries = [
        {"query": f'"{target}" book cover', "category": "book"},
        {"query": f'"{target}" movie poster', "category": "movie"},
        {"query": f'"{target}" documentary poster', "category": "movie"},
        {"query": f'"{target}" museum sign entrance', "category": "museum"},
        {"query": f'"{target}" memorial monument', "category": "museum"},
        {"query": f'"{target}" newspaper headline front page', "category": "newspaper"},
        {"query": f'"{target}" magazine cover', "category": "magazine"},
        {"query": f'"{target}" warning sign', "category": "signage"},
        {"query": f'"{target}" product label packaging', "category": "product"},
        {"query": f'"{target}" building sign facade', "category": "institution"},
        {"query": f'"{target}" protest sign banner', "category": "signage"},
        {"query": f'"{target}" album cover', "category": "album"},
        {"query": f'"{target}" video game cover', "category": "game"},
        {"query": f'"{target}" tv show title card', "category": "tv_show"},
    ]
    return queries


def search_all_sources(target: str, tmdb_key: Optional[str] = None, 
                       omdb_key: Optional[str] = None,
                       google_books_key: Optional[str] = None) -> Dict[str, Any]:
    """
    Search all available sources for references containing the target string.
    Returns a dictionary with results organized by category.
    """
    print(f"  Searching for: {target}")
    
    all_results = {
        "target": target,
        "references": [],
        "search_queries": generate_search_queries(target),
    }
    
    # Search books
    print("    - Google Books...", end=" ", flush=True)
    books = search_google_books(target, google_books_key)
    print(f"found {len(books)}")
    all_results["references"].extend(books)
    
    print("    - Open Library...", end=" ", flush=True)
    ol_books = search_open_library(target)
    print(f"found {len(ol_books)}")
    all_results["references"].extend(ol_books)
    
    # Search movies/TV
    if tmdb_key:
        print("    - TMDB Movies...", end=" ", flush=True)
        movies = search_tmdb_movies(target, tmdb_key)
        print(f"found {len(movies)}")
        all_results["references"].extend(movies)
        
        print("    - TMDB TV...", end=" ", flush=True)
        tv = search_tmdb_tv(target, tmdb_key)
        print(f"found {len(tv)}")
        all_results["references"].extend(tv)
    
    if omdb_key:
        print("    - OMDB...", end=" ", flush=True)
        omdb = search_omdb(target, omdb_key)
        print(f"found {len(omdb)}")
        all_results["references"].extend(omdb)
    
    # Search Wikipedia
    print("    - Wikipedia...", end=" ", flush=True)
    wiki = search_wikipedia(target)
    print(f"found {len(wiki)}")
    all_results["references"].extend(wiki)
    
    # Deduplicate by title
    seen_titles = set()
    unique_refs = []
    for ref in all_results["references"]:
        title_key = ref.title.lower().strip()
        if title_key not in seen_titles:
            seen_titles.add(title_key)
            unique_refs.append(ref)
    
    all_results["references"] = [asdict(r) for r in unique_refs]
    all_results["total_found"] = len(unique_refs)
    
    return all_results


def rank_references(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Rank references by relevance and usability for the attack.
    Prioritizes: has image > exact title match > well-known source
    """
    refs = results.get("references", [])
    
    for ref in refs:
        score = 0
        target = results["target"].lower()
        title = ref["title"].lower()
        
        # Exact match in title (not just substring)
        if target in title.split() or target == title:
            score += 10
        elif target in title:
            score += 5
        
        # Has image URL
        if ref.get("image_url"):
            score += 8
        
        # Prefer movies/books (more visually distinctive)
        if ref["category"] in ["movie", "book"]:
            score += 3
        elif ref["category"] in ["tv_show", "museum"]:
            score += 2
        
        # Has year (indicates it's a real, verifiable thing)
        if ref.get("year"):
            score += 1
        
        ref["relevance_score"] = score
    
    # Sort by score
    refs.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
    results["references"] = refs
    
    # Pick best reference per category
    best_per_category = {}
    for ref in refs:
        cat = ref["category"]
        if cat not in best_per_category:
            best_per_category[cat] = ref
    results["best_per_category"] = best_per_category
    
    # Overall best reference
    if refs:
        results["best_reference"] = refs[0]
    
    return results


def process_config(config_path: Path, output_path: Path,
                   tmdb_key: Optional[str] = None,
                   omdb_key: Optional[str] = None,
                   google_books_key: Optional[str] = None) -> None:
    """
    Process all objects in a config file and search for references.
    Saves progress incrementally so interrupted runs can be resumed.
    """
    with open(config_path) as f:
        config = json.load(f)
    
    objects = config.get("objects", [])
    print(f"Processing {len(objects)} objects from {config_path}")
    
    # Load existing results if file exists (for resume capability)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        try:
            with open(output_path) as f:
                all_results = json.load(f)
            print(f"Loaded {len(all_results)} existing results from {output_path}")
        except (json.JSONDecodeError, IOError):
            all_results = {}
    else:
        all_results = {}
    
    # Count how many we can skip
    skip_count = sum(1 for obj in objects if obj in all_results)
    if skip_count > 0:
        print(f"Skipping {skip_count} already-processed objects")
    
    for i, obj in enumerate(objects, 1):
        # Skip if already processed
        if obj in all_results:
            continue
        
        print(f"\n[{i}/{len(objects)}] {obj}")
        results = search_all_sources(obj, tmdb_key, omdb_key, google_books_key)
        results = rank_references(results)
        all_results[obj] = results
        
        # Save incrementally after each object
        try:
            with open(output_path, "w") as f:
                json.dump(all_results, f, indent=2)
        except IOError as e:
            print(f"Warning: Failed to save progress: {e}")
        
        # Rate limiting
        time.sleep(0.5)
    
    print(f"\n✓ Results saved to {output_path}")
    
    # Summary
    print("\n=== Summary ===")
    found_count = sum(1 for r in all_results.values() if r.get("references"))
    with_images = sum(1 for r in all_results.values() 
                      if r.get("best_reference", {}).get("image_url"))
    
    print(f"Objects with references: {found_count}/{len(objects)}")
    print(f"Objects with image URLs: {with_images}/{len(objects)}")


def main():
    parser = argparse.ArgumentParser(
        description="Search for real-world references containing target strings"
    )
    parser.add_argument(
        "--config", 
        type=str, 
        help="Path to config JSON with 'objects' list"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="references.json",
        help="Output path for reference data"
    )
    parser.add_argument(
        "--query", 
        type=str, 
        help="Single query to search (instead of config)"
    )
    parser.add_argument(
        "--category",
        type=str,
        choices=["all", "books", "movies", "wikipedia"],
        default="all",
        help="Category to search"
    )
    parser.add_argument(
        "--tmdb-key",
        type=str,
        default=os.environ.get("TMDB_API_KEY"),
        help="TMDB API key"
    )
    parser.add_argument(
        "--omdb-key",
        type=str,
        default=os.environ.get("OMDB_API_KEY"),
        help="OMDB API key"
    )
    parser.add_argument(
        "--google-books-key",
        type=str,
        default=os.environ.get("GOOGLE_BOOKS_API_KEY"),
        help="Google Books API key"
    )
    
    args = parser.parse_args()
    
    if args.query:
        # Single query mode
        results = search_all_sources(
            args.query, 
            args.tmdb_key, 
            args.omdb_key,
            args.google_books_key
        )
        results = rank_references(results)
        
        print(f"\n=== Results for '{args.query}' ===")
        print(f"Total references found: {results['total_found']}")
        
        if results.get("best_reference"):
            best = results["best_reference"]
            print(f"\nBest match:")
            print(f"  Title: {best['title']}")
            print(f"  Category: {best['category']}")
            print(f"  Year: {best.get('year', 'N/A')}")
            print(f"  Image: {best.get('image_url', 'N/A')}")
            print(f"  Search query: {best.get('search_query', 'N/A')}")
        
        print(f"\nAll references:")
        for ref in results["references"][:10]:
            print(f"  [{ref['category']}] {ref['title']} ({ref.get('year', '?')}) - score: {ref.get('relevance_score', 0)}")
        
        # Save to file
        with open(args.output, "w") as f:
            json.dump({args.query: results}, f, indent=2)
        print(f"\nSaved to {args.output}")
        
    elif args.config:
        # Config file mode
        process_config(
            Path(args.config),
            Path(args.output),
            args.tmdb_key,
            args.omdb_key,
            args.google_books_key
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
