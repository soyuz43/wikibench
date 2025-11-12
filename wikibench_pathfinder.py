#!/usr/bin/env python3
"""
wikibench_pathfinder.py
Finds a path of Wikipedia article links between two pages using BFS.
"""

import sys
import time
import json
import requests
from collections import deque

API = "https://en.wikipedia.org/w/api.php"
HEADERS = {
    "User-Agent": "WikiBenchBot/0.1 (https://github.com/soyuz43; contact: kebekad673@proton.me)"
}

def get_links(title):
    """Fetch all linked article titles from a Wikipedia page."""
    links = []
    params = {
        "action": "query",
        "titles": title,
        "prop": "links",
        "pllimit": "max",
        "format": "json",
        "redirects": 1
    }
    while True:
        resp = requests.get(API, params=params, headers=HEADERS)
        if resp.status_code != 200:
            print(f"[WARN] Failed to fetch links for {title} (HTTP {resp.status_code})")
            return links
        data = resp.json()
        pages = data.get("query", {}).get("pages", {})
        for page in pages.values():
            if "links" in page:
                links.extend(l["title"] for l in page["links"])
        if "continue" in data:
            params.update(data["continue"])
        else:
            break
        time.sleep(0.2)  # Respect rate limits
    return links

def shortest_path(start, target, max_depth=5):
    """Find shortest Wikipedia link path between start and target using BFS."""
    visited = {start}
    queue = deque([(start, [start])])
    while queue:
        page, path = queue.popleft()
        if page == target:
            return path
        if len(path) > max_depth:
            continue
        for link in get_links(page):
            if link not in visited:
                visited.add(link)
                queue.append((link, path + [link]))
    return None

def main():
    if len(sys.argv) < 3:
        print("Usage: ./wikibench_pathfinder.py <start_title> <target_title> [max_depth]")
        sys.exit(1)

    start = sys.argv[1].replace(" ", "_")
    target = sys.argv[2].replace(" ", "_")
    max_depth = int(sys.argv[3]) if len(sys.argv) > 3 else 5

    print(f"[INFO] Searching path from '{start}' to '{target}' (max depth {max_depth})")
    start_time = time.time()

    path = shortest_path(start, target, max_depth=max_depth)

    elapsed = time.time() - start_time
    if path:
        print(f"[SUCCESS] Path found in {len(path)-1} hops ({elapsed:.2f}s):")
        for step in path:
            print(f" → {step}")
        print(json.dumps(path, indent=2))
    else:
        print(f"[FAILURE] No path found within depth {max_depth} ({elapsed:.2f}s)")

if __name__ == "__main__":
    main()
