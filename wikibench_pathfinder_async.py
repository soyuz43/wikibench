#!/usr/bin/env python3
"""
wikibench_pathfinder_async.py
-----------------------------------
Efficient Wikipedia link pathfinder with:
 - Local caching (JSON per page)
 - Async concurrent link fetching (HTML parsed)
 - Bidirectional BFS
 - Live graph visualization of visited nodes
"""

import os
import json
import time
import asyncio
import aiohttp
import networkx as nx
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup

API = "https://en.wikipedia.org/w/api.php"
CACHE_DIR = "./cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# Exclusions
EXCLUDE_TITLES = {
    "Doi (identifier)", "ISBN (identifier)", "Internet Archive",
    "Wayback Machine", "OCLC (identifier)", "PMC (identifier)",
    "PMID (identifier)", "ISSN (identifier)", "CiteSeerX (identifier)",
    "ARIA Charts", "Billboard 200", "Music Canada",
    "Recording Industry Association of America"
}

EXCLUDE_PATTERNS = [
    " (identifier)", " charts", " (magazine)", " (website)",
    "Archive.org", "AllMusic", "Discogs", "MusicBrainz"
]

HEADERS = {
    "User-Agent": "WikiBenchBot/0.3 (https://github.com/soyuz43; contact: kebekad673@proton.me)"
}

# ------------------------------------------------------
# Cache Helpers
# ------------------------------------------------------

def cache_path(title: str) -> str:
    safe = title.replace("/", "_").replace(" ", "_")
    return os.path.join(CACHE_DIR, f"{safe}.json")

def load_cached_links(title: str):
    path = cache_path(title)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

def save_cached_links(title: str, links):
    path = cache_path(title)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(links, f)

# ------------------------------------------------------
# Filtering
# ------------------------------------------------------

def should_exclude(title: str) -> bool:
    if title in EXCLUDE_TITLES:
        return True
    if any(p.lower() in title.lower() for p in EXCLUDE_PATTERNS):
        return True
    if ":" in title:  # non-article namespaces
        return True
    return False

# ------------------------------------------------------
# Async Fetching (Visible Links)
# ------------------------------------------------------


async def fetch_visible_links(session, title):
    """Fetch visible article links from Wikipedia, robust to HTML structure changes."""
    cached = load_cached_links(title)
    if cached:
        return [l for l in cached if not should_exclude(l)]

    title_norm = title.replace(" ", "_")
    url = f"https://en.wikipedia.org/wiki/{title_norm}"

    try:
        async with session.get(
            url,
            headers={"User-Agent": "Mozilla/5.0 (WikiBenchBot)"},
            allow_redirects=True
        ) as resp:
            if resp.status != 200:
                print(f"[WARN] Failed to fetch {title} ({resp.status})")
                return []
            html = await resp.text(errors="ignore")
    except Exception as e:
        print(f"[WARN] Network error fetching {title}: {e}")
        return []

    soup = BeautifulSoup(html, "html.parser")

    # Ensure we're looking inside the right div
    content_div = soup.find("div", {"id": "mw-content-text"})
    if content_div:
        content_div = content_div.find("div", {"class": "mw-parser-output"}) or content_div
    else:
        print(f"[WARN] No content div found for {title}")
        return []

    links = set()
    for tag in content_div.find_all(["p", "ul", "ol"]):
        for a in tag.find_all("a", href=True):
            href = a["href"]
            if not href.startswith("/wiki/"):
                continue
            if ":" in href:  # skip non-article namespaces
                continue
            target = href.split("/wiki/")[-1]
            if should_exclude(target):
                continue
            links.add(target)

    links = sorted(links)
    save_cached_links(title, links)
    await asyncio.sleep(0.2)
    return links


# ------------------------------------------------------
# Bidirectional BFS
# ------------------------------------------------------

async def bidirectional_bfs(start, target, visualize=True, max_depth=6):
    """Perform bidirectional BFS with visible link fetching."""
    if start == target:
        return [start]

    front_start = {start: [start]}
    front_target = {target: [target]}
    visited_start = {start}
    visited_target = {target}

    G = nx.Graph()
    G.add_nodes_from([start, target])
    pos = nx.spring_layout(G, seed=42, k=0.8)

    plt.ion()
    fig = plt.figure(figsize=(8, 6))

    async with aiohttp.ClientSession(headers=HEADERS) as session:
        depth = 0
        while front_start and front_target and depth < max_depth:
            depth += 1
            print(f"[INFO] Depth {depth} | frontier sizes: {len(front_start)} & {len(front_target)}")

            # Expand from start
            new_front = {}
            tasks = [fetch_visible_links(session, n) for n in front_start]
            results = await asyncio.gather(*tasks)

            for node, links in zip(front_start, results):
                for link in links:
                    if link not in visited_start:
                        visited_start.add(link)
                        new_path = front_start[node] + [link]
                        new_front[link] = new_path
                        G.add_edge(node, link)

                        if link in front_target:
                            path1 = new_path
                            path2 = front_target[link]
                            full = path1 + path2[::-1][1:]
                            if visualize:
                                pos = nx.spring_layout(G, pos=pos, fixed=list(pos.keys()), k=0.8)
                                visualize_graph(G, link, target, pos)
                            plt.ioff()
                            plt.show()
                            return full

            front_start = new_front

            # Expand from target
            new_front = {}
            tasks = [fetch_visible_links(session, n) for n in front_target]
            results = await asyncio.gather(*tasks)

            for node, links in zip(front_target, results):
                for link in links:
                    if link not in visited_target:
                        visited_target.add(link)
                        new_path = front_target[node] + [link]
                        new_front[link] = new_path
                        G.add_edge(node, link)

                        if link in front_start:
                            path1 = front_start[link]
                            path2 = new_path
                            full = path1 + path2[::-1][1:]
                            if visualize:
                                pos = nx.spring_layout(G, pos=pos, fixed=list(pos.keys()), k=0.8)
                                visualize_graph(G, link, target, pos)
                            plt.ioff()
                            plt.show()
                            return full

            front_target = new_front

            if visualize:
                pos = nx.spring_layout(G, pos=pos, fixed=list(pos.keys()), k=0.8)
                visualize_graph(G, None, target, pos)

    plt.ioff()
    plt.show()
    return None

# ------------------------------------------------------
# Visualization
# ------------------------------------------------------

def visualize_graph(G, current_node, target, pos):
    plt.clf()
    node_colors = [
        "red" if n == target else
        "green" if n == current_node else
        "lightblue"
        for n in G.nodes()
    ]
    nx.draw(G, pos, with_labels=False, node_color=node_colors, node_size=80, alpha=0.8)
    plt.title(f"Exploring: {current_node if current_node else '...'}")
    plt.pause(0.001)

# ------------------------------------------------------
# CLI Entry
# ------------------------------------------------------

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: ./wikibench_pathfinder_async.py <start_title> <target_title> [max_depth]")
        sys.exit(1)

    start = sys.argv[1].replace(" ", "_")
    target = sys.argv[2].replace(" ", "_")
    max_depth = int(sys.argv[3]) if len(sys.argv) > 3 else 6

    print(f"[INFO] Searching path from '{start}' to '{target}' (max depth {max_depth})")
    start_time = time.time()
    path = asyncio.run(bidirectional_bfs(start, target, visualize=True, max_depth=max_depth))
    elapsed = time.time() - start_time

    if path:
        print(f"[SUCCESS] Path found in {len(path)-1} hops ({elapsed:.2f}s):")
        for step in path:
            print(f" → {step}")
        print(json.dumps(path, indent=2))
    else:
        print(f"[FAILURE] No path found within depth {max_depth} ({elapsed:.2f}s)")
