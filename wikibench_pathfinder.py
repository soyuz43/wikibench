#!/usr/bin/env python3
"""
wikibench_pathfinder_async.py
-----------------------------------
Efficient Wikipedia link pathfinder with:
 - Local caching (JSON per page)
 - Async concurrent link fetching
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
from collections import deque

API = "https://en.wikipedia.org/w/api.php"
CACHE_DIR = "./cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# Define pages to exclude from the search
EXCLUDE_TITLES = {
    "Doi (identifier)",  # Add other titles you want to exclude here
    "ISBN (identifier)",  # "Another Page Title",
    "Internet Archive",  # "Yet Another Title",
    "ISSN (identifier)"
}

HEADERS = {
    "User-Agent": "WikiBenchBot/0.2 (https://github.com/soyuz43; contact: kebekad673@proton.me)"
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
# Async Fetching
# ------------------------------------------------------

async def fetch_links(session, title):
    """Fetch all linked article titles from a Wikipedia page, with caching and continuation."""
    cached = load_cached_links(title)
    if cached:
        # Apply exclusion filter to cached results as well
        return [link for link in cached if link not in EXCLUDE_TITLES]

    params = {
        "action": "query",
        "titles": title,
        "prop": "links",
        "plnamespace": 0,   # only main articles, not categories/templates
        "pllimit": "max",
        "format": "json",
        "redirects": 1
    }
    all_links = []  # Aggregate links from all batches here
    while True:
        async with session.get(API, params=params) as resp:
            if resp.status != 200:
                print(f"[WARN] Failed to fetch {title} ({resp.status})")
                break # Return what we have so far, or handle error differently
            data = await resp.json()
            pages = data.get("query", {}).get("pages", {})
            for p in pages.values():
                if "links" in p:
                    # Extend the main list with links from this batch, excluding unwanted ones
                    all_links.extend(l["title"] for l in p["links"] if l["title"] not in EXCLUDE_TITLES)
            if "continue" in data:
                params.update(data["continue"])
            else:
                break
        await asyncio.sleep(0.1)  # be polite
    save_cached_links(title, all_links) # Save the filtered, complete list
    return all_links

# ------------------------------------------------------
# Bidirectional BFS
# ------------------------------------------------------

async def bidirectional_bfs(start, target, visualize=True, max_depth=6):
    """Perform bidirectional BFS with async link fetching and optional live visualization."""
    if start == target:
        return [start]

    # State
    front_start = {start: [start]}
    front_target = {target: [target]}
    visited_start = {start}
    visited_target = {target}
    G = nx.Graph()
    G.add_node(start)
    G.add_node(target)

    # Initialize positions for the initial graph state
    pos = nx.spring_layout(G, seed=42, k=0.8)

    plt.ion()
    fig = plt.figure(figsize=(8, 6))

    async with aiohttp.ClientSession(headers=HEADERS) as session:
        depth = 0
        while front_start and front_target and depth < max_depth:
            depth += 1
            print(f"[INFO] Depth {depth} | frontier sizes: {len(front_start)} & {len(front_target)}")

            # Expand from start side
            new_front = {}
            tasks = [fetch_links(session, node) for node in front_start]
            results = await asyncio.gather(*tasks)

            for node, links in zip(front_start, results):
                for link in links:
                    if link not in visited_start:
                        visited_start.add(link)
                        new_path = front_start[node] + [link]
                        new_front[link] = new_path
                        G.add_edge(node, link)

                        # Check if we meet the other frontier
                        if link in front_target:
                            path1 = new_path
                            path2 = front_target[link]
                            full = path1 + path2[::-1][1:]
                            if visualize:
                                # Update layout before final visualization
                                pos = nx.spring_layout(G, pos=pos, fixed=list(pos.keys()), k=0.8)
                                visualize_graph(G, link, target, pos)
                            plt.ioff()
                            plt.show()
                            return full

            front_start = new_front

            # Expand from target side symmetrically
            new_front = {}
            tasks = [fetch_links(session, node) for node in front_target]
            results = await asyncio.gather(*tasks)

            for node, links in zip(front_target, results):
                for link in links:
                    if link not in visited_target:
                        visited_target.add(link)
                        new_path = front_target[node] + [link]
                        new_front[link] = new_path
                        G.add_edge(node, link)

                        # Check if we meet
                        if link in front_start:
                            path1 = front_start[link]
                            path2 = new_path
                            full = path1 + path2[::-1][1:]
                            if visualize:
                                # Update layout before final visualization
                                pos = nx.spring_layout(G, pos=pos, fixed=list(pos.keys()), k=0.8)
                                visualize_graph(G, link, target, pos)
                            plt.ioff()
                            plt.show()
                            return full

            front_target = new_front

            if visualize:
                # Update layout after adding new nodes/edges for this depth iteration
                # Use the previous positions as a base ('pos' dict) and fix those nodes.
                # This helps maintain stability in the visualization.
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
    node_colors = []
    for n in G.nodes():
        if n == target:
            node_colors.append("red")
        elif n == current_node:
            node_colors.append("green")
        else:
            node_colors.append("lightblue")

    nx.draw(G, pos, with_labels=False, node_color=node_colors, node_size=80, alpha=0.8)
    plt.title(f"Exploring: {current_node if current_node else '...' }")
    plt.pause(0.001)

# ------------------------------------------------------
# CLI Entry
# ------------------------------------------------------

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: ./wikibench_pathfinder.py <start_title> <target_title> [max_depth]")
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
