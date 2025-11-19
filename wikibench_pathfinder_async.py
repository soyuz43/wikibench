#!/usr/bin/env python3
"""
wikibench_pathfinder_async.py
-----------------------------------
Efficient Wikipedia link pathfinder with:
 - Local caching (JSON per page)
 - Async concurrent link fetching (HTML parsed)
 - Bidirectional BFS
 - Optional live graph visualization of visited nodes
 - Polite rate limiting & retry/backoff
 - Strict visible in-body link extraction (now delegated to wikibench_linkextractor)
 - Path verification (each hop must be a real visible link)
"""

import os
import json
import time
import asyncio
import aiohttp
import urllib.parse
import networkx as nx
import matplotlib.pyplot as plt
from dotenv import load_dotenv

# NEW: import shared extractor module
from wikibench_linkextractor import extract_link_titles

# Load environment variables from .env file if present
load_dotenv()

CACHE_DIR = "./cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# ----------------------------------------------------------------------
# Logging utilities
# ----------------------------------------------------------------------
def debug(msg: str):
    """Debug logger controlled by WIKIBENCH_DEBUG env var."""
    if os.getenv("WIKIBENCH_DEBUG", "0").lower() in ("1", "true", "yes", "on"):
        print(f"[DEBUG] {msg}")

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
    "User-Agent": "WikiBenchBot/0.3 (+https://github.com/soyuz43; contact: kebekad673@proton.me)",
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Language": "en-US,en;q=0.9",
}

# Politeness / concurrency (defaults: ~6 rps; 8 concurrent connections)
MAX_RPS = float(os.getenv("WIKIBENCH_MAX_RPS", "6"))
MAX_CONCURRENCY = int(os.getenv("WIKIBENCH_MAX_CONCURRENCY", "8"))
MAX_RETRIES = 5

# ------------------------------------------------------
# Cache Helpers
# ------------------------------------------------------

def cache_path(title: str) -> str:
    safe = title.replace("/", "_").replace(" ", "_")
    return os.path.join(CACHE_DIR, f"{safe}.json")


def load_cached_links(title: str):
    path = cache_path(title)
    if os.path.exists(path):
        debug(f"Loading cached links for {title} from {path}")
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def save_cached_links(title: str, links):
    path = cache_path(title)
    debug(f"Saving {len(links)} links for {title} to {path}")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(links, f)


# ------------------------------------------------------
# Filtering & Normalization
# ------------------------------------------------------

def _norm_for_compare(title: str) -> str:
    # Compare both underscore and space forms fairly
    return title.replace("_", " ")


def should_exclude(title: str) -> bool:
    # Accepts underscore titles; compare against both variants
    t_space = _norm_for_compare(title)
    if t_space in EXCLUDE_TITLES:
        return True
    if any(p.lower() in t_space.lower() or p.lower().replace(" ", "_") in title.lower()
           for p in EXCLUDE_PATTERNS):
        return True
    if ":" in title:  # non-article namespaces (e.g., File:, Category:, Help:)
        return True
    return False


# ------------------------------------------------------
# Async Fetching (Visible Links) + Politeness
# ------------------------------------------------------

class RateLimiter:
    """Simple token bucket ~ MAX_RPS per second for all tasks."""
    def __init__(self, rate: float):
        self.rate = max(rate, 0.1)
        self._last = time.monotonic()
        self._allowance = self.rate
        self._lock = asyncio.Lock()

    async def acquire(self):
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self._last
            self._last = now
            self._allowance += elapsed * self.rate
            if self._allowance > self.rate:
                self._allowance = self.rate
            if self._allowance < 1.0:
                # sleep until 1 token available
                wait = (1.0 - self._allowance) / self.rate
                debug(f"RateLimiter sleeping {wait:.3f}s")
                await asyncio.sleep(wait)
                self._allowance = 0.0
            else:
                self._allowance -= 1.0


rate_limiter = RateLimiter(MAX_RPS)


async def fetch_visible_links(session: aiohttp.ClientSession, title: str):
    """
    Fetch visible article links using the shared HTML link extractor.

    Replaces the old bespoke BeautifulSoup traversal with a clean call to:
        extract_link_titles(html)
    """

    cached = load_cached_links(title)
    if cached is not None:
        filtered = [l for l in cached if not should_exclude(l)]
        debug(f"[FETCH] {title}: using cache with {len(filtered)} links (raw {len(cached)})")
        return filtered

    title_norm = title.replace(" ", "_")
    url = f"https://en.wikipedia.org/wiki/{title_norm}"
    debug(f"[FETCH] {title}: requesting {url}")

    # polite limiter + robust retry/backoff
    attempt = 0
    html = None
    while attempt < MAX_RETRIES:
        await rate_limiter.acquire()
        attempt += 1
        try:
            async with session.get(url, allow_redirects=True) as resp:
                status = resp.status
                debug(f"[HTTP] {title}: status={status} attempt={attempt}")
                if status == 200:
                    html = await resp.text(errors="ignore")
                    break
                elif status in (429, 500, 502, 503, 504):
                    backoff = (2 ** (attempt - 1)) + (0.1 * attempt)
                    print(f"[WARN] {status} on {title}; backoff {backoff:.2f}s (attempt {attempt})")
                    await asyncio.sleep(backoff)
                else:
                    print(f"[WARN] Failed to fetch {title} ({status})")
                    return []
        except Exception as e:
            backoff = (2 ** (attempt - 1)) + (0.1 * attempt)
            print(f"[WARN] Network error fetching {title}: {e}; backoff {backoff:.2f}s")
            await asyncio.sleep(backoff)

    if html is None:
        debug(f"[FETCH] {title}: no HTML after {MAX_RETRIES} attempts")
        return []

    # 🌟 NEW: extract all visible link titles using your shared extractor module
    raw_links = extract_link_titles(html)
    debug(f"[PARSE] {title}: extracted {len(raw_links)} raw links")

    # Apply exclusions
    final_links = sorted({l for l in raw_links if not should_exclude(l)})
    debug(f"[FILTER] {title}: {len(final_links)} links remain after exclusions")

    save_cached_links(title, final_links)
    return final_links


async def verify_path(session: aiohttp.ClientSession, path: list[str]) -> bool:
    """Re-verify each hop as a real visible link."""
    debug(f"[VERIFY] Verifying path: {' -> '.join(path)}")
    for src, dst in zip(path, path[1:]):
        out = await fetch_visible_links(session, src)
        found = dst in out
        debug(f"[VERIFY] {src} → {dst} | found={found} | out_count={len(out)}")
        if not found:
            print(f"[VERIFY] Missing edge {src} -> {dst}")
            return False
    return True


# ------------------------------------------------------
# Bidirectional BFS
# ------------------------------------------------------

async def bidirectional_bfs(start, target, visualize=False, max_depth=6):
    """
    Perform bidirectional BFS with visible link fetching and hop verification.
    Returns: (path, meeting_node, parent_start, parent_target)
      - path: list[str] | None
      - meeting_node: str | None
      - parent_start: dict[child->parent] from the forward search
      - parent_target: dict[child->parent] from the backward search
    """
    if start == target:
        return [start], start, {}, {}

    # frontier maps: node -> path so far (kept for intersection checks)
    front_start = {start: [start]}
    front_target = {target: [target]}
    visited_start = {start}
    visited_target = {target}

    # parent maps for visualization
    parent_start: dict[str, str] = {}
    parent_target: dict[str, str] = {}

    meeting_node = None

    G = nx.Graph()
    G.add_nodes_from([start, target])
    pos = nx.spring_layout(G, seed=42, k=0.8)

    if visualize:
        plt.ion()
        plt.figure(figsize=(8, 6))

    connector = aiohttp.TCPConnector(limit=MAX_CONCURRENCY)
    timeout = aiohttp.ClientTimeout(total=30)
    async with aiohttp.ClientSession(headers=HEADERS, connector=connector, timeout=timeout) as session:
        depth = 0
        while front_start and front_target and depth < max_depth:
            depth += 1
            print(f"[INFO] Depth {depth} | frontier sizes: {len(front_start)} & {len(front_target)}")
            debug(f"[FRONT] start: {list(front_start.keys())[:5]} ...")
            debug(f"[FRONT] target: {list(front_target.keys())[:5]} ...")

            # -------- Expand from start side
            new_front = {}
            tasks = [asyncio.create_task(fetch_visible_links(session, n)) for n in front_start]
            results = await asyncio.gather(*tasks)

            for node, links in zip(front_start, results):
                debug(f"[EXPAND start] {node}: {len(links)} links")
                for link in links:
                    if link not in visited_start:
                        visited_start.add(link)
                        parent_start[link] = node
                        new_path = front_start[node] + [link]
                        new_front[link] = new_path
                        G.add_edge(node, link)

                        if link in front_target:
                            # found meeting
                            meeting_node = link
                            debug(f"[MEET] at {meeting_node} from start-side expansion")
                            path1 = new_path
                            path2 = front_target[link]
                            full = path1 + path2[::-1][1:]
                            if await verify_path(session, full):
                                if visualize:
                                    pos = nx.spring_layout(G, pos=pos, fixed=list(pos.keys()), k=0.8)
                                    visualize_graph(G, link, target, pos)
                                    plt.ioff()
                                    plt.show()
                                return full, meeting_node, parent_start, parent_target
                            else:
                                print("[INFO] Discarded false path after verification; continuing.")
                                if visualize:
                                    pos = nx.spring_layout(G, pos=pos, fixed=list(pos.keys()), k=0.8)
                                    visualize_graph(G, link, target, pos)

            front_start = new_front

            # -------- Expand from target side
            new_front = {}
            tasks = [asyncio.create_task(fetch_visible_links(session, n)) for n in front_target]
            results = await asyncio.gather(*tasks)

            for node, links in zip(front_target, results):
                debug(f"[EXPAND target] {node}: {len(links)} links")
                for link in links:
                    if link not in visited_target:
                        visited_target.add(link)
                        parent_target[link] = node
                        new_path = front_target[node] + [link]
                        new_front[link] = new_path
                        G.add_edge(node, link)

                        if link in front_start:
                            meeting_node = link
                            debug(f"[MEET] at {meeting_node} from target-side expansion")
                            path1 = front_start[link]
                            path2 = new_path
                            full = path1 + path2[::-1][1:]
                            if await verify_path(session, full):
                                if visualize:
                                    pos = nx.spring_layout(G, pos=pos, fixed=list(pos.keys()), k=0.8)
                                    visualize_graph(G, link, target, pos)
                                    plt.ioff()
                                    plt.show()
                                return full, meeting_node, parent_start, parent_target
                            else:
                                print("[INFO] Discarded false path after verification; continuing.")
                                if visualize:
                                    pos = nx.spring_layout(G, pos=pos, fixed=list(pos.keys()), k=0.8)
                                    visualize_graph(G, link, target, pos)

            front_target = new_front

            if visualize:
                pos = nx.spring_layout(G, pos=pos, fixed=list(pos.keys()), k=0.8)
                visualize_graph(G, None, target, pos)

    if visualize:
        plt.ioff()
        plt.show()
    return None, None, parent_start, parent_target


# ------------------------------------------------------
# Visualization helpers (unchanged)
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


def visualize_search_path(forward_parents, backward_parents, meeting_node, start, goal, cache_dir):
    """
    Generate a simple text visualization of how the bidirectional search met.
    Writes a formatted tree-style view of the forward and backward exploration
    to a subdirectory within the cache directory (cache/viz/).
    """
    def build_path(parents, node):
        path = [node]
        while node in parents:
            node = parents[node]
            path.append(node)
        path.reverse()
        return path

    # forward: start -> ... -> meeting
    forward_path = build_path(forward_parents, meeting_node)
    # backward: target -> ... -> meeting
    backward_path = build_path(backward_parents, meeting_node)

    forward_depth = len(forward_path) - 1
    backward_depth = len(backward_path) - 1

    lines = []
    lines.append(f"Forward search (depth 0–{forward_depth}):")
    for i, node in enumerate(forward_path):
        indent = "     " * i
        lines.append(f"{indent}{node}")
        if i < len(forward_path) - 1:
            lines.append(f"{indent} └─> {forward_path[i+1]}")

    lines.append("")
    lines.append(f"Backward search (depth 0–{backward_depth}):")
    for i, node in enumerate(backward_path):
        indent = "     " * i
        lines.append(f"{indent}{node}")
        if i < len(backward_path) - 1:
            lines.append(f"{indent} └─> {backward_path[i+1]}")

    viz_text = "\n".join(lines)
    print(viz_text)

    # create subdirectory inside cache for visualizations
    viz_dir = os.path.join(cache_dir, "viz")
    os.makedirs(viz_dir, exist_ok=True)

    out_path = os.path.join(viz_dir, f"search_viz_{start}_to_{goal}.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(viz_text)

    print(f"[INFO] Saved search visualization to {out_path}")


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

    visualize = os.getenv("WIKIBENCH_VIZ", "0") == "1"

    print(f"[INFO] Searching path from '{start}' to '{target}' (max depth {max_depth})")
    print(f"[INFO] Viz={'ON' if visualize else 'OFF'} | MAX_RPS={MAX_RPS} | CONC={MAX_CONCURRENCY}")
    start_time = time.time()

    path, meeting_node, parent_start, parent_target = asyncio.run(
        bidirectional_bfs(start, target, visualize=visualize, max_depth=max_depth)
    )

    elapsed = time.time() - start_time

    if path:
        print(f"[SUCCESS] Path found in {len(path)-1} hops ({elapsed:.2f}s):")
        for step in path:
            print(f" → {step}")
        print(json.dumps(path, indent=2))

        if meeting_node is not None:
            visualize_search_path(parent_start, parent_target, meeting_node, start, target, CACHE_DIR)
    else:
        print(f"[FAILURE] No path found within depth {max_depth} ({elapsed:.2f}s)")
