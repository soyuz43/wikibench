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
 - Strict visible in-body link extraction
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
from bs4 import BeautifulSoup

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
    "User-Agent": "WikiBenchBot/0.3 (+https://github.com/soyuz43; contact: kebekad673@proton.me)",
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Language": "en-US,en;q=0.9",
}

# Politeness / concurrency (defaults: ~6 rps; 8 concurrent connections)
MAX_RPS = float(os.getenv("WIKIBENCH_MAX_RPS", "6"))
MAX_CONCURRENCY = int(os.getenv("WIKIBENCH_MAX_CONCURRENCY", "8"))
MAX_RETRIES = 5

# Disallowed ancestor classes when harvesting links
DISALLOWED_ANCESTORS = {
    "navbox", "vertical-navbox", "sidebar", "infobox", "toc", "mw-references-wrap",
    "reflist", "hatnote", "thumb", "gallery", "metadata", "mbox-small", "sistersitebox"
}

# Stop scanning after these H2 sections
STOP_SECTIONS = {"See also", "References", "External links", "Further reading", "Notes"}


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


def normalize_href_to_title(href: str) -> str | None:
    # Must be a wiki article link
    if not href.startswith("/wiki/"):
        return None
    # Strip query/fragment
    href = href.split("?", 1)[0].split("#", 1)[0]
    # Extract title segment
    raw = href[len("/wiki/"):]
    if not raw:
        return None
    # Percent-decode but keep underscores (canonical)
    decoded = urllib.parse.unquote(raw)
    # Filter namespaces early
    if ":" in decoded:
        return None
    return decoded


def has_disallowed_ancestor(tag) -> bool:
    for parent in tag.parents:
        if getattr(parent, "get", None) is None:
            continue
        cls = parent.get("class") or []
        if any(c in DISALLOWED_ANCESTORS for c in cls):
            return True
        # Some boxes use role=navigation
        role = parent.get("role")
        if role == "navigation":
            return True
        # Stop at main parser output
        if parent.get("class") and "mw-parser-output" in parent.get("class", []):
            break
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
                await asyncio.sleep(wait)
                self._allowance = 0.0
            else:
                self._allowance -= 1.0


rate_limiter = RateLimiter(MAX_RPS)


# Stop scanning after these H2 sections
STOP_SECTIONS = {"See also", "References", "External links", "Further reading", "Notes"}

def _is_stop_h2(tag) -> bool:
    """Return True if this <h2> is a terminal section like 'See also', 'References', etc."""
    if not tag or getattr(tag, "name", None) != "h2":
        return False
    htxt = tag.get_text(separator=" ", strip=True)
    return any(s in htxt for s in STOP_SECTIONS)

async def fetch_visible_links(session: aiohttp.ClientSession, title: str):
    """Fetch visible article links from Wikipedia, robust to HTML structure changes."""
    cached = load_cached_links(title)
    if cached:
        return [l for l in cached if not should_exclude(l)]

    title_norm = title.replace(" ", "_")
    url = f"https://en.wikipedia.org/wiki/{title_norm}"

    # polite limiter + robust retry/backoff
    attempt = 0
    html = None
    while attempt < MAX_RETRIES:
        await rate_limiter.acquire()
        attempt += 1
        try:
            async with session.get(url, allow_redirects=True) as resp:
                status = resp.status
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
        return []

    soup = BeautifulSoup(html, "html.parser")

    # Main content root
    content_div = soup.find("div", {"id": "mw-content-text"})
    if content_div:
        content_div = content_div.find("div", {"class": "mw-parser-output"}) or content_div
    else:
        print(f"[WARN] No content div found for {title}")
        return []

    # Walk descendants in DOM order; stop when we hit a terminal <h2>
    links: set[str] = set()
    stop_seen = False
    for node in content_div.descendants:
        if stop_seen:
            break
        if getattr(node, "name", None) == "h2":
            if _is_stop_h2(node):
                stop_seen = True
            continue
        name = getattr(node, "name", None)
        if name not in ("p", "ul", "ol"):
            continue
        # harvest anchors but exclude those under disallowed ancestors
        for a in node.find_all("a", href=True):
            if has_disallowed_ancestor(a):
                continue
            target = normalize_href_to_title(a["href"])
            if not target:
                continue
            if should_exclude(target):
                continue
            links.add(target)

    links_sorted = sorted(links)
    save_cached_links(title, links_sorted)

    # Optional debug (set WIKIBENCH_DEBUG=1)
    if os.getenv("WIKIBENCH_DEBUG") == "1":
        print(f"[DEBUG] Extracted {len(links_sorted)} links from {title}: "
            f"{'Adolf_Hitler IN RESULT' if 'Adolf_Hitler' in links_sorted else 'Adolf_Hitler NOT FOUND'}")

    return links_sorted


async def verify_path(session: aiohttp.ClientSession, path: list[str]) -> bool:
    """Re-verify each hop as a real visible link."""
    for src, dst in zip(path, path[1:]):
        out = await fetch_visible_links(session, src)
        if dst not in out:
            print(f"[VERIFY] Missing edge {src} -> {dst}")
            return False
    return True


# ------------------------------------------------------
# Bidirectional BFS
# ------------------------------------------------------

async def bidirectional_bfs(start, target, visualize=False, max_depth=6):
    """Perform bidirectional BFS with visible link fetching and hop verification."""
    if start == target:
        return [start]

    front_start = {start: [start]}
    front_target = {target: [target]}
    visited_start = {start}
    visited_target = {target}

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

            # Expand from start
            new_front = {}
            tasks = [asyncio.create_task(fetch_visible_links(session, n)) for n in front_start]
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
                            if await verify_path(session, full):
                                if visualize:
                                    pos = nx.spring_layout(G, pos=pos, fixed=list(pos.keys()), k=0.8)
                                    visualize_graph(G, link, target, pos)
                                    plt.ioff()
                                    plt.show()
                                return full
                            else:
                                print("[INFO] Discarded false path after verification; continuing.")
                                if visualize:
                                    pos = nx.spring_layout(G, pos=pos, fixed=list(pos.keys()), k=0.8)
                                    visualize_graph(G, link, target, pos)

            front_start = new_front

            # Expand from target
            new_front = {}
            tasks = [asyncio.create_task(fetch_visible_links(session, n)) for n in front_target]
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
                            if await verify_path(session, full):
                                if visualize:
                                    pos = nx.spring_layout(G, pos=pos, fixed=list(pos.keys()), k=0.8)
                                    visualize_graph(G, link, target, pos)
                                    plt.ioff()
                                    plt.show()
                                return full
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

    visualize = os.getenv("WIKIBENCH_VIZ", "0") == "1"

    print(f"[INFO] Searching path from '{start}' to '{target}' (max depth {max_depth})")
    print(f"[INFO] Viz={'ON' if visualize else 'OFF'} | MAX_RPS={MAX_RPS} | CONC={MAX_CONCURRENCY}")
    start_time = time.time()
    path = asyncio.run(bidirectional_bfs(start, target, visualize=visualize, max_depth=max_depth))
    elapsed = time.time() - start_time

    if path:
        print(f"[SUCCESS] Path found in {len(path)-1} hops ({elapsed:.2f}s):")
        for step in path:
            print(f" → {step}")
        print(json.dumps(path, indent=2))
    else:
        print(f"[FAILURE] No path found within depth {max_depth} ({elapsed:.2f}s)")
