#!/usr/bin/env python3
"""
wikibench_linkextractor.py
-----------------------------------
Shared extraction + fetching utilities for WikiBench.

Provides:

✔ fetch_article_html(title)
✔ extract_text_with_links(html)
✔ extract_link_titles(html)

This keeps extract_text_with_links.py and wikibench_pathfinder_async.py
in perfect sync.
"""

import re
import html
import os
import requests
import urllib.parse
from bs4 import BeautifulSoup, Comment

# ------------------------------------------------------
# HTTP constants
# ------------------------------------------------------

API_URL = "https://en.wikipedia.org/w/api.php"
UA = "WikiBenchBot/0.1 (https://github.com/soyuz43/wikibench; contact: kebekad673@proton.me)"


def debug(msg: str):
    if os.getenv("WIKIBENCH_DEBUG", "0").lower() in ("1", "true", "yes"):
        print(f"[DEBUG] {msg}")


# ------------------------------------------------------
# Shared section constants
# ------------------------------------------------------

STRIP_SECTIONS = {
    "see also",
    "references",
    "external links",
    "notes",
    "further reading",
}

STOP_H2_SECTIONS = {
    "See also",
    "References",
    "External links",
    "Notes",
    "Further reading",
}

DISALLOWED_ANCESTORS = {
    "navbox", "vertical-navbox", "sidebar", "infobox", "toc",
    "mw-references-wrap", "reflist", "hatnote", "thumb",
    "gallery", "metadata", "mbox-small", "sistersitebox"
}

BAD_PREFIXES = (
    "Special:", "File:", "Help:", "Category:", "Template:", "Template_talk:",
    "Talk:", "Wikipedia:", "Portal:", "Draft:", "Module:", "MediaWiki:",
)


# ------------------------------------------------------
# Wikipedia fetching
# ------------------------------------------------------

def fetch_article_html(title: str) -> str:
    """
    Fetch the rendered HTML fragment for a Wikipedia article.
    This is shared by the CLI extractor and the async pathfinder.
    """
    params = {
        "action": "parse",
        "page": title,
        "format": "json",
        "prop": "text",
        "redirects": 1
    }

    headers = {
        "User-Agent": UA,
        "Accept-Encoding": "gzip"
    }

    debug(f"Requesting Wikipedia for '{title}'")
    r = requests.get(API_URL, params=params, headers=headers, timeout=20)
    debug(f"status: {r.status_code}")

    r.raise_for_status()

    try:
        data = r.json()
    except Exception:
        debug("JSON decode failed. Response snippet:")
        debug(r.text[:2000])
        raise RuntimeError("Failed to decode JSON from Wikipedia API.")

    if "error" in data:
        raise RuntimeError(f"API error: {data['error']}")

    return data["parse"]["text"]["*"]


# ------------------------------------------------------
# Title normalization
# ------------------------------------------------------

def normalize_wiki_title(href: str) -> str | None:
    if not href or not href.startswith("/wiki/"):
        return None

    tail = href.split("/wiki/", 1)[1]
    tail = tail.split("#", 1)[0].split("?", 1)[0]
    if not tail:
        return None

    if any(tail.startswith(ns) for ns in BAD_PREFIXES):
        return None

    decoded = urllib.parse.unquote(tail)
    return decoded.replace("_", " ")


def normalize_wiki_title_preserve_underscore(href: str) -> str | None:
    if not href or not href.startswith("/wiki/"):
        return None

    raw = href.split("/wiki/", 1)[1]
    raw = raw.split("#", 1)[0].split("?", 1)[0]
    if not raw:
        return None

    if any(raw.startswith(ns) for ns in BAD_PREFIXES):
        return None

    return urllib.parse.unquote(raw)


# ------------------------------------------------------
# HTML stripping utilities
# ------------------------------------------------------

def remove_sections_by_heading(content: BeautifulSoup) -> None:
    for h2 in list(content.find_all("h2")):
        heading_txt = h2.get_text(" ", strip=True).lower()
        heading_txt = re.sub(r"\s*\[edit\]\s*$", "", heading_txt)
        if heading_txt in STRIP_SECTIONS:
            node = h2
            while node:
                nxt = node.find_next_sibling()
                node.decompose()
                if not nxt or nxt.name == "h2":
                    break


def has_disallowed_ancestor(a_tag) -> bool:
    for parent in a_tag.parents:
        cls = parent.get("class") or []
        if any(c in DISALLOWED_ANCESTORS for c in cls):
            return True
        if parent.get("role") == "navigation":
            return True
        if parent.get("class") and "mw-parser-output" in parent.get("class", []):
            break
    return False


# ------------------------------------------------------
# Main extraction: <link>label</link>
# ------------------------------------------------------

def extract_text_with_links(html_snippet: str) -> str:
    soup = BeautifulSoup(html_snippet, "html.parser")
    content = soup.find("div", class_="mw-parser-output") or soup

    for c in content.find_all(string=lambda t: isinstance(t, Comment)):
        c.extract()

    drop_selectors = [
        "table", ".infobox", ".navbox", ".metadata", ".ambox", ".hatnote", ".toc",
        ".reflist", ".reference", ".mw-editsection", ".thumb", "figure",
        "style", "script", "noscript", ".shortdescription", ".mw-empty-elt",
    ]
    for sel in drop_selectors:
        for el in content.select(sel):
            el.decompose()

    remove_sections_by_heading(content)

    # Rewrite anchors → <link>
    for a in content.find_all("a"):
        href = a.get("href", "")
        title = normalize_wiki_title(href)
        if title:
            label = a.get_text(separator="", strip=False)
            new = soup.new_tag("link")
            new["title"] = title
            new.string = label
            a.replace_with(new)
        else:
            a.unwrap()

    # Remove cites
    for sup in content.find_all("sup"):
        sup.decompose()

    # Unwrap all non-link tags
    for tag in list(content.find_all()):
        if tag.name != "link":
            tag.unwrap()

    out = html.unescape(str(content))
    out = re.sub(r"[ \t]+\n", "\n", out)
    out = re.sub(r"\n{3,}", "\n\n", out).strip()

    return out


# ------------------------------------------------------
# Graph-mode visible link extraction
# ------------------------------------------------------

def extract_link_titles(html_page: str) -> list[str]:
    soup = BeautifulSoup(html_page, "html.parser")

    content = soup.find("div", {"id": "mw-content-text"})
    if content:
        content = content.find("div", class_="mw-parser-output") or content
    if not content:
        return []

    found: set[str] = set()
    stop = False

    for node in content.descendants:
        if stop:
            break

        if getattr(node, "name", None) == "h2":
            htxt = node.get_text(" ", strip=True)
            if any(s in htxt for s in STOP_H2_SECTIONS):
                stop = True
                continue

        if getattr(node, "name", None) not in ("p", "ul", "ol"):
            continue

        for a in node.find_all("a", href=True):
            if has_disallowed_ancestor(a):
                continue
            title = normalize_wiki_title_preserve_underscore(a["href"])
            if title:
                found.add(title)

    return sorted(found)
