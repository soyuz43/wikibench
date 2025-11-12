#!/usr/bin/env python3
import sys
import re
import html
import os
import argparse
import urllib.parse
import requests
from bs4 import BeautifulSoup, Comment

API_URL = "https://en.wikipedia.org/w/api.php"
UA = "WikiBenchBot/0.1 (https://github.com/soyuz43/wikibench; contact: kebekad673@proton.me)"

STRIP_SECTIONS = {
    "see also",
    "references",
    "external links",
    "notes",
    "further reading",
}


# ----------------------------------------------------------------------
# Logging utilities
# ----------------------------------------------------------------------
def info(msg: str):
    print(f"[INFO] {msg}")

def debug(msg: str):
    if os.getenv("WIKIBENCH_DEBUG", "0") in ("1", "true", "True"):
        print(f"[DEBUG] {msg}")

def error(msg: str):
    print(f"[ERROR] {msg}", file=sys.stderr)


# ----------------------------------------------------------------------
# Core WikiBench Extractor Logic
# ----------------------------------------------------------------------
def normalize_wiki_title(href: str) -> str | None:
    """Convert /wiki/Foo_Bar#Section → 'Foo Bar'; drop non-article namespaces."""
    if not href or not href.startswith("/wiki/"):
        return None
    tail = href.split("/wiki/", 1)[1]
    tail = tail.split("#", 1)[0].split("?", 1)[0]
    bad_ns = (
        "Special:", "File:", "Help:", "Category:", "Template:", "Template_talk:",
        "Talk:", "Wikipedia:", "Portal:", "Draft:", "Module:", "MediaWiki:",
    )
    if any(tail.startswith(ns) for ns in bad_ns):
        return None
    return urllib.parse.unquote(tail).replace("_", " ")


def get_article_html(title: str) -> str:
    """Fetch rendered HTML snippet of the article body via action=parse."""
    params = {"action": "parse", "page": title, "format": "json", "prop": "text", "redirects": 1}
    headers = {"User-Agent": UA, "Accept-Encoding": "gzip"}
    debug(f"Requesting {API_URL} for '{title}'")
    r = requests.get(API_URL, params=params, headers=headers, timeout=20)
    debug(f"status: {r.status_code}")
    r.raise_for_status()
    try:
        data = r.json()
    except Exception:
        error("Failed to decode JSON or parse article.")
        debug(f"Response text: {r.text[:2000]}")
        raise
    if "error" in data:
        raise RuntimeError(f"API error: {data['error']}")
    return data["parse"]["text"]["*"]


def remove_sections_by_heading(content: BeautifulSoup) -> None:
    """Remove entire sections starting at <h2> whose text matches STRIP_SECTIONS."""
    for h2 in list(content.find_all("h2")):
        heading_txt = h2.get_text(" ", strip=True).lower()
        heading_txt = re.sub(r"\s*\[edit\]\s*$", "", heading_txt).strip()
        if heading_txt in STRIP_SECTIONS:
            node = h2
            while node:
                nxt = node.find_next_sibling()
                node.decompose()
                if not nxt or nxt.name == "h2":
                    break


def extract_text_with_links(html_snippet: str) -> str:
    """Return simplified HTML with <link title="...">text</link> markup only."""
    soup = BeautifulSoup(html_snippet, "html.parser")
    content = soup.find("div", class_="mw-parser-output") or soup

    # 1) Drop comments
    for c in content.find_all(string=lambda t: isinstance(t, Comment)):
        c.extract()

    # 2) Remove non-prose elements
    drop_selectors = [
        "table", ".infobox", ".navbox", ".metadata", ".ambox", ".hatnote", ".toc",
        ".reflist", ".reference", ".mw-editsection", ".thumb", "figure",
        "style", "script", "noscript", ".shortdescription", ".mw-empty-elt",
    ]
    for sel in drop_selectors:
        for el in content.select(sel):
            el.decompose()

    # 3) Remove non-content sections
    remove_sections_by_heading(content)

    # 4) Rewrite anchors → <link>
    for a in content.find_all("a"):
        href = a.get("href", "")
        title = normalize_wiki_title(href)
        if title:
            label = a.get_text(separator="", strip=False)
            new_tag = soup.new_tag("link")
            new_tag["title"] = title
            new_tag.string = label
            a.replace_with(new_tag)
        else:
            a.unwrap()

    # 5) Remove superscript cites
    for sup in content.find_all("sup"):
        sup.decompose()

    # 6) Unwrap everything except <link>
    for tag in list(content.find_all()):
        if tag.name not in ("link",):
            tag.unwrap()

    # 7) Serialize and normalize whitespace
    out_html = html.unescape(str(content))
    out_html = re.sub(r"[ \t]+\n", "\n", out_html)
    out_html = re.sub(r"\n{3,}", "\n\n", out_html).strip()

    return out_html


# ----------------------------------------------------------------------
# CLI entrypoint
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Extract clean HTML text with <link> tags from a Wikipedia article."
    )
    parser.add_argument("title", help="Wikipedia article title (underscores optional)")
    parser.add_argument("-o", "--out", metavar="FILE", help="Write output to FILE instead of stdout")
    args = parser.parse_args()

    title = args.title.replace(" ", "_")
    info(f"Fetching and cleaning article: {title}")

    try:
        html_snippet = get_article_html(title)
        cleaned = extract_text_with_links(html_snippet)
    except Exception as e:
        error(f"Failed to extract article: {e}")
        sys.exit(1)

    if args.out:
        try:
            with open(args.out, "w", encoding="utf-8") as f:
                f.write(cleaned)
            info(f"Wrote cleaned output to {args.out}")
        except Exception as e:
            error(f"Could not write to output file: {e}")
            sys.exit(1)
    else:
        print(cleaned)


if __name__ == "__main__":
    main()
