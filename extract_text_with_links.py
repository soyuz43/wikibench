#!/usr/bin/env python3
import sys
import re
import html
import urllib.parse
import requests
from bs4 import BeautifulSoup, Comment

API_URL = "https://en.wikipedia.org/w/api.php"
UA = "WikiBenchBot/0.1 (https://github.com/soyuz43/wikibench; contact: kebekad673@proton.me)"

# Headings whose entire sections should be removed (case-insensitive, trimmed)
STRIP_SECTIONS = {
    "see also",
    "references",
    "external links",
    "notes",
    "further reading",
}

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
    r = requests.get(API_URL, params=params, headers=headers, timeout=20)
    print(f"[DEBUG] status: {r.status_code}")
    r.raise_for_status()
    try:
        data = r.json()
    except Exception:
        print("Failed to decode JSON or parse article.\nResponse text was:\n", r.text[:2000])
        raise
    if "error" in data:
        raise RuntimeError(f"API error: {data['error']}")
    return data["parse"]["text"]["*"]

def remove_sections_by_heading(content: BeautifulSoup) -> None:
    """
    Remove entire sections starting at <h2> whose text matches STRIP_SECTIONS,
    up to (but not including) the next <h2>.
    """
    for h2 in list(content.find_all("h2")):
        # <h2><span class="mw-headline">Text</span></h2> → pick visible text
        heading_txt = h2.get_text(" ", strip=True).lower()
        # strip trailing 'edit' etc.
        heading_txt = re.sub(r"\s*\[edit\]\s*$", "", heading_txt).strip()
        if heading_txt in STRIP_SECTIONS:
            # delete h2 and everything until the next h2
            node = h2
            while node:
                nxt = node.find_next_sibling()
                node.decompose()
                if not nxt or nxt.name == "h2":
                    break
                node = nxt

def extract_text_with_links(html_snippet: str) -> str:
    """
    Return HTML like:
    <div class="mw-content-ltr mw-parser-output" ...>
    ...prose with <link title="...">text</link>...
    </div>
    All non-prose scaffolding removed; sections in STRIP_SECTIONS removed.
    """
    soup = BeautifulSoup(html_snippet, "html.parser")
    content = soup.find("div", class_="mw-parser-output") or soup

    # 1) Drop HTML comments (API stats, expansion reports, etc.)
    for c in content.find_all(string=lambda t: isinstance(t, Comment)):
        c.extract()

    # 2) Remove non-prose / chrome
    drop_selectors = [
        "table",             # infoboxes, navboxes, data tables
        ".infobox",
        ".navbox",
        ".metadata",
        ".ambox",
        ".hatnote",
        ".toc",
        ".reflist",
        ".reference",
        ".mw-editsection",
        ".thumb",
        "figure",
        "style",
        "script",
        "noscript",
        ".shortdescription",
        ".mw-empty-elt",
    ]
    for sel in drop_selectors:
        for el in content.select(sel):
            el.decompose()

    # 3) Remove sections by heading (“See also”, “References”, …)
    remove_sections_by_heading(content)

    # 4) Convert internal <a> → <link title="…">label</link>; unwrap other anchors
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

    # 5) Remove residual superscript cite markers
    for sup in content.find_all("sup"):
        sup.decompose()

    # 6) Unwrap everything except our outer container div and <link> tags
    for tag in list(content.find_all()):
        if tag.name not in ("link",):
            tag.unwrap()

    # 7) Serialize; keep container div (content is that div) and unescape entities
    out_html = str(content)
    out_html = html.unescape(out_html)

    # 8) Normalize whitespace a bit (optional)
    out_html = re.sub(r"[ \t]+\n", "\n", out_html)
    out_html = re.sub(r"\n{3,}", "\n\n", out_html).strip()

    return out_html

def main():
    title = sys.argv[1] if len(sys.argv) > 1 else "UPS_Airlines_Flight_2976"
    html_snippet = get_article_html(title)
    cleaned = extract_text_with_links(html_snippet)
    print(cleaned)

if __name__ == "__main__":
    main()
