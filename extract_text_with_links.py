#!/usr/bin/env python3
"""
extract_text_with_links.py
-----------------------------------
Thin CLI wrapper around the shared wikibench_linkextractor module.

This script now simply:
 - fetches the article HTML using the extractor module
 - injects <link title="..."> elements
 - prints the cleaned HTML for the harness

This preserves full backwards compatibility with older components
that still shell out to this script.
"""

import sys
import os
import argparse

from wikibench_linkextractor import (
    fetch_article_html,
    extract_text_with_links
)

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
# CLI logic
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Extract clean HTML text with <link> tags from a Wikipedia article "
                    "using the shared wikibench_linkextractor module."
    )
    parser.add_argument("title", help="Wikipedia article title")
    parser.add_argument("-o", "--out", metavar="FILE",
        help="Write output to FILE instead of stdout")
    args = parser.parse_args()

    title = args.title.replace(" ", "_")
    info(f"Fetching + cleaning article: {title}")

    try:
        # 1. Fetch HTML via the shared extractor
        debug("Fetching raw Wikipedia HTML…")
        html = fetch_article_html(title)

        # 2. Clean + inject <link> tags
        debug("Injecting <link> tags from visible DOM links…")
        cleaned_html = extract_text_with_links(html)

    except Exception as e:
        error(f"Failed to extract article: {e}")
        sys.exit(1)

    # Output
    if args.out:
        try:
            with open(args.out, "w", encoding="utf-8") as f:
                f.write(cleaned_html)
            info(f"Wrote cleaned output to {args.out}")
        except Exception as e:
            error(f"Could not write to output file: {e}")
            sys.exit(1)
    else:
        print(cleaned_html)


if __name__ == "__main__":
    main()
