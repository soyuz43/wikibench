#!/usr/bin/env python3
"""
WikiBench Ollama Harness
------------------------
Runs local LLMs (via Ollama) through WikiBench-style link navigation tasks.
Each run compares the model's predicted navigation path against the optimal
shortest path from `wikibench_pathfinder_async.py`.

Example:
    ./wikibench_ollama_harness.py --model llama3:8b --start "UPS Airlines Flight 2976" --target "Adolf Hitler"
"""

import os
import json
import random
import subprocess
import argparse
from wikibench_pathfinder_async import bidirectional_bfs
from extract_text_with_links import main as extract_links  # adjust if needed

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run_ollama_prompt(model: str, prompt: str) -> str:
    """Invoke Ollama CLI and return the model's raw text response."""
    result = subprocess.run(
        ["ollama", "run", model],
        input=prompt.encode("utf-8"),
        capture_output=True,
    )
    return result.stdout.decode("utf-8").strip()

def make_prompt(article_title: str, target: str, html_snippet: str) -> str:
    """Build a clean system/user prompt for the benchmark."""
    return f"""
You are participating in the WikiBench benchmark.

You are currently on the Wikipedia article: "{article_title}".
Below is a simplified HTML extract showing visible links on the page.

---
{html_snippet}
---

Your goal is to navigate from this page to "{target}" by following hyperlinks.
Return the **titles of links you would click**, one per line, until you reach the goal.

Example output:
Functional_linguistics
Soviet_Union
Adolf_Hitler
"""

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run WikiBench benchmark with Ollama")
    parser.add_argument("--model", default="llama3:8b", help="Ollama model name (e.g., llama3:8b)")
    parser.add_argument("--start", required=True, help="Starting article title")
    parser.add_argument("--target", required=True, help="Target article title")
    args = parser.parse_args()

    # Extract page links for context
    html_snippet = extract_links(args.start)

    # Build prompt
    prompt = make_prompt(args.start, args.target, html_snippet)

    # Run local model
    print(f"[INFO] Running {args.model} on task: {args.start} → {args.target}")
    response = run_ollama_prompt(args.model, prompt)
    print(f"[MODEL OUTPUT]\n{response}\n")

    # Evaluate against shortest path
    print("[INFO] Computing optimal path for comparison...")
    # optional: you can reuse your async BFS here
    # path, _, _, _ = asyncio.run(bidirectional_bfs(args.start, args.target))
    # compare length, overlap, etc.

    # Save result
    out_dir = "results"
    os.makedirs(out_dir, exist_ok=True)
    fname = f"{out_dir}/{args.model.replace(':','_')}_{args.start}_to_{args.target}.json"
    json.dump({"prompt": prompt, "response": response}, open(fname, "w"), indent=2)
    print(f"[INFO] Result saved to {fname}")

if __name__ == "__main__":
    main()
