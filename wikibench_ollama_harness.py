#!/usr/bin/env python3
"""
WikiBench Ollama Harness
------------------------
Runs local LLMs (via Ollama) through WikiBench-style link navigation tasks.
Each run compares the model's navigation choices against the ground-truth
shortest path (optional) and records a detailed trace.

Usage:
  # Pick a model interactively from Ollama's installed models
  ./wikibench_ollama_harness.py \
    --pick-model \
    --start "UPS Airlines Flight 2976" \
    --target "Adolf Hitler" \
    --compare

  # Or specify a model explicitly
  ./wikibench_ollama_harness.py \
    --model llama3:8b \
    --start "UPS Airlines Flight 2976" \
    --target "Adolf Hitler"

Notes:
- The harness loads environment variables from a .env file if present.
- The default prompt template is resolved relative to this script at:
    prompts/prompt.yml
  You can override with --prompt or the env var WIKIBENCH_PROMPT_PATH.
- Ollama endpoint selection is handled by $OLLAMA_HOST; the harness
  uses that for /api/tags and passes it through to the `ollama` CLI.
"""

import os
import re
import sys
import json
import time
import argparse
import subprocess
import asyncio
import urllib.request
import urllib.error
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import yaml
from dotenv import load_dotenv

# Local imports
from wikibench_pathfinder_async import bidirectional_bfs

# ---------------------------------------------------------------------------
# Init
# ---------------------------------------------------------------------------

load_dotenv()
SCRIPT_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def info(msg: str):   print(f"[INFO] {msg}")
def warn(msg: str):   print(f"[WARN] {msg}")
def error(msg: str):  print(f"[ERROR] {msg}")
def debug(msg: str):
    if os.getenv("WIKIBENCH_DEBUG") == "1":
        print(f"[DEBUG] {msg}")

# ---------------------------------------------------------------------------
# Ollama model discovery (/api/tags + fallback)
# ---------------------------------------------------------------------------

def ensure_http_base(host: Optional[str]) -> str:
    """
    Normalize OLLAMA_HOST to an HTTP base like: http://127.0.0.1:11434
    """
    default = "http://127.0.0.1:11434"
    if not host:
        return default
    h = host.strip()
    if not h.startswith("http://") and not h.startswith("https://"):
        h = f"http://{h}"
    return h

def fetch_ollama_tags_via_http(base_url: str, timeout: float = 3.0) -> Optional[Dict[str, Any]]:
    """
    Returns JSON dict from /api/tags or None on failure.
    """
    url = base_url.rstrip("/") + "/api/tags"
    debug(f"Fetching Ollama tags via HTTP: {url}")
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status != 200:
                warn(f"Ollama /api/tags returned HTTP {resp.status}")
                return None
            data = resp.read().decode("utf-8", errors="ignore")
            return json.loads(data)
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
        warn(f"HTTP fetch to {url} failed: {e}")
        return None
    except Exception as e:
        warn(f"Unexpected error fetching tags via HTTP: {e}")
        return None

def fallback_list_models_via_cli() -> List[str]:
    """
    Fallback: parse 'ollama list' plain text. We take the first column per row.
    """
    try:
        proc = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            env=os.environ.copy(),
        )
        out = (proc.stdout or "").strip()
        if not out:
            return []
        lines = [ln for ln in out.splitlines() if ln.strip()]
        # Skip potential header; extract first token per line
        models = []
        for ln in lines:
            tok = ln.split()[0]
            if tok and ":" in tok:  # heuristic
                models.append(tok)
        return models
    except Exception as e:
        warn(f"Fallback 'ollama list' failed: {e}")
        return []

def human_bytes(n: Optional[int]) -> str:
    if n is None:
        return "?"
    step = 1024.0
    units = ["B", "KB", "MB", "GB", "TB"]
    s = float(n)
    for u in units:
        if s < step:
            return f"{s:.1f} {u}"
        s /= step
    return f"{s:.1f} PB"

def discover_installed_models() -> List[Dict[str, Any]]:
    """
    Returns a list of dicts with at least: name, size_bytes?, family?, param_size?, quant?
    Tries /api/tags then falls back to `ollama list`.
    """
    base = ensure_http_base(os.getenv("OLLAMA_HOST"))
    payload = fetch_ollama_tags_via_http(base)
    if payload and isinstance(payload, dict) and "models" in payload:
        models = []
        for m in payload["models"]:
            name = m.get("name") or ""
            details = m.get("details") or {}
            models.append({
                "name": name,
                "size_bytes": m.get("size"),
                "family": details.get("family"),
                "param_size": details.get("parameter_size"),
                "quant": details.get("quantization_level"),
                "modified_at": m.get("modified_at"),
                "digest": m.get("digest"),
            })
        if models:
            return models

    # Fallback to CLI
    names = fallback_list_models_via_cli()
    return [{"name": n, "size_bytes": None, "family": None, "param_size": None, "quant": None} for n in names]

def choose_model_interactively(models: List[Dict[str, Any]], page_size: int = 20) -> Optional[str]:
    if not models:
        error("No Ollama models found. Use `ollama pull <model>` to install one.")
        return None

    info(f"Discovered {len(models)} model(s). Use number to select, 'n' next page, 'p' previous, 'q' to quit.")
    page = 0
    total_pages = max(1, (len(models) + page_size - 1) // page_size)

    while True:
        start = page * page_size
        end = min(len(models), start + page_size)
        print()
        print(f"[INFO] Models {start+1}-{end} of {len(models)} (page {page+1}/{total_pages})")
        for idx, m in enumerate(models[start:end], start=start + 1):
            size_s = human_bytes(m.get("size_bytes"))
            fam = m.get("family") or "-"
            quant = m.get("quant") or "-"
            psize = m.get("param_size") or "-"
            print(f"  [{idx:>3}] {m['name']:<32} | {size_s:>8} | family={fam} | params={psize} | quant={quant}")

        choice = input("\nSelect model #: ").strip().lower()
        if choice == "n":
            if page + 1 < total_pages:
                page += 1
            else:
                info("Already at last page.")
            continue
        if choice == "p":
            if page > 0:
                page -= 1
            else:
                info("Already at first page.")
            continue
        if choice == "q":
            return None

        if choice.isdigit():
            i = int(choice)
            if 1 <= i <= len(models):
                info(f"Selected model: {models[i-1]['name']}")
                return models[i - 1]["name"]
            else:
                warn("Invalid number.")
        else:
            warn("Enter a number, or 'n'/'p'/'q'.")

# ---------------------------------------------------------------------------
# Prompt template handling
# ---------------------------------------------------------------------------

def resolve_prompt_path(cli_path: Optional[str]) -> Path:
    """
    Resolution order:
      1) --prompt <path>
      2) $WIKIBENCH_PROMPT_PATH
      3) <repo>/prompts/prompt.yml (default)
    """
    if cli_path:
        p = Path(cli_path).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"Prompt file not found: {p}")
        return p

    env_path = os.getenv("WIKIBENCH_PROMPT_PATH")
    if env_path:
        p = Path(env_path).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"WIKIBENCH_PROMPT_PATH points to missing file: {p}")
        return p

    default = SCRIPT_DIR / "prompts" / "prompt.yml"
    if not default.exists():
        raise FileNotFoundError(
            f"Default prompt not found at {default}. "
            f"Pass --prompt or set $WIKIBENCH_PROMPT_PATH."
        )
    return default

def load_prompt_template(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def render_prompt(template: Dict[str, Any],
                  article_title: str,
                  target_title: str,
                  html_snippet: str,
                  visited: List[str]) -> str:
    """
    We keep the template as YAML and do string replacement for placeholders embedded in scalars.
    Placeholders:
      - {{TARGET_TITLE}}
      - {{VISITED_JSON}}
      - {{ARTICLE_HTML}}
    """
    yaml_text = yaml.dump(template, sort_keys=False, allow_unicode=True, width=100000)
    yaml_text = yaml_text.replace("{{TARGET_TITLE}}", target_title)
    yaml_text = yaml_text.replace("{{VISITED_JSON}}", json.dumps(visited, ensure_ascii=False))
    yaml_text = yaml_text.replace("{{ARTICLE_HTML}}", html_snippet)
    # Add a tiny header so it's clear to the model what follows
    return f"# WikiBench Prompt\n{yaml_text}"

# ---------------------------------------------------------------------------
# HTML link harvesting (from extract_text_with_links output)
# ---------------------------------------------------------------------------

_LINK_TITLE_RE = re.compile(r'<link\b[^>]*\btitle="([^"]+)"', re.IGNORECASE)

def normalize_title(t: str) -> str:
    return t.strip().replace(" ", "_")

def title_key_for_compare(t: str) -> str:
    # Compare in a normalized, case-insensitive underscore form
    return normalize_title(t).lower()

def extract_link_titles(html_snippet: str) -> Tuple[List[str], set]:
    """
    Returns:
      - ordered list of raw titles found (as in the HTML)
      - normalized set for fast membership checks
    """
    titles = _LINK_TITLE_RE.findall(html_snippet or "")
    normalized_set = {title_key_for_compare(t) for t in titles}
    return titles, normalized_set

# ---------------------------------------------------------------------------
# Run extract_text_with_links.py as a subprocess to get HTML snippet
# ---------------------------------------------------------------------------

def get_article_html(title: str) -> str:
    """
    Calls ./extract_text_with_links.py <title> and returns its stdout.
    Falls back to `python3 extract_text_with_links.py` if direct exec fails.
    Optionally trims output to the main <div ... mw-parser-output ...> block.
    """
    script = SCRIPT_DIR / "extract_text_with_links.py"
    cmds = [
        [str(script), title],
        [sys.executable or "python3", str(script), title],
        ["python3", str(script), title],
    ]
    last_err = None
    for cmd in cmds:
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                env=os.environ.copy(),
                check=True,
            )
            out = (proc.stdout or "").strip()
            if not out:
                continue
            # Try to keep only the main content div to reduce noise in prompts
            m = re.search(r'(<div[^>]*mw-parser-output[^>]*>.*?</div>)', out, flags=re.IGNORECASE | re.DOTALL)
            return m.group(1) if m else out
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"Failed to run extract_text_with_links.py: {last_err}")

# ---------------------------------------------------------------------------
# Ollama invocation
# ---------------------------------------------------------------------------

def run_ollama_prompt(model: str, prompt: str) -> str:
    """
    Invoke Ollama CLI and return the model's raw text response.
    We pass through the current environment so OLLAMA_HOST is honored.
    """
    env = os.environ.copy()
    proc = subprocess.run(
        ["ollama", "run", model],
        input=prompt.encode("utf-8"),
        capture_output=True,
        env=env,
    )
    out = proc.stdout.decode("utf-8", errors="ignore")
    err = proc.stderr.decode("utf-8", errors="ignore").strip()
    if err:
        debug(f"Ollama stderr: {err}")
    return out.strip()

# ---------------------------------------------------------------------------
# Choice parsing and validation
# ---------------------------------------------------------------------------

def first_nonempty_line(s: str) -> str:
    for line in (s or "").splitlines():
        line = line.strip()
        if line:
            return line
    return ""

def sanitize_model_choice(raw: str) -> str:
    """
    Accepts raw model output; returns a single candidate title or "STOP".
    Strips bullets, backticks, quotes, arrows.
    """
    line = first_nonempty_line(raw)
    if not line:
        return ""

    # Strip common wrappers
    line = line.strip("`")
    if line.startswith(("```", "~~~")):
        parts = line.split()
        if len(parts) > 1:
            line = parts[1]
        else:
            line = ""

    # Remove bullets/arrows
    line = line.lstrip("-*•>→").strip()

    if line.upper() == "STOP":
        return "STOP"

    # Remove surrounding quotes
    if (line.startswith('"') and line.endswith('"')) or (line.startswith("'") and line.endswith("'")):
        line = line[1:-1].strip()

    return normalize_title(line)

def validate_choice(choice: str,
                    available_norm: set,
                    visited_norm: set,
                    target_norm: str) -> Tuple[bool, str]:
    """
    Returns (is_valid, reason_if_invalid).
    """
    if not choice:
        return False, "empty_output"
    if choice == "STOP":
        return False, "stop"
    if choice in visited_norm:
        return False, "repeated"
    if choice not in available_norm and choice != target_norm:
        return False, "not_in_available_links"
    return True, ""

# ---------------------------------------------------------------------------
# Navigation loop
# ---------------------------------------------------------------------------

def run_navigation(model: str,
                   start_title: str,
                   target_title: str,
                   max_steps: int,
                   max_retries: int,
                   prompt_template: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Executes step-by-step navigation until success/stop/failure/max_steps.
    Returns (trace_dict, summary_dict).
    """

    current = normalize_title(start_title)
    target = normalize_title(target_title)
    visited: List[str] = [current]
    visited_norm = {title_key_for_compare(current)}

    info(f"Running {model} on: {current} → {target}")

    trace: List[Dict[str, Any]] = []
    status = "failed"
    reached_target = False

    for step in range(1, max_steps + 1):
        html_snippet = get_article_html(current)
        titles_raw, available_norm = extract_link_titles(html_snippet)

        prompt = render_prompt(prompt_template, current, target, html_snippet, visited)

        attempt = 0
        choice_final = ""
        raw_history: List[str] = []
        reason_if_invalid = ""
        valid = False

        while attempt <= max_retries:
            attempt += 1

            raw = run_ollama_prompt(model, prompt)
            raw_history.append(raw)
            choice = sanitize_model_choice(raw)
            choice_key = title_key_for_compare(choice)

            is_valid, reason = validate_choice(
                choice_key, available_norm, visited_norm, title_key_for_compare(target)
            )

            if is_valid:
                choice_final = choice
                valid = True
                break

            if reason == "stop":
                status = "stopped"
                reason_if_invalid = reason
                break

            reason_if_invalid = reason
            warn(f"Invalid selection (attempt {attempt}/{max_retries}): {choice or '<empty>'} — {reason}")
            if attempt <= max_retries:
                feedback = (
                    "\n# FEEDBACK:\n"
                    f"The previous selection \"{choice or '<empty>'}\" was invalid ({reason}). "
                    "Choose exactly ONE title that appears in <link title=\"...\"> elements, "
                    "or output STOP if none remain. Output a SINGLE line ONLY.\n"
                )
                prompt = prompt + feedback

        trace.append({
            "step": step,
            "current": current,
            "target": target,
            "available_links_count": len(titles_raw),
            "model_raw_attempts": raw_history,
            "chosen": choice_final if valid else None,
            "valid": valid,
            "invalid_reason": (None if valid else reason_if_invalid),
            "retries_used": (attempt - 1),
        })

        if not valid:
            if reason_if_invalid == "stop":
                info("Model requested STOP.")
                status = "stopped"
            else:
                error("Model failed to produce a valid next link within retry budget.")
                status = "failed"
            break

        next_title = choice_final
        if title_key_for_compare(next_title) == title_key_for_compare(target):
            visited.append(next_title)
            reached_target = True
            status = "success"
            info("Target reached!")
            break

        if title_key_for_compare(next_title) in visited_norm:
            warn(f"Model repeated a title: {next_title}. Stopping.")
            status = "failed"
            break

        visited.append(next_title)
        visited_norm.add(title_key_for_compare(next_title))
        current = next_title

    else:
        status = "max_steps_reached"

    summary = {
        "model": model,
        "start": start_title,
        "target": target_title,
        "max_steps": max_steps,
        "max_retries": max_retries,
        "status": status,
        "visited": visited,
        "hops": max(0, len(visited) - 1),
        "reached_target": reached_target,
    }
    return {"trace": trace}, summary

# ---------------------------------------------------------------------------
# Optimal path comparison
# ---------------------------------------------------------------------------

def compute_optimal_path(start_title: str, target_title: str, max_depth: int = 6) -> Dict[str, Any]:
    """
    Calls bidirectional_bfs to find the optimal path (if any).
    """
    start = normalize_title(start_title)
    target = normalize_title(target_title)
    info(f"Computing optimal path for: {start} → {target}")
    t0 = time.time()
    path, meeting_node, parent_start, parent_target = asyncio.run(
        bidirectional_bfs(start, target, visualize=False, max_depth=max_depth)
    )
    dt = time.time() - t0
    if path:
        return {"path": path, "hops": len(path) - 1, "elapsed_sec": round(dt, 2)}
    return {"path": None, "hops": None, "elapsed_sec": round(dt, 2)}

# ---------------------------------------------------------------------------
# Persist results
# ---------------------------------------------------------------------------

def save_results(model: str,
                 start: str,
                 target: str,
                 prompt_path: Path,
                 trace: Dict[str, Any],
                 summary: Dict[str, Any],
                 comparison: Optional[Dict[str, Any]],
                 out_format: str = "json") -> Path:
    out_dir = SCRIPT_DIR / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    base = f"{ts}_{model.replace(':','_')}_{normalize_title(start)}_to_{normalize_title(target)}"

    payload = {
        "meta": {
            "prompt_template_path": str(prompt_path),
            "ollama_host": os.getenv("OLLAMA_HOST", "default"),
        },
        "summary": summary,
        "trace": trace["trace"],
    }

    if comparison is not None:
        payload["comparison"] = comparison

    out_path = out_dir / f"{base}.{ 'yaml' if out_format == 'yaml' else 'json'}"
    if out_format == "yaml":
        with open(out_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=True)
    else:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    info(f"Result saved to {out_path}")
    return out_path

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run WikiBench benchmark with Ollama")
    parser.add_argument("--model", help="Ollama model name (e.g., llama3:8b). Omit to pick interactively.")
    parser.add_argument("--pick-model", action="store_true",
                        help="Interactively choose an installed Ollama model via /api/tags or `ollama list`.")
    parser.add_argument("--start", required=True, help="Starting article title")
    parser.add_argument("--target", required=True, help="Target article title")
    parser.add_argument("--max-steps", type=int, default=10, help="Maximum navigation steps")
    parser.add_argument("--max-retries", type=int, default=2, help="Retries per step for invalid output")
    parser.add_argument("--compare", action="store_true", help="Compute optimal path for comparison (slow)")
    parser.add_argument("--prompt", help="Path to YAML prompt template (overrides env/default)")
    parser.add_argument("--out-format", choices=["json", "yaml"], default="json", help="Results file format")
    args = parser.parse_args()

    # Resolve model selection
    model = args.model
    if args.pick_model or not model:
        models = discover_installed_models()
        chosen = choose_model_interactively(models)
        if not chosen:
            error("No model selected. Exiting.")
            return
        model = chosen

    prompt_path = resolve_prompt_path(args.prompt)
    prompt_template = load_prompt_template(prompt_path)

    trace, summary = run_navigation(
        model=model,
        start_title=args.start,
        target_title=args.target,
        max_steps=args.max_steps,
        max_retries=args.max_retries,
        prompt_template=prompt_template,
    )

    comparison = compute_optimal_path(args.start, args.target) if args.compare else None

    save_results(
        model=model,
        start=args.start,
        target=args.target,
        prompt_path=prompt_path,
        trace=trace,
        summary=summary,
        comparison=comparison,
        out_format=args.out_format,
    )

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        warn("Interrupted by user.")
