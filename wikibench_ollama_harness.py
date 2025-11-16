#!/usr/bin/env python3
"""
WikiBench Ollama Harness
------------------------
Runs local LLMs (via Ollama) through WikiBench-style link navigation tasks.
Each run compares the model's navigation choices against the ground-truth
shortest path (optional) and records a detailed trace.

Usage:
  ./wikibench_ollama_harness.py \
    --model llama3.1:latest \
    --start "UPS Airlines Flight 2976" \
    --target "Adolf Hitler" \
    --compare

Notes:
- Loads environment variables from a .env file if present.
- Default prompt template path resolves to: prompts/prompt.yml
  (override with --prompt or $WIKIBENCH_PROMPT_PATH).
- Ollama endpoint is handled by the `ollama` CLI; $OLLAMA_HOST is passed through.
- Includes an optional interactive model picker (--pick-model).
"""

import os
import re
import json
import time
import argparse
import subprocess
import asyncio
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from urllib.request import urlopen, Request
from urllib.error import URLError

import yaml
from dotenv import load_dotenv

# Local imports
from wikibench_pathfinder_async import bidirectional_bfs
# We call the extractor as a subprocess to avoid import/CLI mismatch.
# (We still import the symbol name so users see the dependency,
#  but we will not call it directly.)
from extract_text_with_links import main as _extract_links_marker  # noqa: F401

# ---------------------------------------------------------------------------
# Init
# ---------------------------------------------------------------------------

load_dotenv()
SCRIPT_DIR = Path(__file__).resolve().parent

def _env_flag(name: str, default: bool = False) -> bool:
    """
    Interpret common truthy/falsey env values.

    Truthy examples: "1", "true", "yes", "on"
    Falsey examples: "0", "false", "no", "off", "" (or unset -> default)
    """
    val = os.getenv(name)
    if val is None:
        return default
    return str(val).strip().lower() not in ("0", "false", "no", "off", "")
    

# Should we colorize log output?
# Controlled via WIKIBENCH_COLOR in .env (default: True)
USE_COLOR = _env_flag("WIKIBENCH_COLOR", default=True)

def _color(code: str, text: str) -> str:
    if not USE_COLOR:
        return text
    return f"\033[{code}m{text}\033[0m"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def info(msg: str) -> None:
    # cyan
    print(_color("36", f"[INFO] {msg}"))

def warn(msg: str) -> None:
    # yellow
    print(_color("33", f"[WARN] {msg}"))

def error(msg: str) -> None:
    # red + bold
    print(_color("1;31", f"[ERROR] {msg}"))

def debug(msg: str) -> None:
    if os.getenv("WIKIBENCH_DEBUG") == "1":
        # magenta
        print(_color("35", f"[DEBUG] {msg}"))

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
    # 1) Explicit CLI flag always wins; treat it as given (cwd-relative or absolute)
    if cli_path:
        p = Path(cli_path).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"Prompt file not found: {p}")
        return p

    # 2) Environment variable, relative to SCRIPT_DIR if not absolute
    env_path = os.getenv("WIKIBENCH_PROMPT_PATH")
    if env_path:
        p = Path(env_path)
        if not p.is_absolute():
            # resolve relative to the harness location (repo root-ish)
            p = (SCRIPT_DIR / p).expanduser().resolve()
        else:
            p = p.expanduser().resolve()

        if not p.exists():
            raise FileNotFoundError(f"WIKIBENCH_PROMPT_PATH points to missing file: {p}")
        return p

    # 3) Default prompt in prompts/prompt.yml next to the harness
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
    Placeholders inside the YAML template:
      - {{TARGET_TITLE}}
      - {{VISITED_JSON}}
      - {{ARTICLE_HTML}}
    """
    yaml_text = yaml.dump(template, sort_keys=False, allow_unicode=True, width=100000)
    yaml_text = yaml_text.replace("{{TARGET_TITLE}}", target_title)
    yaml_text = yaml_text.replace("{{VISITED_JSON}}", json.dumps(visited, ensure_ascii=False))
    yaml_text = yaml_text.replace("{{ARTICLE_HTML}}", html_snippet)
    return f"# WikiBench Prompt\n{yaml_text}"

# ---------------------------------------------------------------------------
# HTML extraction (subprocess call to extractor CLI)
# ---------------------------------------------------------------------------

def get_article_html(title: str) -> str:
    """
    Runs ./extract_text_with_links.py <title> and captures stdout.
    Falls back to 'python3 extract_text_with_links.py', then 'python'.
    Returns raw string (already simplified with <link title="...">...).
    """
    title_arg = title.replace("_", " ")
    candidates = [
        [str(SCRIPT_DIR / "extract_text_with_links.py"), title_arg],
        ["python3", str(SCRIPT_DIR / "extract_text_with_links.py"), title_arg],
        ["python", str(SCRIPT_DIR / "extract_text_with_links.py"), title_arg],
    ]
    for cmd in candidates:
        try:
            debug(f"Trying extractor: {' '.join(cmd)}")
            proc = subprocess.run(cmd, capture_output=True, check=True)
            out = proc.stdout.decode("utf-8", errors="ignore")
            # Extract only the mw-parser-output block if present (keeps noise down)
            m = re.search(r'(<div class="mw-content-ltr mw-parser-output".*?</div>)', out, re.DOTALL)
            return m.group(1) if m else out
        except Exception as e:
            debug(f"Extractor failed with {e}; trying next candidate...")
    raise RuntimeError("Failed to run extract_text_with_links.py via subprocess.")

# ---------------------------------------------------------------------------
# Parse <link title="...">
# ---------------------------------------------------------------------------

_LINK_TITLE_RE = re.compile(r'<link\b[^>]*\btitle="([^"]+)"', re.IGNORECASE)

def normalize_title(t: str) -> str:
    return t.strip().replace(" ", "_")

def title_key_for_compare(t: str) -> str:
    return normalize_title(t).lower()

def extract_link_titles(html_snippet: str) -> Tuple[List[str], Dict[str, str], set]:
    """
    Returns:
      - ordered list of raw titles (as they appear)
      - map: normalized_key -> canonical_title_with_underscores
      - set of normalized keys for membership checks
    """
    titles = _LINK_TITLE_RE.findall(html_snippet or "")
    key_to_title: Dict[str, str] = {}
    for t in titles:
        key = title_key_for_compare(t)
        # last one wins, but titles are usually consistent on the page
        key_to_title[key] = normalize_title(t)
    available_norm = set(key_to_title.keys())
    return titles, key_to_title, available_norm

# ---------------------------------------------------------------------------
# Ollama model discovery & invocation (optional picker)
# ---------------------------------------------------------------------------

def discover_ollama_models_http(host: str) -> Optional[List[Dict[str, Any]]]:
    url = host.rstrip("/") + "/api/tags"
    debug(f"Fetching Ollama tags via HTTP: {url}")
    try:
        req = Request(url, headers={"Accept": "application/json"})
        with urlopen(req, timeout=5) as resp:
            payload = json.loads(resp.read().decode("utf-8", errors="ignore"))
            # Shape: {"models":[{"name":"llama3:8b", "details": {...}}, ...]}
            return payload.get("models", [])
    except URLError as e:
        debug(f"HTTP discovery failed: {e}")
        return None
    except Exception as e:
        debug(f"HTTP discovery error: {e}")
        return None

def discover_ollama_models_cli() -> List[Dict[str, Any]]:
    """
    Fallback to `ollama list`. Parse coarse fields.
    """
    try:
        out = subprocess.check_output(["ollama", "list"]).decode("utf-8", errors="ignore").strip()
    except Exception as e:
        warn(f"Failed to run 'ollama list': {e}")
        return []
    lines = [ln for ln in out.splitlines() if ln.strip()]
    # skip header row if present
    if lines and lines[0].lower().startswith("name"):
        lines = lines[1:]
    models: List[Dict[str, Any]] = []
    for ln in lines:
        cols = [c.strip() for c in re.split(r"\s{2,}", ln)]
        if not cols:
            continue
        name = cols[0]
        size = cols[1] if len(cols) > 1 else ""
        models.append({"name": name, "details": {"parameter_size": "", "quantization_level": "", "size": size}})
    return models

def interactive_pick_model(models: List[Dict[str, Any]]) -> Optional[str]:
    if not models:
        warn("No models discovered.")
        return None

    pretty = []
    for m in models:
        name = m.get("name", "")
        det = m.get("details", {}) or {}
        size_bytes = det.get("size", 0)
        size_str = ""
        if isinstance(size_bytes, (int, float)) and size_bytes > 0:
            gb = size_bytes / (1024**3)
            mb = size_bytes / (1024**2)
            size_str = f"{gb:.1f} GB" if gb >= 0.8 else f"{mb:.1f} MB"
        elif isinstance(size_bytes, str):
            size_str = size_bytes
        family = det.get("family", "") or det.get("format", "")
        params = det.get("parameter_size", "")
        quant  = det.get("quantization_level", "")
        pretty.append((name, size_str, family, params, quant))

    page_size = 20
    page = 0
    total = len(pretty)

    info(f"Discovered {total} model(s). Use number to select, 'n' next page, 'p' previous, 'q' to quit.")
    while True:
        start = page * page_size
        end = min(start + page_size, total)
        print()
        info(f"Models {start+1}-{end} of {total} (page {page+1}/{(total-1)//page_size+1})")
        for idx, (name, size_str, family, params, quant) in enumerate(pretty[start:end], start=start+1):
            print(f"  [{idx:3d}] {name:<28} | {size_str:>8} | family={family or '-'} | params={params or '-'} | quant={quant or '-'}")

        sel = input("\nSelect model #: ").strip().lower()
        if sel == "q":
            return None
        if sel == "n":
            if end < total:
                page += 1
            continue
        if sel == "p":
            if page > 0:
                page -= 1
            continue

        if sel.isdigit():
            i = int(sel)
            if 1 <= i <= total:
                chosen = pretty[i-1][0]
                info(f"Selected model: {chosen}")
                return chosen
            else:
                warn("Out of range.")
        else:
            warn("Invalid input.")

def pick_model_via_discovery(default_host: str = "http://127.0.0.1:11434") -> Optional[str]:
    host = os.getenv("OLLAMA_HOST", default_host)
    models = discover_ollama_models_http(host)
    if models is None:
        debug("Falling back to `ollama list` for discovery.")
        models = discover_ollama_models_cli()
    return interactive_pick_model(models)

def run_ollama_prompt(model: str, prompt: str) -> str:
    """
    Invoke Ollama CLI and return the model's raw text response.
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
# Choice parsing & validation
# ---------------------------------------------------------------------------

_QUOTED_RE = re.compile(r'(?P<q>`{1,3}|["\'])\s*(?P<txt>[^`"\']+?)\s*(?P=q)')
_TITLE_TOKEN_RE = re.compile(r"[A-Za-z0-9 _().,'\-]+")

def first_nonempty_line(s: str) -> str:
    for line in (s or "").splitlines():
        line = line.strip()
        if line:
            return line
    return ""

def parse_choice(raw: str,
                 available_map: Dict[str, str],
                 target_norm: str) -> Tuple[str, str]:
    """
    Try to resolve the model text to a valid choice.

    Returns (choice_key_or_SPECIAL, reason) where:
      - choice_key_or_SPECIAL is:
          * normalized key of a title from available_map,
          * "__STOP__" (STOP),
          * "" on failure.
      - reason is one of:
          "ok", "stop", "could_not_parse", "target_not_in_available_links"
    """
    text = raw or ""
    if not text.strip():
        return "", "could_not_parse"

    # Normalize lines once
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    # --- STOP detection (even though prompt doesn't mention STOP anymore) ---
    for ln in lines:
        if ln.upper() == "STOP":
            return "__STOP__", "stop"

    # --- Build candidate strings ---

    candidates: List[str] = []

    # 1) Prefer the *last* non-meta line ("Thinking..." style preambles are ignored)
    def is_meta_line(ln: str) -> bool:
        low = ln.lower()
        if low.startswith("thinking"):
            return True
        if low in ("...done thinking.", "done.", "done"):
            return True
        return False

    core_line = ""
    for ln in reversed(lines):
        if not is_meta_line(ln):
            core_line = ln
            break

    if core_line:
        # Strip common chatter prefixes on that core line
        prefixes = [
            "the selected link is",
            "selected link is",
            "selected:",
            "selection:",
            "next:",
            "next link:",
            "choose:",
            "choice:",
            "pick:",
            "the output will be",
            "the final output will be",
        ]
        low = core_line.lower()
        for p in prefixes:
            if low.startswith(p):
                if ":" in core_line:
                    core_line = core_line.split(":", 1)[1].strip()
                else:
                    core_line = core_line[len(p):].strip()
                break

        # From core line: quoted substrings, post-colon text, whole line, tokens
        for m in _QUOTED_RE.finditer(core_line):
            candidates.append(m.group("txt").strip())
        if ":" in core_line:
            candidates.append(core_line.split(":", 1)[1].strip())
        candidates.append(core_line.strip())
        for m in _TITLE_TOKEN_RE.finditer(core_line):
            tok = m.group(0).strip()
            if tok:
                candidates.append(tok)

    # 2) From the entire text: quoted substrings anywhere
    for m in _QUOTED_RE.finditer(text):
        candidates.append(m.group("txt").strip())

    # 3) From each line: text after colon (e.g., "The answer is: Ada_Lovelace")
    for ln in lines:
        if ":" in ln:
            candidates.append(ln.split(":", 1)[1].strip())

    # 4) All non-meta lines themselves
    for ln in lines:
        if not is_meta_line(ln):
            candidates.append(ln)

    # 5) Tokenization over the whole text (catches Ada_Lovelace, National_Transportation_Safety_Board, etc.)
    for m in _TITLE_TOKEN_RE.finditer(text):
        tok = m.group(0).strip()
        if tok:
            candidates.append(tok)

    # Clean up obvious junk candidates
    cleaned: List[str] = []
    for c in candidates:
        if not c:
            continue
        low = c.lower()
        if low.startswith("thinking"):
            continue
        if low in ("...done thinking.", "done.", "done"):
            continue
        cleaned.append(c)

    candidates = cleaned
    if not candidates:
        return "", "could_not_parse"

    # --- Target-mention detection for "target_not_in_available_links" ---
    avail_keys = set(available_map.keys())
    cand_keys = [title_key_for_compare(c) for c in candidates if c]

    target_space = target_norm.replace("_", " ").lower()
    target_mentioned = False
    for c in candidates:
        cl = c.lower()
        if title_key_for_compare(c) == target_norm or target_space in cl:
            target_mentioned = True

    # 1) Exact normalized match with available titles?
    for ck in cand_keys:
        if ck in avail_keys:
            return ck, "ok"

    # 2) Substring / fuzzy-ish match against canonical titles
    for ck, canon in available_map.items():
        canon_space = canon.replace("_", " ").lower()
        for c in candidates:
            if canon_space in c.lower():
                return ck, "ok"

    # 3) Target was clearly mentioned but isn't actually a link here → hallucination
    if target_mentioned:
        return "", "target_not_in_available_links"

    return "", "could_not_parse"


def validate_not_visited(choice_key: str, visited_norm: set) -> Tuple[bool, str]:
    if choice_key in visited_norm:
        return False, "repeated"
    return True, "ok"

# ---------------------------------------------------------------------------
# Retry feedback
# ---------------------------------------------------------------------------

def build_strict_feedback(available_map: Dict[str, str],
                          target_norm: str,
                          max_chars: int = 12000) -> str:
    """
    Construct a strict re-ask that enumerates allowed titles and format rules.

    IMPORTANT: We only list titles that actually exist as <link title="...">
    on this page. The target will only be present here if it is truly linked.
    """
    allowed = list(dict.fromkeys(available_map.values()))  # preserve order, de-dup

    block = "\n".join(allowed)
    if len(block) > max_chars:
        block = block[:max_chars] + "\n..."

    target_natural = target_norm.replace("_", " ")

    return (
        "\n# STRICT OUTPUT FORMAT (RETRY)\n"
        f"Your long-term goal is to reach the Wikipedia article titled \"{target_natural}\".\n"
        "From the list below, choose EXACTLY ONE title that you believe moves you closer to that target.\n"
        "Do NOT choose randomly. Prefer links that are semantically related to the target topic or\n"
        "that are likely to lead toward it through a small number of hops.\n"
        "You may only output one of the titles listed below, or STOP if no valid move exists.\n"
        "Return EXACTLY ONE of the following titles, verbatim, on a SINGLE line.\n"
        "No quotes, no extra words, no punctuation. If no valid move exists, output STOP.\n"
        "Allowed titles:\n"
        f"{block}\n"
    )

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

    target_norm = title_key_for_compare(target)

    for step in range(1, max_steps + 1):
        html_snippet = get_article_html(current)
        titles_raw, available_map, available_norm = extract_link_titles(html_snippet)

        # Prepare base prompt
        prompt = render_prompt(prompt_template, current, target, html_snippet, visited)

        attempt = 0
        choice_key_final = ""
        raw_history: List[str] = []
        invalid_reason = ""
        valid = False

        while attempt <= max_retries:
            attempt += 1
            # debug(f"\n=== Prompt for step {step}, attempt {attempt} ===\n{prompt}")
            raw = run_ollama_prompt(model, prompt)
            debug(f"\n=== Raw model output ===\n{raw}")

            raw_history.append(raw)

            choice_key, reason = parse_choice(raw, available_map, target_norm)

            if choice_key == "__STOP__":
                status = "stopped"
                invalid_reason = "stop"
                break

            if not choice_key:
                # could_not_parse, target_not_in_available_links, etc.
                invalid_reason = reason
            else:
                # Must be a genuine link on this page
                if choice_key not in available_norm:
                    invalid_reason = "not_in_available_links"
                else:
                    ok, r = validate_not_visited(choice_key, visited_norm)
                    if not ok:
                        invalid_reason = r  # "repeated"
                    else:
                        choice_key_final = choice_key
                        valid = True
                        break

            # Retry with strict feedback
            warn(f"Invalid selection (attempt {attempt}/{max_retries}): {raw.splitlines()[0:1] or ['<empty>']} — {invalid_reason}")
            if attempt <= max_retries:
                prompt = prompt + build_strict_feedback(available_map, target_norm)

        # Record step trace
        chosen_title = None
        if valid:
            chosen_title = available_map[choice_key_final]

        trace.append({
            "step": step,
            "current": current,
            "target": target,
            "available_links_count": len(titles_raw),
            "model_raw_attempts": raw_history,
            "chosen": chosen_title,
            "valid": valid,
            "invalid_reason": (None if valid else invalid_reason),
            "retries_used": (attempt - 1),
        })

        if not valid:
            if invalid_reason == "stop":
                info("Model requested STOP.")
                status = "stopped"
            else:
                error("Model failed to produce a valid next link within retry budget.")
                status = "failed"
            break

        # Advance
        next_title = chosen_title  # canonical title with underscores
        if title_key_for_compare(next_title) == target_norm:
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

    # Prefer a path relative to the repo root (SCRIPT_DIR) if possible
    try:
        rel_prompt_path = prompt_path.relative_to(SCRIPT_DIR)
        prompt_path_str = str(rel_prompt_path)
    except ValueError:
        # prompt is outside the repo; keep absolute path
        prompt_path_str = str(prompt_path)

    payload: Dict[str, Any] = {
        "meta": {
            "prompt_template_path": prompt_path_str,
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

def main() -> None:
    parser = argparse.ArgumentParser(description="Run WikiBench benchmark with Ollama")
    parser.add_argument("--model", help="Ollama model name (e.g., llama3.1:latest)")
    parser.add_argument(
        "--pick-model",
        action="store_true",
        help="Interactively pick an installed Ollama model",
    )
    parser.add_argument("--start", required=True, help="Starting article title")
    parser.add_argument(
        "--target",
        required=False,
        help="Target article title (falls back to $WIKIBENCH_TARGET if omitted)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=20,
        help="Maximum navigation steps",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help="Retries per step for invalid output",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compute optimal path for comparison (slow)",
    )
    parser.add_argument(
        "--prompt",
        help="Path to YAML prompt template (overrides env/default)",
    )
    parser.add_argument(
        "--out-format",
        choices=["json", "yaml"],
        default="json",
        help="Results file format",
    )
    args = parser.parse_args()

    # Model selection
    model = args.model
    if not model and args.pick_model:
        chosen = pick_model_via_discovery()
        if not chosen:
            error("No model selected.")
            return
        model = chosen
    if not model:
        model = "llama3.1:latest"
        warn(f"--model not provided; using default: {model}")

    # Target resolution: CLI flag wins, otherwise fall back to env
    target = args.target or os.getenv("WIKIBENCH_TARGET")
    if not target:
        error(
            "No target article specified. Use --target or set $WIKIBENCH_TARGET in your .env."
        )
        return

    prompt_path = resolve_prompt_path(args.prompt)
    prompt_template = load_prompt_template(prompt_path)

    trace, summary = run_navigation(
        model=model,
        start_title=args.start,
        target_title=target,
        max_steps=args.max_steps,
        max_retries=args.max_retries,
        prompt_template=prompt_template,
    )

    # Print a concise machine-readable summary for this run
    # {
    #   "summary": {
    #     "model": "...",
    #     "start": "...",
    #     "target": "...",
    #     "max_steps": ...,
    #     "max_retries": ...,
    #     "status": "...",
    #     "visited": [...],
    #     "hops": ...,
    #     "reached_target": ...
    #   }
    # }
    print(json.dumps({"summary": summary}, ensure_ascii=False, indent=2))

    comparison = compute_optimal_path(args.start, target) if args.compare else None

    save_results(
        model=model,
        start=args.start,
        target=target,
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
