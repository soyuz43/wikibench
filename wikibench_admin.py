#!/usr/bin/env python3
"""
WikiBench Admin CLI
-------------------
Administrative and developer tool for managing WikiBench cache, environment,
and interactive utilities. Provides commands to clean cached pages, inspect
active configuration, and launch an interactive Python shell.

Usage:
    ./wikibench_admin.py clean
    ./wikibench_admin.py env
    ./wikibench_admin.py shell

Environment:
    WIKIBENCH_DEBUG=1     # Enable verbose debug logging
"""

import os
import sys
import shutil
import argparse
import readline  # optional, improves shell UX
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE_DIR, "cache")
DEBUG = bool(int(os.getenv("WIKIBENCH_DEBUG", "0")))

# ---------------------------------------------------------------------------
# Logging helpers (consistent with wikibench_pathfinder_async.py)
# ---------------------------------------------------------------------------

def info(msg: str):
    print(f"[INFO] {msg}")

def success(msg: str):
    print(f"[SUCCESS] {msg}")

def warn(msg: str):
    print(f"[WARN] {msg}")

def error(msg: str):
    print(f"[ERROR] {msg}", file=sys.stderr)

def debug(msg: str):
    if DEBUG:
        print(f"[DEBUG] {msg}")

# ---------------------------------------------------------------------------
# Cache management
# ---------------------------------------------------------------------------

def clean_cache(confirm: bool = True):
    """
    Remove all cached JSON files in the cache directory.
    Optionally delete visualization outputs (cache/viz subdirectory).
    """
    if not os.path.exists(CACHE_DIR):
        warn(f"Cache directory does not exist at {CACHE_DIR}")
        return

    files = [f for f in os.listdir(CACHE_DIR) if os.path.isfile(os.path.join(CACHE_DIR, f))]
    viz_dir = os.path.join(CACHE_DIR, "viz")

    if not files and not os.path.exists(viz_dir):
        info("Cache is already empty (no cached files or visualizations).")
        return

    if files:
        info(f"Found {len(files)} file(s) in cache directory.")
        if confirm:
            response = input(f"[!]  This will delete {len(files)} cached file(s). Proceed? [y/N]: ").strip().lower()
            if response != "y":
                info("Cleanup aborted by user.")
                return

        for filename in files:
            path = os.path.join(CACHE_DIR, filename)
            try:
                os.remove(path)
                debug(f"Deleted {filename}")
            except Exception as e:
                error(f"Failed to remove {filename}: {e}")

        success(f"Cleared {len(files)} cached file(s).")

    # --- New visualization cleanup section ---
    if os.path.exists(viz_dir):
        response = input("Delete visualizations as well? [y/N]: ").strip().lower()
        if response == "y":
            try:
                shutil.rmtree(viz_dir)
                success("Deleted visualization subdirectory (cache/viz).")
            except Exception as e:
                error(f"Failed to delete visualization directory: {e}")
        else:
            info("Kept visualization subdirectory.")
    else:
        debug("No visualization subdirectory found.")

# ---------------------------------------------------------------------------
# Environment inspection
# ---------------------------------------------------------------------------

def show_env():
    """
    Display WikiBench-related environment variables.
    """
    info("Active WikiBench environment configuration:\n")
    env_keys = sorted(k for k in os.environ if k.startswith("WIKIBENCH_"))
    if not env_keys:
        warn("No WikiBench-specific environment variables found.")
        return

    for key in env_keys:
        val = os.getenv(key)
        print(f"  {key} = {val}")
    print()

# ---------------------------------------------------------------------------
# Interactive shell
# ---------------------------------------------------------------------------

def launch_shell():
    """
    Launch an interactive Python shell preloaded with WikiBench admin utilities.
    """
    import code
    banner = (
        "\nWikiBench Interactive Shell\n"
        "--------------------------------\n"
        "Preloaded globals:\n"
        "  - clean_cache(confirm=False)\n"
        "  - show_env()\n"
        "  - info(), debug(), warn(), success(), error()\n"
        "\nType exit() or Ctrl-D to quit.\n"
    )
    namespace = globals().copy()
    namespace.update(locals())
    info("Launching interactive shell...")
    code.interact(banner=banner, local=namespace)
    success("Exited interactive shell.")

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="WikiBench Administration CLI",
        epilog="Example: ./wikibench_admin.py clean",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # Subcommands
    sub.add_parser("clean", help="Clear cache directory")
    sub.add_parser("env", help="Show WikiBench environment configuration")
    sub.add_parser("shell", help="Launch interactive WikiBench shell")

    args = parser.parse_args()

    if args.command == "clean":
        clean_cache()
    elif args.command == "env":
        show_env()
    elif args.command == "shell":
        launch_shell()
    else:
        parser.print_help()

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        warn("Operation cancelled by user.")
