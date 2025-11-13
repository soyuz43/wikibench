# WikiBench

**WikiBench** is an experimental benchmarking framework designed to evaluate large language models on their ability to perform *semantic navigation* through the hyperlink structure of Wikipedia.

The benchmark measures how efficiently an LLM can traverse from a **random starting article** to a fixed **target endpoint** (e.g., `Adolf_Hitler`) by declaring intermediate links — simulating the “fewest clicks” challenge of the human *Wiki Game*.

WikiBench provides reproducible ground-truth paths, cached Wikipedia content, and automated graph traversal tools for model comparison.

---

## Core Components

### `wikibench_pathfinder_async.py`

Implements a **bidirectional asynchronous breadth-first search (BFS)** to determine the shortest valid hyperlink path between two Wikipedia entities.
Outputs include:

* Verified shortest path for benchmarking.
* Forward and backward traversal trees.
* Optional human-readable visualizations stored in `cache/viz/`.

This module defines the **objective baseline** for evaluating LLM reasoning efficiency.

---

### `extract_text_with_links.py`

Fetches and parses the HTML content of a Wikipedia article, extracting text and hyperlinks as structured, LLM-readable markup.

Example output:

```html
<div class="mw-content-ltr mw-parser-output" dir="ltr" lang="en">
UPS Airlines Flight 2976 was a scheduled domestic <link title="Air cargo">cargo flight</link> ...
</div>
```

This allows models to “see” the page’s contents and available link choices without direct web access, enabling simulation of reasoning under constrained observation.

---

### `wikibench_admin.py`

Administrative and developer CLI for managing the WikiBench cache and environment.

Capabilities:

* Clear cached Wikipedia data interactively (`clean` command).
* Optional prompt to remove visualization outputs.
* Display environment configuration (`env` command).
* Launch interactive maintenance shell (`shell` command).

---

## Project Layout

```
wikibench/
├── cache/                      # Cached pages and visualizations
├── extract_text_with_links.py  # Wikipedia text/link extraction utility
├── prompt.yml                  # Configuration or LLM prompt template
├── wikibench_admin.py          # Administrative CLI
└── wikibench_pathfinder_async.py # Ground-truth pathfinding and visualization logic
```

---

## Installation

### Requirements

* Python ≥ 3.9
* Dependencies: `aiohttp`, `beautifulsoup4`, `python-dotenv`

```bash
pip install aiohttp beautifulsoup4 python-dotenv
```

### Setup

```bash
git clone https://github.com/<your-username>/wikibench.git
cd wikibench
mkdir -p cache
```

(Optional) Configure `.env`:

```bash
WIKIBENCH_DEBUG=1
```

---

## Usage

### 1. Generate a Ground Truth Path

Find the shortest hyperlink path between two Wikipedia pages:

```bash
./wikibench_pathfinder_async.py "Systemic_functional_linguistics" "Adolf_Hitler" 2
```

**Example output:**

```
Forward search (depth 0–2):
Systemic_functional_linguistics
 └─> Functional_linguistics
      └─> Soviet_Union

Backward search (depth 0–1):
Adolf_Hitler
 └─> Soviet_Union
```

Results are saved under:

```
cache/viz/<timestamp>.txt
```

---

### 2. Extract Wikipedia Page Links

Generate an LLM-readable version of a page for reasoning experiments:

```bash
./extract_text_with_links.py "UPS Airlines Flight 2976"
```

Outputs cleaned article HTML with embedded `<link title="...">` tags.

---

### 3. Manage Cache and Environment

**Clean cache:**

```bash
./wikibench_admin.py clean
```

Prompts to remove both cached data and visualization directories.

**Inspect environment:**

```bash
./wikibench_admin.py env
```

**Interactive shell:**

```bash
./wikibench_admin.py shell
```

Preloaded utilities:

```python
clean_cache(confirm=False)
show_env()
info(), warn(), debug(), success(), error()
```

---

## Benchmark Concept

WikiBench evaluates **navigational reasoning** rather than static factual recall.
Each LLM episode consists of:

1. Starting from a random article.
2. Choosing a linked concept (`<link title="...">`) to move to next.
3. Repeating until it reaches the **target endpoint** or fails within a depth limit.

The system compares:

* **LLM traversal efficiency** (number of hops).
* **Semantic plausibility** of link selection.
* **Deviation** from the known optimal path (computed by `wikibench_pathfinder_async.py`).

This structure allows comparative scoring of models on **goal-directed reasoning**, **semantic compression**, and **graph navigation**.

---

## Design Principles

* **Determinism:** Ground-truth paths reproducible via BFS traversal.
* **LLM Interpretability:** Link extraction formatted for model consumption.
* **Asynchronous Efficiency:** Parallelized Wikipedia requests for rapid benchmarking.
* **Transparent Evaluation:** All intermediate states are cached and logged.

---

## Roadmap

* [ ] Implement scoring functions for LLM path deviation metrics.
* [ ] Integrate evaluation API for model interaction episodes.
* [ ] Add support for configurable endpoints and multiple benchmarks.
* [ ] Introduce leaderboard visualization and run aggregation tools.

---

## License

Released under the **MIT License**.
See `LICENSE` for details.
