# TODO: Wikibench Benchmark Infrastructure

## Core Development Tasks

### LLM Benchmark Harness
- [x] Create `wikibench_ollama_harness.py`
  - [x] Accept model name, start + target concepts, and optional `max_depth`
  - [x] Call into async BFS logic
  - [x] Let the LLM choose which link to follow next
  - [x] Log full reasoning path

- [ ] Support multiple LLM backends
  - [ ] Local (Ollama)
  - [ ] API (OpenAI, Groq, etc. — via env vars if present)
  - [ ] Let user choose which provider at runtime (if multiple detected)

- [ ] Use env vars for configuration
  - [ ] Target concept (e.g., `WIKIBENCH_TARGET`)
  - [ ] Max depth (default: 6)
  - [ ] API keys (`OPENAI_API_KEY`, `GROQ_API_KEY`, etc.)

## Benchmark Infrastructure

### Benchmark Setup
- [ ] Accept CSV/YAML with `start,target` pairs
  - [ ] Allow `start=RANDOM` for random concept selection
- [ ] Precompute optimal paths via BFS and store hop counts
- [ ] Run models against this and compare model hops vs optimal

### Result Handling
- [x] Output JSON per run: path, hops, time, model metadata
- [ ] Write summary: success/failure, percent optimal, etc.

### Reproducibility
- [ ] Support random seed override via CLI or env var
- [ ] Log model name, version, timestamp, seed in results
- [ ] Consider fixed seeds for official benchmark runs

## Deployment & Stretch Goals

### Docker & Platform Support
- [ ] Add Dockerfile with all Python dependencies
- [ ] Add CLI entrypoint (`CMD ["python", "wikibench_ollama_harness.py"]`)
- [ ] Test on Windows native, WSL, and Linux

### Optional Web UI (GitHub Pages + Local Backend)
- [ ] Build static frontend for config input (model, start/target, etc.)
- [ ] Make backend Python server that listens on localhost and executes benchmark
- [ ] Bridge GitHub Pages frontend to local backend via JS (fetch calls to `localhost:port`)

## Optional Polish

- [ ] Replace hardcoded debug `"Adolf_Hitler"` with actual target variable
- [x] Add colored logging for debug/info/success paths
- [ ] Add CLI help messages for all scripts
- [x] Create `requirements.txt` or `pyproject.toml`
- [ ] Add usage examples to README
- [ ] Consider a `--headless` flag to suppress visualization

_Last updated: 2025-11-15_
