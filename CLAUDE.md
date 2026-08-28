# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

A minimal single-turn RAG baseline: query → vector search over chunks → LLM answers from retrieved chunks. Mirrors the "Standard RAG" baseline from the AirQA paper (512-token chunks, all-MiniLM-L6-v2 embeddings, Milvus, 5K-token context budget). Not a git repository; no test suite or linter is configured.

## Commands

```bash
source .venv/bin/activate                # venv already exists at .venv
pip install -r requirements.txt

# Index documents (incremental & idempotent — keyed by resolved path)
python ingest.py                         # everything under ./data, recursively
python ingest.py --reindex               # re-embed already-indexed files
python ingest.py --reset                 # wipe the collection first

# Query
python query.py "question"
python query.py --backend ollama "question"
python query.py --show-chunks "question" # inspect retrieved context

# AirQA evaluation (resumable — re-running skips answered uuids)
python eval_airqa.py --gold gold.jsonl --corpus papers/ --out results/run1
# then score with AirQA's own harness:
# cd AirQA && python utils/eval_utils.py --gold gold.jsonl --dir results/run1
```

The Anthropic backend needs `ANTHROPIC_API_KEY`; the Ollama backend needs a local Ollama server running.

### AirQA corpora

- `results/airqa/papers/` — 36 PDFs manually fetched for an earlier 46-paper-coverage eval subset (10 more were already in `data/`; 4 were unobtainable). Superseded by the full corpus below, but kept since some were hard to re-acquire (e.g. OpenReview blocks `curl`). That run's outputs (`gold.jsonl`/`result.jsonl`/`score.jsonl`/`evaluation.txt`) were deleted 2026-08-27 to start fresh; findings are preserved in project memory.
- `milvus_airqa_full.db` (collection `airqa_full`, 530,159 chunks) — the *full* AirQA PDF corpus, ~13,956 papers at `/Users/jwm704/Documents/code-projects/AIR-QA-Dataset/air-qa-pdf/`, indexed via `--milvus-uri milvus_airqa_full.db --collection airqa_full`. Full gold set (1246 questions) at `/Users/jwm704/Documents/code-projects/AIR-QA-Dataset/data/test_data.jsonl` — every question's required PDFs are present in this corpus. No scored eval has been completed against it yet (see Gotchas).

## Architecture

Pipeline flow, one module per stage under `vanilla_rag/`:

1. **parse.py** — PyMuPDF raw text extraction (`SUPPORTED_SUFFIXES` gates which files ingest picks up)
2. **chunk.py** — segments text into 512-token windows using the *MiniLM tokenizer* (all token counting in the project — chunk size and the 5K context budget — uses this tokenizer, not the generator LLM's)
3. **embed.py** — all-MiniLM-L6-v2 (384-dim)
4. **store.py** — `VectorStore` wraps `MilvusClient`; works with both Milvus Lite (local `.db` file, the default) and a server URI. Chunks carry a `source` field (resolved file path) used for dedup/skip, reindex, and delete
5. **pipeline.py** — `RAGPipeline` orchestrates: retrieval fetches top-20 candidates, then packs chunks best-first until the 5K-token budget is hit (~9–10 chunks)
6. **generate.py** — `Generator` protocol with `AnthropicGenerator` and `OllamaGenerator` implementations; `build_prompt` numbers passages so the LLM can cite them

**Configuration** (`config.py`): a frozen `RAGConfig` dataclass. Values load from `config.toml` at the project root; every key optional, unknown keys raise (so typos can't silently revert to defaults). `DEFAULT_CONFIG` is built at import time. To vary settings per-run without editing the file, use `dataclasses.replace(DEFAULT_CONFIG, ...)` — this is how `eval_airqa.py` points at a separate Milvus db (`milvus_airqa_eval.db`) so evals never touch the main index (`milvus_vanilla_rag.db`).

Entry-point scripts at the root (`ingest.py`, `query.py`, `eval_airqa.py`) are thin CLIs over `RAGPipeline`.

## Gotchas

- all-MiniLM-L6-v2 truncates to 256 tokens internally, so each 512-token chunk is embedded from its first ~256 tokens — intentional, matches the baseline setup.
- `eval_airqa.py` rewrites `result.jsonl` in gold order after every answer; errors are recorded as `[ERROR]: ...` answers (scored 0) rather than aborting the run.
- An existing Milvus collection reopens in 'released' state; `VectorStore.__init__` handles the required `load_collection` call.
- A prior attempt to eval the full AirQA gold set against `milvus_airqa_full.db` failed silently in the sense that it completed and wrote results, but every answer was `[ERROR]: Could not resolve authentication method...` because `ANTHROPIC_API_KEY` wasn't set. That output was deleted 2026-08-27 — rerun with the key set (or `--backend ollama`) to get a real full-corpus score.
