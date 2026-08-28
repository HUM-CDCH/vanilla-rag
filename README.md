# Vanilla RAG Pipeline

A minimal single-turn RAG baseline: **query → vector search over chunks → LLM answers from the retrieved chunks**.

## Parameters (Standard RAG baseline)

| Component | Setting |
|---|---|
| Document parsing | PyMuPDF raw text extraction |
| Chunk size | 512 tokens (no overlap) |
| Embedding model | `all-MiniLM-L6-v2` (384-dim) |
| Vector database | Milvus (Milvus Lite by default — a local `.db` file) |
| Context budget | retrieved context in the prompt capped at 5,000 tokens |
| Generation | Claude (`claude-opus-5`) or a local LLM via Ollama |

All of these are set in [config.toml](config.toml) at the project root — edit that file to change chunk size, models, database URI, budgets, etc. Every key is optional; anything omitted falls back to the defaults in [vanilla_rag/config.py](vanilla_rag/config.py), and unknown keys are rejected so typos can't silently revert to defaults. Programmatic use can load an alternative file with `load_config("path/to/other.toml")`.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

For the Claude backend, authenticate with `export ANTHROPIC_API_KEY=...` (or an `ant auth login` profile). For the local backend, have [Ollama](https://ollama.com) running with a pulled model (default: `gpt-oss:20b`; override with `--model`).

## Usage

```bash
# 1. Index a document collection: drop your PDFs into ./data and run
python ingest.py                       # indexes everything under ./data, recursively
python ingest.py my_corpus/            # or any other folder
python ingest.py paper.pdf notes.md    # or individual files
python ingest.py --reindex             # re-embed files already in the index
python ingest.py --reset               # wipe the collection and start over

# 2. Ask questions
python query.py "What method does the paper propose?"
python query.py --backend ollama "What method does the paper propose?"
python query.py --show-chunks "..."    # inspect the retrieved context
```

Indexing is **incremental and idempotent**: documents are identified by resolved path, so re-running `python ingest.py` after adding new PDFs to `data/` only processes the new files. Corrupt or unreadable documents are reported and skipped without aborting the run (the script exits non-zero if any failed).

Or from Python:

```python
from vanilla_rag import RAGPipeline
from vanilla_rag.generate import AnthropicGenerator

pipeline = RAGPipeline()
pipeline.ingest_collection("data")   # or pipeline.ingest("paper.pdf")
result = pipeline.answer("What is the main finding?", AnthropicGenerator())
print(result["answer"])
```

## How it works

1. **Ingest** ([vanilla_rag/parse.py](vanilla_rag/parse.py), [chunk.py](vanilla_rag/chunk.py)): PyMuPDF extracts raw text; it is segmented into 512-token windows using the MiniLM tokenizer.
2. **Embed & index** ([embed.py](vanilla_rag/embed.py), [store.py](vanilla_rag/store.py)): chunks are embedded with all-MiniLM-L6-v2 and stored in Milvus with cosine similarity.
3. **Retrieve** ([pipeline.py](vanilla_rag/pipeline.py)): the query is embedded, the top-20 candidates are fetched, and chunks are packed best-first until the 5K-token context budget is reached (~9–10 chunks).
4. **Generate** ([generate.py](vanilla_rag/generate.py)): the chunks and question go to the LLM, which is instructed to answer only from the provided context and cite passage numbers.

## Notes & deviations

- **MinerU**: the paper's baseline combines PyMuPDF and MinerU for parsing. This pipeline uses PyMuPDF only; if you need MinerU's layout/element extraction (tables, formulas), run it separately and ingest its Markdown output as `.md` files.
- **MiniLM sequence limit**: all-MiniLM-L6-v2 truncates inputs to 256 tokens internally, so each 512-token chunk is embedded from its first ~256 tokens. This is inherent to using this model with 512-token chunks (the baseline setup has the same property).
- **Token counting**: both the 512-token chunk size and the 5K context budget are counted with the MiniLM tokenizer, so one consistent unit is used throughout. Counts differ slightly from the generator LLM's own tokenizer.
- **Milvus**: Milvus Lite stores everything in `milvus_vanilla_rag.db` — no server needed. To use a full Milvus deployment, set `milvus_uri` to e.g. `http://localhost:19530` in the config.
