"""Central configuration for the vanilla RAG pipeline.

Parameters follow the Standard RAG (single-turn) baseline:
  - 512-token chunks
  - all-MiniLM-L6-v2 embeddings (384-dim)
  - Milvus vector database
  - retrieved context capped at 5K tokens

Values are loaded from `config.toml` in the project root (or the current
directory); the dataclass defaults below apply for any key the file omits.
"""

import tomllib
from dataclasses import dataclass, fields
from pathlib import Path


@dataclass(frozen=True)
class RAGConfig:
    # Chunking
    chunk_size: int = 512          # tokens per chunk
    chunk_overlap: int = 0         # baseline uses plain segmentation, no overlap

    # Embedding
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384

    # Vector database (Milvus Lite: a local file; swap for a server URI in prod)
    milvus_uri: str = "milvus_vanilla_rag.db"
    collection_name: str = "rag_chunks"
    metric_type: str = "COSINE"

    # Retrieval
    top_k_candidates: int = 20     # candidates fetched before budget packing
    context_budget_tokens: int = 5000  # max retrieved context in the LLM prompt

    # Generation
    anthropic_model: str = "claude-opus-5"
    ollama_model: str = "gpt-oss:20b"
    ollama_host: str = "http://localhost:11434"
    max_answer_tokens: int = 16000


def _find_config_file() -> Path | None:
    candidates = [
        Path.cwd() / "config.toml",
        Path(__file__).resolve().parent.parent / "config.toml",  # project root
    ]
    for path in candidates:
        if path.is_file():
            return path
    return None


def load_config(path: str | Path | None = None) -> RAGConfig:
    """Build a RAGConfig from a TOML file, falling back to defaults.

    Unknown keys in the file raise, so typos don't silently fall back to
    default values.
    """
    config_path = Path(path) if path else _find_config_file()
    if config_path is None:
        return RAGConfig()

    with open(config_path, "rb") as f:
        overrides = tomllib.load(f)

    valid = {f.name for f in fields(RAGConfig)}
    unknown = set(overrides) - valid
    if unknown:
        raise ValueError(
            f"Unknown key(s) in {config_path}: {sorted(unknown)}. "
            f"Valid keys: {sorted(valid)}"
        )
    return RAGConfig(**overrides)


DEFAULT_CONFIG = load_config()
