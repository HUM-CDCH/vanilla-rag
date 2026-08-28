"""The vanilla RAG pipeline: query -> retrieve -> pack to 5K budget -> generate."""

from pathlib import Path

from .chunk import chunk_text, count_tokens
from .config import DEFAULT_CONFIG, RAGConfig
from .embed import Embedder
from .generate import Generator, build_prompt
from .parse import SUPPORTED_SUFFIXES, extract_text
from .store import VectorStore


class RAGPipeline:
    def __init__(self, config: RAGConfig = DEFAULT_CONFIG):
        self.config = config
        self.embedder = Embedder(config.embedding_model)
        self.store = VectorStore(
            uri=config.milvus_uri,
            collection_name=config.collection_name,
            dim=config.embedding_dim,
            metric_type=config.metric_type,
        )

    # ---------- Ingestion ----------

    def ingest(self, path: str, reindex: bool = False) -> int:
        """Parse, chunk, embed, and index one document.

        Documents are identified by resolved path: already-indexed files are
        skipped (returns -1) unless `reindex` is set, in which case their old
        chunks are replaced. Returns the number of chunks indexed.
        """
        source = str(Path(path).resolve())
        if self.store.has_source(source):
            if not reindex:
                return -1
            self.store.delete_source(source)

        text = extract_text(path)
        chunks = chunk_text(
            text,
            self.embedder.tokenizer,
            chunk_size=self.config.chunk_size,
            overlap=self.config.chunk_overlap,
        )
        if not chunks:
            return 0
        embeddings = self.embedder.embed_chunks(chunks)
        return self.store.insert(embeddings, chunks, source=source)

    def ingest_collection(
        self, root: str, reindex: bool = False, on_progress=print
    ) -> dict:
        """Recursively index every supported document under `root`.

        One failing document (corrupt PDF, encrypted file, ...) is reported
        and skipped rather than aborting the run. Returns summary counts.
        """
        files = sorted(
            p
            for p in Path(root).rglob("*")
            if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
        )
        summary = {"indexed": 0, "skipped": 0, "failed": 0, "chunks": 0}
        for i, path in enumerate(files, 1):
            prefix = f"[{i}/{len(files)}] {path}"
            try:
                n = self.ingest(str(path), reindex=reindex)
            except Exception as exc:
                summary["failed"] += 1
                on_progress(f"{prefix}: FAILED ({exc})")
                continue
            if n < 0:
                summary["skipped"] += 1
                on_progress(f"{prefix}: already indexed, skipped")
            else:
                summary["indexed"] += 1
                summary["chunks"] += n
                on_progress(f"{prefix}: {n} chunks")
        return summary

    # ---------- Retrieval ----------

    def retrieve(self, query: str) -> list[dict]:
        """Fetch top candidates and keep the best ones within the token budget."""
        query_vector = self.embedder.embed_query(query)
        candidates = self.store.search(query_vector, top_k=self.config.top_k_candidates)

        selected: list[dict] = []
        budget = self.config.context_budget_tokens
        used = 0
        for cand in candidates:  # already sorted best-first by Milvus
            cost = count_tokens(cand["text"], self.embedder.tokenizer)
            if used + cost > budget:
                break
            selected.append(cand)
            used += cost
        return selected

    # ---------- End-to-end ----------

    def answer(self, query: str, generator: Generator) -> dict:
        chunks = self.retrieve(query)
        if not chunks:
            return {"answer": "No relevant context found in the database.", "chunks": []}
        prompt = build_prompt(query, chunks)
        return {"answer": generator.generate(prompt), "chunks": chunks}
