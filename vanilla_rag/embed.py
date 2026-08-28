"""Embedding with all-MiniLM-L6-v2 (sentence-transformers)."""

from sentence_transformers import SentenceTransformer


class Embedder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        # Reuse the model's tokenizer for chunking and budget counting.
        self.tokenizer = self.model.tokenizer
        # Warm-up encode: torch's OpenMP thread pool must be initialized
        # before milvus-lite spins up its own, or encoding segfaults on macOS.
        self.model.encode("warmup")

    def embed_chunks(self, chunks: list[str]) -> list[list[float]]:
        vectors = self.model.encode(chunks, batch_size=64, show_progress_bar=True)
        return [v.tolist() for v in vectors]

    def embed_query(self, query: str) -> list[float]:
        return self.model.encode(query).tolist()
