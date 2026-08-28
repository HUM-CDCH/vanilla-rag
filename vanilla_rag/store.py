"""Milvus vector store.

Uses MilvusClient, which works both with Milvus Lite (pass a local .db file
path) and a full Milvus server (pass a URI like http://localhost:19530).
"""

import json

from pymilvus import MilvusClient


class VectorStore:
    def __init__(
        self,
        uri: str,
        collection_name: str = "rag_chunks",
        dim: int = 384,
        metric_type: str = "COSINE",
    ):
        self.client = MilvusClient(uri)
        self.collection_name = collection_name
        if not self.client.has_collection(collection_name):
            self.client.create_collection(
                collection_name=collection_name,
                dimension=dim,
                metric_type=metric_type,
                auto_id=True,
            )
        else:
            # An existing collection reopens in 'released' state — load it
            # before any search/query.
            self.client.load_collection(collection_name)

    def insert(
        self, embeddings: list[list[float]], chunks: list[str], source: str
    ) -> int:
        rows = [
            {"vector": vec, "text": text, "source": source}
            for vec, text in zip(embeddings, chunks)
        ]
        res = self.client.insert(collection_name=self.collection_name, data=rows)
        return res["insert_count"]

    def search(self, query_vector: list[float], top_k: int = 20) -> list[dict]:
        results = self.client.search(
            collection_name=self.collection_name,
            data=[query_vector],
            limit=top_k,
            output_fields=["text", "source"],
        )
        return [
            {
                "text": hit["entity"]["text"],
                "source": hit["entity"]["source"],
                "score": hit["distance"],
            }
            for hit in results[0]
        ]

    def has_source(self, source: str) -> bool:
        """True if any chunks from this source document are already indexed."""
        rows = self.client.query(
            collection_name=self.collection_name,
            filter=f"source == {json.dumps(source)}",
            output_fields=["id"],
            limit=1,
        )
        return len(rows) > 0

    def delete_source(self, source: str) -> None:
        """Remove all chunks belonging to one source document."""
        self.client.delete(
            collection_name=self.collection_name,
            filter=f"source == {json.dumps(source)}",
        )

    def count(self) -> int:
        stats = self.client.get_collection_stats(self.collection_name)
        return int(stats["row_count"])

    def drop(self) -> None:
        if self.client.has_collection(self.collection_name):
            self.client.drop_collection(self.collection_name)
