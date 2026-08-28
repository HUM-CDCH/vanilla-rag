#!/usr/bin/env python3
"""Index documents (single files or whole folders) into the Milvus database.

Usage:
    python ingest.py                    # index everything under ./data
    python ingest.py my_corpus/         # index a different folder (recursive)
    python ingest.py paper.pdf notes.md # index individual files
    python ingest.py --reindex          # re-embed files that are already indexed
    python ingest.py --reset            # drop the whole collection first

Already-indexed files (matched by resolved path) are skipped, so the script
can be re-run after adding new PDFs to the folder and only the new ones are
processed. A corrupt or unreadable document is reported and skipped.
"""

import argparse
import sys
from pathlib import Path

from vanilla_rag import RAGPipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Index documents or document collections into the RAG database."
    )
    parser.add_argument(
        "paths",
        nargs="*",
        default=["data"],
        help="Files and/or folders to index (default: ./data, recursive)",
    )
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="Re-embed documents that are already in the index",
    )
    parser.add_argument(
        "--reset", action="store_true", help="Drop the existing collection first"
    )
    args = parser.parse_args()

    pipeline = RAGPipeline()
    if args.reset:
        pipeline.store.drop()
        pipeline.store = type(pipeline.store)(
            uri=pipeline.config.milvus_uri,
            collection_name=pipeline.config.collection_name,
            dim=pipeline.config.embedding_dim,
            metric_type=pipeline.config.metric_type,
        )
        print("Collection reset.")

    totals = {"indexed": 0, "skipped": 0, "failed": 0, "chunks": 0}
    for path in args.paths:
        p = Path(path)
        if not p.exists():
            print(f"{path}: not found", file=sys.stderr)
            totals["failed"] += 1
        elif p.is_dir():
            summary = pipeline.ingest_collection(str(p), reindex=args.reindex)
            for key in totals:
                totals[key] += summary[key]
        else:
            try:
                n = pipeline.ingest(str(p), reindex=args.reindex)
            except Exception as exc:
                print(f"{path}: FAILED ({exc})", file=sys.stderr)
                totals["failed"] += 1
                continue
            if n < 0:
                print(f"{path}: already indexed, skipped")
                totals["skipped"] += 1
            else:
                print(f"{path}: {n} chunks")
                totals["indexed"] += 1
                totals["chunks"] += n

    print(
        f"\nDone. {totals['indexed']} documents indexed "
        f"({totals['chunks']} chunks), {totals['skipped']} skipped, "
        f"{totals['failed']} failed. "
        f"Collection now holds {pipeline.store.count()} chunks."
    )
    if totals["failed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
