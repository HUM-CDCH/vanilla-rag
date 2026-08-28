#!/usr/bin/env python3
"""Ask a question against the indexed documents.

Usage:
    python query.py "What is the main contribution of the paper?"
    python query.py --backend ollama "..."          # local LLM instead of Claude
    python query.py --show-chunks "..."             # print retrieved context too
"""

import argparse
import dataclasses

from vanilla_rag import RAGPipeline
from vanilla_rag.config import DEFAULT_CONFIG
from vanilla_rag.generate import AnthropicGenerator, OllamaGenerator


def main() -> None:
    parser = argparse.ArgumentParser(description="Query the RAG pipeline.")
    parser.add_argument("question", help="The question to answer")
    parser.add_argument(
        "--backend",
        choices=["anthropic", "ollama"],
        default="anthropic",
        help="LLM used for generation (default: anthropic)",
    )
    parser.add_argument(
        "--model", default=None, help="Override the backend's default model name"
    )
    parser.add_argument(
        "--show-chunks", action="store_true", help="Print the retrieved chunks"
    )
    parser.add_argument(
        "--milvus-uri",
        default=None,
        help="Milvus URI to query (default: the main index, milvus_vanilla_rag.db)",
    )
    parser.add_argument(
        "--collection",
        default=None,
        help="Milvus collection to query (default: the main index's collection)",
    )
    args = parser.parse_args()

    cfg = DEFAULT_CONFIG
    if args.milvus_uri is not None or args.collection is not None:
        overrides = {}
        if args.milvus_uri is not None:
            overrides["milvus_uri"] = args.milvus_uri
        if args.collection is not None:
            overrides["collection_name"] = args.collection
        cfg = dataclasses.replace(cfg, **overrides)

    pipeline = RAGPipeline(cfg)

    if args.backend == "anthropic":
        generator = AnthropicGenerator(
            model=args.model or cfg.anthropic_model, max_tokens=cfg.max_answer_tokens
        )
    else:
        generator = OllamaGenerator(
            model=args.model or cfg.ollama_model, host=cfg.ollama_host
        )

    result = pipeline.answer(args.question, generator)

    if args.show_chunks:
        print("=== Retrieved context ===")
        for i, c in enumerate(result["chunks"], 1):
            print(f"\n[{i}] score={c['score']:.4f} source={c['source']}")
            print(c["text"][:500] + ("..." if len(c["text"]) > 500 else ""))
        print("\n=== Answer ===")

    print(result["answer"])

    if result["chunks"]:
        # Rank the source files by their best-scoring chunk.
        files: dict[str, dict] = {}
        for c in result["chunks"]:
            entry = files.setdefault(c["source"], {"best": c["score"], "chunks": 0})
            entry["chunks"] += 1
            entry["best"] = max(entry["best"], c["score"])
        ranked = sorted(files.items(), key=lambda kv: kv[1]["best"], reverse=True)

        print("\n=== Sources used ===")
        for rank, (source, info) in enumerate(ranked, 1):
            print(
                f"{rank}. {source}  "
                f"(best score {info['best']:.4f}, {info['chunks']} chunk(s) in context)"
            )


if __name__ == "__main__":
    main()
