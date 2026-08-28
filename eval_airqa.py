#!/usr/bin/env python3
"""Run the vanilla RAG pipeline over an AirQA question file.

Produces a `result.jsonl` ({"uuid": ..., "answer": ...} per line, same order
as the gold file) that AirQA's `utils/eval_utils.py` can score:

    python eval_airqa.py --gold gold.jsonl --corpus papers/ --out results/run1
    cd AirQA && python utils/eval_utils.py --gold gold.jsonl --dir results/run1

The corpus is indexed into its own Milvus collection (default: a separate
database file) so the project's main index is left untouched. Runs are
resumable: questions whose uuid already appears in result.jsonl are skipped.

Pass --skip-ingest to bypass the corpus scan entirely when the target
collection is already fully indexed (the scan is a sequential per-file dedup
check and gets expensive against a large, already-indexed corpus).
"""

import argparse
import dataclasses
import json
from pathlib import Path

from vanilla_rag import RAGPipeline
from vanilla_rag.config import DEFAULT_CONFIG
from vanilla_rag.generate import AnthropicGenerator, OllamaGenerator


def load_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", required=True, help="AirQA-format question file (jsonl)")
    parser.add_argument("--corpus", default=None, help="Folder of papers to index")
    parser.add_argument(
        "--skip-ingest",
        action="store_true",
        help="Skip the corpus scan (use when the target collection is already fully indexed)",
    )
    parser.add_argument("--out", required=True, help="Output dir for result.jsonl")
    parser.add_argument("--backend", choices=["anthropic", "ollama"], default="anthropic")
    parser.add_argument("--model", default=None, help="Override the backend's default model")
    parser.add_argument(
        "--milvus-uri",
        default="milvus_airqa_eval.db",
        help="Milvus URI for the eval index (default: a separate local db file)",
    )
    parser.add_argument("--collection", default="airqa_eval")
    args = parser.parse_args()
    if not args.skip_ingest and not args.corpus:
        parser.error("--corpus is required unless --skip-ingest is set")

    cfg = dataclasses.replace(
        DEFAULT_CONFIG, milvus_uri=args.milvus_uri, collection_name=args.collection
    )
    pipeline = RAGPipeline(cfg)

    if args.skip_ingest:
        print("=== Skipping corpus scan (--skip-ingest); using collection as-is ===")
    else:
        print(f"=== Indexing corpus: {args.corpus} ===")
        summary = pipeline.ingest_collection(args.corpus)
        print(f"Ingest summary: {summary}")

    if args.backend == "anthropic":
        generator = AnthropicGenerator(
            model=args.model or cfg.anthropic_model, max_tokens=cfg.max_answer_tokens
        )
    else:
        generator = OllamaGenerator(
            model=args.model or cfg.ollama_model, host=cfg.ollama_host
        )

    gold = load_jsonl(args.gold)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    result_path = out_dir / "result.jsonl"
    done = {r["uuid"]: r["answer"] for r in load_jsonl(result_path)} if result_path.exists() else {}
    if done:
        print(f"Resuming: {len(done)} answers already present")

    answers: dict[str, str] = {}
    for i, ex in enumerate(gold, 1):
        uuid = ex["uuid"]
        if uuid in done:
            answers[uuid] = done[uuid]
            continue
        # AirQA answers are graded against a required format, so the format
        # instruction is part of the task input.
        query = (
            f"{ex['question']}\n\n{ex['answer_format']}\n\n"
            "Output only the final answer in the required format, "
            "with no citations or extra commentary."
        )
        try:
            answer = pipeline.answer(query, generator)["answer"]
        except Exception as exc:  # keep the run going; scored as 0 by AirQA
            answer = f"[ERROR]: {exc}"
        answers[uuid] = answer
        print(f"[{i}/{len(gold)}] {uuid}: {answer[:80]!r}")
        # Rewrite the full file in gold order after every answer (resumable).
        with open(result_path, "w", encoding="utf-8") as f:
            for e in gold:
                if e["uuid"] in answers or e["uuid"] in done:
                    a = answers.get(e["uuid"], done.get(e["uuid"]))
                    f.write(json.dumps({"uuid": e["uuid"], "answer": a}, ensure_ascii=False) + "\n")

    print(f"Wrote {result_path} ({len(answers)} answers)")


if __name__ == "__main__":
    main()
