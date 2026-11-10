#!/usr/bin/env python3
"""
Retrieval evaluation.

The golden set lives in data/golden_set.csv with columns `question,chunk_id`, where
chunk_id is the DocumentChunk.chunk_id (md5 of source file name + chunk text) of the
chunk that answers the question. For each question we retrieve the top-k chunks and
check whether the expected chunk_id is among them.

NOTE: chunk_ids depend on the chunking settings and file contents. If you change
chunk_size/chunk_overlap or edit the source documents, the IDs in the CSV go stale
and must be regenerated.

Metrics (one relevant chunk per question, so recall@k == hit@k):
  - Hit@k : fraction of questions whose expected chunk appears in the top-k
  - MRR   : mean of 1/rank of the expected chunk (0 if it is not retrieved)

Modes:
  offline : chunk the markdown files and embed them in memory with embedding.py
            (LiteLLM + Ollama). No database needed.
  db      : query the `document_chunks` table in Postgres (pgvector), embedding the
            question with the same function ingestion.py used for the stored vectors.

Usage:
  python evals/eval_retrieval.py --mode offline --k 5
  python evals/eval_retrieval.py --mode db --k 5 --verbose
"""

import argparse
import asyncio
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List

ROOT = Path(__file__).resolve().parent.parent
# Make project-level modules (embedding.py, ingestion.py) importable when run from evals/.
sys.path.insert(0, str(ROOT))

RAW_DIR = ROOT / "data" / "raw" / "hr-manual"
GOLDEN_CSV = ROOT / "data" / "golden_set.csv"
CORPUS_FILES = [
    RAW_DIR / "markdown" / "manual.md",
    RAW_DIR / "markdown" / "tools.md",
    RAW_DIR / "platform-how-to-guides.md",
]


@dataclass
class EvalCase:
    question: str
    chunk_id: str  # ID of the chunk that answers the question


# A retriever takes (question, k) and returns the top-k chunk IDs, best first.
Retriever = Callable[[str, int], List[str]]


def load_golden_set(path: Path) -> List[EvalCase]:
    with open(path, newline="", encoding="utf-8") as f:
        return [EvalCase(row["question"], row["chunk_id"]) for row in csv.DictReader(f)]


def rank_of(retrieved_ids: List[str], expected_id: str) -> int:
    """1-based rank of the expected chunk, or 0 if it was not retrieved."""
    return retrieved_ids.index(expected_id) + 1 if expected_id in retrieved_ids else 0


def evaluate(retrieve: Retriever, cases: List[EvalCase], k: int, verbose: bool = False) -> Dict[str, float]:
    hits = 0
    reciprocal_ranks = []
    for case in cases:
        rank = rank_of(retrieve(case.question, k), case.chunk_id)
        hits += rank > 0
        reciprocal_ranks.append(1 / rank if rank else 0.0)
        if verbose or not rank:
            status = f"rank {rank}" if rank else "MISS"
            print(f"  [{status:>6}] {case.question}")
    return {
        "cases": len(cases),
        f"hit@{k}": hits / len(cases),
        "mrr": sum(reciprocal_ranks) / len(cases),
    }


# --------------------------------------------------------------------------- offline
def build_offline_retriever() -> Retriever:
    from embedding import embed_query, embed_texts
    from ingestion import parse_markdown_file

    chunks = []
    for path in CORPUS_FILES:
        chunks += parse_markdown_file(path)
    texts = [c.page_content for c in chunks]
    ids = [c.chunk_id for c in chunks]
    print(f"Embedding {len(texts)} chunks from {len(CORPUS_FILES)} files...")

    vectors: List[List[float]] = []
    for i in range(0, len(texts), 32):
        vectors += embed_texts(texts[i:i + 32])
    norms = [math.sqrt(sum(x * x for x in v)) for v in vectors]

    def retrieve(question: str, k: int) -> List[str]:
        q = embed_query(question)
        qn = math.sqrt(sum(x * x for x in q))
        scores = [
            sum(a * b for a, b in zip(q, v)) / (qn * n)
            for v, n in zip(vectors, norms)
        ]
        top = sorted(range(len(ids)), key=scores.__getitem__, reverse=True)[:k]
        return [ids[i] for i in top]

    return retrieve


# ------------------------------------------------------------------------------ db
async def run_db_eval(cases: List[EvalCase], k: int, verbose: bool) -> Dict[str, float]:
    # Imported lazily so offline mode doesn't need a database or asyncpg set up.
    from ingestion import db, get_embedding

    await db.connect()
    try:
        # Embed all questions first (sync LiteLLM calls), then query the DB.
        vectors = {c.question: get_embedding(c.question) for c in cases}
        results: Dict[str, List[str]] = {}
        async with db.pool.acquire() as conn:
            for question, vec in vectors.items():
                vec_str = "[" + ",".join(map(str, vec)) + "]"
                rows = await conn.fetch(
                    "SELECT id FROM document_chunks ORDER BY embedding <=> $1::vector LIMIT $2",
                    vec_str, k,
                )
                results[question] = [r["id"] for r in rows]
    finally:
        await db.disconnect()

    return evaluate(lambda q, _k: results[q], cases, k, verbose)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval quality on a golden question set.")
    parser.add_argument("--mode", choices=["offline", "db"], default="offline")
    parser.add_argument("--golden", type=Path, default=GOLDEN_CSV, help="path to the golden set CSV")
    parser.add_argument("--k", type=int, default=5, help="number of chunks to retrieve per question")
    parser.add_argument("--verbose", action="store_true", help="print the rank for every question, not just misses")
    args = parser.parse_args()

    cases = load_golden_set(args.golden)
    if args.mode == "db":
        metrics = asyncio.run(run_db_eval(cases, args.k, args.verbose))
    else:
        metrics = evaluate(build_offline_retriever(), cases, args.k, args.verbose)

    print("\nResults")
    print("-" * 30)
    for name, value in metrics.items():
        print(f"{name:>8}: {value:.3f}" if isinstance(value, float) else f"{name:>8}: {value}")


if __name__ == "__main__":
    main()
