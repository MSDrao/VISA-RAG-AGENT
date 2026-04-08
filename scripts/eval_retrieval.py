"""
scripts/eval_retrieval.py

Run retrieval-only evaluation against a starter question set.

Usage:
  python scripts/eval_retrieval.py
  python scripts/eval_retrieval.py --limit 10
"""

import argparse
import json
import os
import sys

from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrieval.query_processor import QueryProcessor
from retrieval.retriever import HybridRetriever
from retrieval.reranker import CrossEncoderReranker


def load_questions(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def format_chunk(chunk) -> str:
    return (
        f"{chunk.source_id} | score={chunk.score:.3f} | "
        f"{chunk.source_url}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default="data/eval/questions.json")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    questions = load_questions(args.path)
    if args.limit:
        questions = questions[:args.limit]

    qp = QueryProcessor()
    retriever = HybridRetriever()
    reranker = CrossEncoderReranker()

    hits = 0

    for idx, item in enumerate(questions, start=1):
        question = item["question"]
        expects = item.get("expected_url_keywords", [])
        official_only = item.get("official_sources_only", False)

        processed = qp.process(
            question=question,
            conversation_history=None,
            visa_type_filter=None,
            official_sources_only=official_only,
        )

        retrieved = retriever.retrieve(
            query=processed.original,
            expanded_query=processed.expanded,
            metadata_filters=processed.metadata_filters if processed.metadata_filters else None,
            query_agencies=processed.detected_agencies,
            query_forms=processed.detected_forms,
            query_process_terms=processed.detected_process_terms,
        )
        reranked = reranker.rerank(processed.original, retrieved)

        found = False
        if expects:
            for chunk in reranked[:5]:
                if any(keyword in chunk.source_url for keyword in expects):
                    found = True
                    break
        else:
            found = len(reranked) > 0

        hits += int(found)

        print("=" * 80)
        print(f"{idx}. {question}")
        print(f"expected_url_keywords={expects}")
        print(f"result={'HIT' if found else 'MISS'}")
        print("top reranked:")
        for chunk in reranked[:5]:
            print(f"  - {format_chunk(chunk)}")

    print("=" * 80)
    print(f"Eval complete: {hits}/{len(questions)} hits in top 5")


if __name__ == "__main__":
    main()
