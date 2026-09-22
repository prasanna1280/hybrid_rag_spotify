"""Compatibility CLI for retrieval baseline and DeepEval."""

import argparse
import json
from pathlib import Path

from hybrid_rag import HybridRAG


def retrieval_hit_rate(rag, test_cases):
    hits = 0
    for case in test_cases:
        results = rag.retrieve(case["question"])
        retrieved_pages = {r["page"] for r in results}
        expected = set(case.get("expected_pages", []))
        if not expected or expected & retrieved_pages:
            hits += 1
    return hits / len(test_cases) if test_cases else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--deep", action="store_true", help="Run DeepEval RAG metrics")
    parser.add_argument("--cases", type=int, default=None)
    args = parser.parse_args()

    if args.deep:
        from deep_eval_runner import run_evaluation
        report = run_evaluation(args.cases)
        print(json.dumps(report["scores"], indent=2))
        print("TARGET_MET:", report["target_met"])
        return

    cases = json.loads(Path("evals/test_cases.json").read_text(encoding="utf-8"))
    rag = HybridRAG()
    score = retrieval_hit_rate(rag, cases)
    print(f"Retrieval page hit rate: {score:.2%}")


if __name__ == "__main__":
    main()
