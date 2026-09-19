import json
from pathlib import Path

from hybrid_rag import HybridRAG


def retrieval_hit_rate(rag, test_cases):
    hits = 0

    for case in test_cases:
        results = rag.retrieve(case["question"])
        retrieved_pages = {r["page"] for r in results}

        expected = set(case.get("expected_pages", []))
        if expected & retrieved_pages:
            hits += 1

    return hits / len(test_cases) if test_cases else 0.0


def run():
    test_path = Path("evals/test_cases.json")
    cases = json.loads(test_path.read_text(encoding="utf-8"))

    rag = HybridRAG()
    score = retrieval_hit_rate(rag, cases)

    print(f"Retrieval hit rate: {score:.2%}")
    print("Use these results as a baseline before changing chunking,")
    print("embedding models, graph extraction, or retrieval weights.")


if __name__ == "__main__":
    run()
