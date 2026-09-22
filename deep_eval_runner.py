"""DeepEval runner for the Spotify architecture Hybrid RAG.

Runs the real HybridRAG pipeline against evals/test_cases.json and evaluates:
- Answer Relevancy
- Contextual Precision
- Contextual Recall

Contextual Relevancy and Faithfulness are intentionally disabled in this
fast/local profile. Contextual Relevancy previously hit an OpenAI structured
output length failure during evaluation; removing it keeps the evaluation
stable and reproducible.
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

# Keep DeepEval's per-attempt timeout reasonable without making failed
# evaluations hang for several minutes.
os.environ.setdefault("DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE", "120")

from config import DEEPEVAL_MODEL, EVAL_THRESHOLD, EVAL_MAX_CASES
from hybrid_rag import HybridRAG


EVALUATION_PROFILE = "fast_retrieval_generation"
FAITHFULNESS_ENABLED = False
ASYNC_EVALUATION = False

METRIC_NAMES = [
    "Answer Relevancy",
    "Contextual Precision",
    "Contextual Recall",
]

METRIC_CLASS_TO_KEY = {
    "AnswerRelevancyMetric": "answer_relevancy",
    "ContextualPrecisionMetric": "contextual_precision",
    "ContextualRecallMetric": "contextual_recall",
}


def _load_metrics():
    """Create the three stable DeepEval metrics used by this profile."""
    from deepeval.metrics import (
        AnswerRelevancyMetric,
        ContextualPrecisionMetric,
        ContextualRecallMetric,
    )

    common = {
        "threshold": EVAL_THRESHOLD,
        "model": DEEPEVAL_MODEL,
        "include_reason": False,
        "async_mode": False,
    }

    return [
        AnswerRelevancyMetric(**common),
        ContextualPrecisionMetric(**common),
        ContextualRecallMetric(**common),
    ]


def _safe_score(metric):
    value = getattr(metric, "score", None)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_retrieval_context(value):
    """DeepEval requires retrieval_context to be list[str]."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    return [str(value)]


def _metric_label(metric):
    return metric.__class__.__name__.replace("Metric", "")


def run_evaluation(max_cases=None, progress_callback=None):
    from deepeval.test_case import LLMTestCase

    max_cases = max_cases or EVAL_MAX_CASES
    dataset_path = Path("evals/test_cases.json")
    if not dataset_path.exists():
        raise FileNotFoundError(f"Evaluation dataset not found: {dataset_path}")

    cases = json.loads(dataset_path.read_text(encoding="utf-8"))[:max_cases]
    if not cases:
        raise ValueError("No evaluation cases found in evals/test_cases.json")

    print("=" * 70)
    print("HYBRID RAG - DEEPEVAL")
    print("=" * 70)
    print(f"Evaluation cases: {len(cases)}")
    print(f"Evaluation profile: {EVALUATION_PROFILE}")
    print(f"Faithfulness: {'ENABLED' if FAITHFULNESS_ENABLED else 'DISABLED'}")
    print(f"Async evaluation: {'ENABLED' if ASYNC_EVALUATION else 'DISABLED'}")
    print()

    rag = HybridRAG()
    metrics = _load_metrics()
    rows = []

    # First run the actual RAG once per case. This prevents duplicate model,
    # embedding and Neo4j work when metrics are evaluated.
    test_cases = []
    rag_results = []

    for idx, case in enumerate(cases, start=1):
        question = case["question"]
        print(f"[{idx}/{len(cases)}] Running RAG: {question}")

        result = rag.ask(question)
        actual_output = result.get("answer", "")
        retrieval_context = _as_retrieval_context(
            result.get("retrieval_context", [])
        )

        test_case = LLMTestCase(
            input=question,
            actual_output=actual_output,
            expected_output=case.get("expected_output", ""),
            retrieval_context=retrieval_context,
        )

        test_cases.append(test_case)
        rag_results.append(result)

    print()
    print("Starting DeepEval...")
    print()

    for metric in metrics:
        print(
            f"DeepEval metric: {_metric_label(metric)} "
            f"(model={DEEPEVAL_MODEL}, async_mode=False)"
        )
    print()

    # Evaluate sequentially. This avoids the rate-limit/concurrency problems
    # encountered with the previous async configuration.
    for idx, (case, test_case, result) in enumerate(
        zip(cases, test_cases, rag_results), start=1
    ):
        metric_values = {}
        reasons = {}

        for metric in metrics:
            try:
                metric.measure(test_case)
                score = _safe_score(metric)
                key = METRIC_CLASS_TO_KEY[metric.__class__.__name__]
                metric_values[key] = score
                reasons[key] = getattr(metric, "reason", "") or ""
            except Exception as exc:
                key = METRIC_CLASS_TO_KEY[metric.__class__.__name__]
                metric_values[key] = None
                reasons[key] = f"Metric error: {type(exc).__name__}: {exc}"

        expected_pages = {int(p) for p in case.get("expected_pages", [])}
        retrieved_pages = {
            int(s["page"])
            for s in result.get("sources", [])
            if s.get("page") is not None
        }
        page_hit = bool(expected_pages & retrieved_pages) if expected_pages else True

        passed_metrics = sum(
            1
            for value in metric_values.values()
            if value is not None and value >= EVAL_THRESHOLD
        )
        case_passed = passed_metrics == len(metrics)

        row = {
            "id": case.get("id", f"T{idx:02d}"),
            "category": case.get("category", "General"),
            "question": case["question"],
            "expected_output": case.get("expected_output", ""),
            "actual_output": result.get("answer", ""),
            "expected_pages": sorted(expected_pages),
            "retrieved_pages": sorted(retrieved_pages),
            "page_hit": page_hit,
            "case_passed": case_passed,
            "metrics": metric_values,
            "reasons": reasons,
            "self_rag": result.get("self_rag_review", {}),
            "sources": result.get("sources", []),
        }
        rows.append(row)

        status = "PASS" if case_passed else "FAIL"
        print(
            f"[{idx}/{len(cases)}] {status} | "
            f"Answer={metric_values.get('answer_relevancy', 0):.2f} | "
            f"Precision={metric_values.get('contextual_precision', 0):.2f} | "
            f"Recall={metric_values.get('contextual_recall', 0):.2f}"
        )

        if progress_callback:
            progress_callback(idx, len(cases), row)

    def average(key):
        values = [
            row["metrics"].get(key)
            for row in rows
            if row["metrics"].get(key) is not None
        ]
        return sum(values) / len(values) if values else 0.0

    averages = {
        "answer_relevancy": average("answer_relevancy"),
        "contextual_precision": average("contextual_precision"),
        "contextual_recall": average("contextual_recall"),
    }

    retrieval_score = (
        averages["contextual_precision"]
        + averages["contextual_recall"]
    ) / 2.0

    generation_score = averages["answer_relevancy"]
    overall_score = (generation_score + retrieval_score) / 2.0

    page_hit_rate = (
        sum(1 for row in rows if row["page_hit"]) / len(rows)
        if rows
        else 0.0
    )

    case_pass_rate = (
        sum(1 for row in rows if row["case_passed"]) / len(rows)
        if rows
        else 0.0
    )

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "evaluation_profile": EVALUATION_PROFILE,
        "deepeval_model": DEEPEVAL_MODEL,
        "threshold": EVAL_THRESHOLD,
        "cases_evaluated": len(rows),
        "answer_relevancy": averages["answer_relevancy"],
        "contextual_precision": averages["contextual_precision"],
        "contextual_recall": averages["contextual_recall"],
        "retrieval_score": retrieval_score,
        "generation_score": generation_score,
        "overall_score": overall_score,
        "page_hit_rate": page_hit_rate,
        "case_pass_rate": case_pass_rate,
        "faithfulness_evaluated": False,
        "contextual_relevancy_evaluated": False,
        "grounding_gate_met": retrieval_score >= EVAL_THRESHOLD,
        "target_met": overall_score >= EVAL_THRESHOLD,
        "cases": rows,
    }

    output_path = Path("data/eval_latest.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return report


def print_final_report(report):
    print()
    print("=" * 70)
    print("FINAL EVALUATION RESULT")
    print("=" * 70)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print()
    print(f"TARGET_MET: {report['target_met']}")
    print()
    print("Summary")
    print("-------")
    print(f"Answer Relevancy     : {report['answer_relevancy']:.2%}")
    print(f"Contextual Precision : {report['contextual_precision']:.2%}")
    print(f"Contextual Recall    : {report['contextual_recall']:.2%}")
    print(f"Retrieval Score      : {report['retrieval_score']:.2%}")
    print(f"Generation Score     : {report['generation_score']:.2%}")
    print(f"Overall Score        : {report['overall_score']:.2%}")
    print(f"Page Hit Rate        : {report['page_hit_rate']:.2%}")
    print(f"Case Pass Rate       : {report['case_pass_rate']:.2%}")
    print(f"Target Met           : {report['target_met']}")
    print(f"Saved Report         : data/eval_latest.json")
    print("=" * 70)


def main():
    report = run_evaluation()
    print_final_report(report)


if __name__ == "__main__":
    main()
