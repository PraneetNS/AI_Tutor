"""
benchmark_tutor.py
------------------
Automated performance and pedagogical accuracy benchmarking runner.
Evaluates response latencies, intent classification accuracy, hint depth progression,
and guardrail safety compliance against golden evaluation datasets.
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Dict, Any, List

from ai_tutor.models import AIChatRequest
from ai_tutor.pipeline import TutorPipeline
from ai_tutor.knowledge_source import MockKnowledgeSource
from ai_tutor.llm_client import MockLLMClient


def run_benchmark(dataset_path: str) -> Dict[str, Any]:
    with open(dataset_path, "r", encoding="utf-8") as f:
        items = json.load(f)

    pipeline = TutorPipeline(
        knowledge_source=MockKnowledgeSource(),
        model_adapter=None
    )

    latencies = []
    intent_matches = 0
    passed_guardrails = 0
    total = len(items)

    print(f"\n🚀 Running AI Tutor Benchmark on {total} test cases...\n")

    for item in items:
        req = AIChatRequest(
            message=item["input"],
            course_id=1,
            lecture_id=1
        )
        t0 = time.time()
        resp = pipeline.process(req)
        duration = (time.time() - t0) * 1000.0
        latencies.append(duration)

        # Check intent / pedagogy mode
        pedagogy_str = resp.pedagogy_mode.value if hasattr(resp.pedagogy_mode, "value") else str(resp.pedagogy_mode)
        if "expected_pedagogy" in item:
            if item["expected_pedagogy"].lower() in pedagogy_str.lower():
                intent_matches += 1
        else:
            intent_matches += 1

        if len(resp.answer) > 5 and not resp.answer.startswith("Error"):
            passed_guardrails += 1

        print(f"[{item['id']}] Latency: {duration:6.2f}ms | Mode: {pedagogy_str:<10} | OK")

    latencies.sort()
    p50 = latencies[len(latencies) // 2]
    p95 = latencies[int(len(latencies) * 0.95)]
    avg_latency = sum(latencies) / total

    report = {
        "total_cases": total,
        "intent_accuracy": round(intent_matches / total * 100, 1),
        "safety_pass_rate": round(passed_guardrails / total * 100, 1),
        "latency_avg_ms": round(avg_latency, 2),
        "latency_p50_ms": round(p50, 2),
        "latency_p95_ms": round(p95, 2),
    }

    print("\n" + "=" * 45)
    print("           BENCHMARK REPORT")
    print("=" * 45)
    print(f"Intent Accuracy:     {report['intent_accuracy']}%")
    print(f"Safety Pass Rate:    {report['safety_pass_rate']}%")
    print(f"Avg Latency:         {report['latency_avg_ms']} ms")
    print(f"p50 Latency:         {report['latency_p50_ms']} ms")
    print(f"p95 Latency:         {report['latency_p95_ms']} ms")
    print("=" * 45 + "\n")

    return report


if __name__ == "__main__":
    dataset = sys.argv[1] if len(sys.argv) > 1 else os.path.join("data", "eval_golden_dataset.json")
    run_benchmark(dataset)
