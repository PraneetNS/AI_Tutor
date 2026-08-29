"""
Unit tests executing tutor benchmark suite.
"""

import os
from scripts.benchmark_tutor import run_benchmark


def test_golden_dataset_benchmark():
    dataset_path = os.path.join("data", "eval_golden_dataset.json")
    assert os.path.exists(dataset_path)
    
    report = run_benchmark(dataset_path)
    assert report["total_cases"] == 5
    assert report["safety_pass_rate"] >= 80.0
    assert report["latency_p50_ms"] < 2000.0
