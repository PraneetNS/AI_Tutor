"""
generate_coverage_summary.py
----------------------------
Parses pytest-cov json report and outputs a markdown summary table for CI workflows.
"""

import json
import sys
import os


def generate_summary(json_path: str = "coverage.json"):
    if not os.path.exists(json_path):
        print(f"Coverage file {json_path} not found. Skipping summary.")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    totals = data.get("totals", {})
    percent_covered = totals.get("percent_covered", 0.0)

    print("\n" + "=" * 45)
    print("        TEST COVERAGE SUMMARY")
    print("=" * 45)
    print(f"Total Statements: {totals.get('num_statements', 0)}")
    print(f"Covered Lines:    {totals.get('covered_lines', 0)}")
    print(f"Missing Lines:    {totals.get('missing_lines', 0)}")
    print(f"Coverage Percent: {percent_covered:.2f}%")
    print("=" * 45 + "\n")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "coverage.json"
    generate_summary(path)
