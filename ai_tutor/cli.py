"""
cli.py
------
Antigravity AI Tutor Command-Line Interface.
Supports interactive tutoring, curriculum graph inspection & export, BKT simulations,
and guardrail safety evaluation.
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from .concept_graph import create_ml_concept_graph
from .exporter import CurriculumExporter
from .learner_model import BKTUpdater
from .guardrails import ResponseGuardrail
from .pipeline import TutorPipeline
from .knowledge_source import MockKnowledgeSource
from .llm_client import MockLLMClient
from .models import AIChatRequest, PedagogyState, Role


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ai-tutor",
        description="Antigravity AI Tutor CLI: Socratic Tutoring, Graph Inspection & BKT Simulation"
    )
    subparsers = parser.add_subparsers(dest="subcommand", help="Available subcommands")

    # 1. graph-info
    p_graph = subparsers.add_parser("graph-info", help="Display summary statistics of the curriculum DAG")
    p_graph.add_argument("--domain", default="machine_learning", help="Domain to inspect")

    # 2. export-graph
    p_export = subparsers.add_parser("export-graph", help="Export curriculum graph to JSON or DOT")
    p_export.add_argument("--format", choices=["json", "dot", "cytoscape"], default="json")
    p_export.add_argument("--output", help="Optional output file path")

    # 3. simulate-bkt
    p_bkt = subparsers.add_parser("simulate-bkt", help="Simulate Bayesian Knowledge Tracing transitions")
    p_bkt.add_argument("--p-l0", type=float, default=0.30, help="Initial prior mastery P(L0)")
    p_bkt.add_argument("--answers", required=True, help="Comma-separated sequence of 1 (correct) or 0 (incorrect)")

    # 4. eval-guardrail
    p_guard = subparsers.add_parser("eval-guardrail", help="Evaluate text safety through the guardrail filter")
    p_guard.add_argument("--text", required=True, help="Response text to inspect")

    # 5. ask
    p_ask = subparsers.add_parser("ask", help="Query the AI tutor with a single question")
    p_ask.add_argument("--question", required=True, help="Question text")
    p_ask.add_argument("--course-id", type=int, default=1)
    p_ask.add_argument("--lecture-id", type=int, default=1)

    return parser


def run_graph_info(args: argparse.Namespace) -> int:
    graph = create_ml_concept_graph()
    concepts = graph.store.all_concepts()
    print(f"=== Curriculum Graph: {args.domain} ===")
    print(f"Total Concepts: {len(concepts)}")
    for c in concepts[:5]:
        prereqs = [p.name for p in graph.get_direct_prerequisites(c.concept_id)]
        prereq_str = f" (Prereqs: {', '.join(prereqs)})" if prereqs else " (Root)"
        print(f" - [{c.concept_id}] {c.name}{prereq_str}")
    if len(concepts) > 5:
        print(f" ... and {len(concepts) - 5} more concepts.")
    return 0


def run_export_graph(args: argparse.Namespace) -> int:
    graph = create_ml_concept_graph()
    if args.format == "dot":
        out = CurriculumExporter.to_dot(graph)
    elif args.format == "cytoscape":
        import json
        out = json.dumps(CurriculumExporter.to_cytoscape(graph), indent=2)
    else:
        out = CurriculumExporter.to_json(graph)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(out)
        print(f"Exported graph to {args.output}")
    else:
        print(out)
    return 0


def run_simulate_bkt(args: argparse.Namespace) -> int:
    updater = BKTUpdater()
    answers = [x.strip() in ("1", "true", "correct", "t") for x in args.answers.split(",")]
    p_known = args.p_l0

    print(f"Initial P(L0): {p_known:.4f}")
    print("-" * 35)
    for idx, ans in enumerate(answers, start=1):
        p_known = updater.update(prior_mastery=p_known, correct=ans, hints_used=0)
        status = "Correct" if ans else "Incorrect"
        print(f"Step {idx:02d} [{status:<9}]: P(L) = {p_known:.4f}")
    return 0


def run_eval_guardrail(args: argparse.Namespace) -> int:
    guardrail = ResponseGuardrail()
    pedagogy_state = PedagogyState()
    req = AIChatRequest(message="Test student question")
    result = guardrail.validate_and_sanitize(
        raw_answer=args.text,
        pedagogy_state=pedagogy_state,
        request=req
    )
    print(f"Safe: {result.is_safe}")
    print(f"Sanitized Answer: {result.sanitized_answer}")
    if result.flags:
        print(f"Flags: {', '.join(result.flags)}")
    return 0



def run_ask(args: argparse.Namespace) -> int:
    pipeline = TutorPipeline(
        knowledge_source=MockKnowledgeSource(),
        model_adapter=None
    )
    req = AIChatRequest(
        message=args.question,
        course_id=args.course_id,
        lecture_id=args.lecture_id
    )
    resp = pipeline.process(req)
    print("=== AI Tutor Response ===")
    print(f"Answer: {resp.answer}")
    print(f"Pedagogy Mode: {resp.pedagogy_mode}")
    print(f"Hint Level: {resp.hint_level}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.subcommand:
        parser.print_help()
        return 1

    handlers = {
        "graph-info": run_graph_info,
        "export-graph": run_export_graph,
        "simulate-bkt": run_simulate_bkt,
        "eval-guardrail": run_eval_guardrail,
        "ask": run_ask,
    }

    handler = handlers.get(args.subcommand)
    if handler:
        return handler(args)
    return 1


if __name__ == "__main__":
    sys.exit(main())
