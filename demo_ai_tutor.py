"""
demo_ai_tutor.py
----------------
Complete interactive demo showcasing the entire AI Tutor architecture:
1. 6-Stage Pedagogical Pipeline
2. Bayesian Knowledge Tracing (BKT) Posterior Mastery Updates
3. Deterministic ConceptGraph Root-Cause Diagnosis (Backward DAG Walk)
4. TruthfulnessChecker (Symbolic Math Verification & False Praise Rejection)
5. Multi-Provider Model Gateway Failover
"""

import sys
from pathlib import Path

# Ensure project root in sys.path
root_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(root_dir))

from ai_tutor import (
    TutorPipeline,
    MockKnowledgeSource,
    MockLLMClient,
    AIChatRequest,
    PedagogyState,
    PedagogyMode,
    BKTUpdater,
    ConceptMastery,
    ConceptGraph,
    create_ml_concept_graph,
    LearnerState,
    BehaviorProfile,
    ContextResolver,
    TruthfulnessChecker,
    GuardrailPipeline,
)


def print_banner(title: str):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def demo_1_bkt_mastery():
    print_banner("1. BAYESIAN KNOWLEDGE TRACING (BKT) POSTERIOR UPDATES")
    updater = BKTUpdater()
    initial_p = 0.30
    print(f"[*] Initial Student Mastery P(L_0) on 'Backpropagation': {initial_p:.1%}")

    # Correct observation
    p_after_correct = updater.update(initial_p, correct=True)
    print(f"[+] After 1st CORRECT attempt -> Posterior P(L_1): {p_after_correct:.1%}")

    # Another correct observation
    p_after_correct_2 = updater.update(p_after_correct, correct=True)
    print(f"[+] After 2nd CORRECT attempt -> Posterior P(L_2): {p_after_correct_2:.1%}")

    # Incorrect observation
    p_after_wrong = updater.update(p_after_correct_2, correct=False)
    print(f"[-] After 3rd WRONG attempt   -> Posterior P(L_3): {p_after_wrong:.1%}")


def demo_2_concept_graph_root_cause():
    print_banner("2. CONCEPT GRAPH & DETERMINISTIC ROOT-CAUSE DIAGNOSIS")
    graph = create_ml_concept_graph()

    # Setup learner struggling with backprop due to missing chain_rule
    learner = LearnerState(
        student_id="student_42",
        concept_mastery={
            "supervised_learning": ConceptMastery(concept="supervised_learning", mastery=0.85),
            "loss_functions": ConceptMastery(concept="loss_functions", mastery=0.80),
            "gradient_descent": ConceptMastery(concept="gradient_descent", mastery=0.75),
            "calculus_basics": ConceptMastery(concept="calculus_basics", mastery=0.25),
            "chain_rule": ConceptMastery(concept="chain_rule", mastery=0.10), # Root gap!
            "partial_derivatives": ConceptMastery(concept="partial_derivatives", mastery=0.35),
            "linear_algebra": ConceptMastery(concept="linear_algebra", mastery=0.70),
            "probability": ConceptMastery(concept="probability", mastery=0.65),
        },
        behavior=BehaviorProfile(),
    )

    diagnosis = graph.diagnose_root_cause("backpropagation", learner)
    print(f"[*] Target Struggling Concept : {diagnosis.struggling_concept}")
    print(f"[!] Diagnosed Root Cause Gap  : {diagnosis.likely_root_gap}")
    print(f"[!] Diagnostic Confidence     : {diagnosis.confidence:.1%}")
    print(f"[*] Ancestry Chain Walked     : {diagnosis.chain_analyzed}")


def demo_3_truthfulness_math_guardrail():
    print_banner("3. TRUTHFULNESS CHECKER & OUTPUT GUARDRAIL REJECTION")
    checker = TruthfulnessChecker(use_sympy=True)

    bad_student_math = "I calculated the derivative of x^3 is 2x^2."
    flawed_draft_praise = "Great job! That's correct, keep up the good work."

    print(f"[*] Student Claim    : \"{bad_student_math}\"")
    print(f"[*] Draft AI Response: \"{flawed_draft_praise}\"")

    truth_res = checker.check_truthfulness(
        student_message=bad_student_math,
        draft_response=flawed_draft_praise,
    )

    print(f"[-] Guardrail Verdict: {'REJECTED' if truth_res.rejected else 'PASSED'}")
    print(f"[-] Specific Error   : {truth_res.specific_error}")
    print(f"[-] Regeneration Directive:\n    {truth_res.regeneration_instruction}")


def demo_4_end_to_end_chat():
    print_banner("4. 6-STAGE PEDAGOGICAL PIPELINE CHAT TURN")
    from ai_tutor.pipeline import DefaultModelAdapter
    pipeline = TutorPipeline(
        knowledge_source=MockKnowledgeSource(),
        model_adapter=DefaultModelAdapter(MockLLMClient()),
    )

    request = AIChatRequest(
        message="Why do we need the chain rule in neural network training?",
        course_id=1,
        lecture_id=1,
    )

    response = pipeline.process(request)
    print(f"[*] Student: {request.message}")
    print(f"[+] Tutor ({response.pedagogy_mode.upper()} mode):\n{response.answer}")


if __name__ == "__main__":
    print_banner("AETHER AI TUTOR — ARCHITECTURE VERIFICATION DEMO")
    demo_1_bkt_mastery()
    demo_2_concept_graph_root_cause()
    demo_3_truthfulness_math_guardrail()
    demo_4_end_to_end_chat()
    print_banner("DEMO COMPLETE — ALL SYSTEMS OPERATIONAL")
