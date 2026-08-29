"""
test_strategy_engine.py
-----------------------
Unit and integration tests for StrategyEngine:
- Taxonomy selection
- Empirical tie breaking via learner_strategy_effectiveness ratio
- Assess step feedback loop updates
- Multi-user and multi-domain isolation
"""

import pytest
from ai_tutor.strategy_engine import (
    StrategyEngine,
    InMemoryStrategyEffectivenessStore,
    STRATEGY_TAXONOMY
)
from ai_tutor.models import (
    LearnerState,
    ConceptMastery,
    Misconception,
    StrategyAction
)
from ai_tutor.concept_graph import create_ml_concept_graph


def test_default_taxonomy_fallback_order_no_history():
    """When a student has no history, strategy selection falls back to default taxonomy order."""
    store = InMemoryStrategyEffectivenessStore()
    engine = StrategyEngine(store=store)

    # For explanation category, default order is ["analogy", "visual", "worked_example", "first_principles"]
    chosen_type = engine.resolve_strategy_type_tie(
        user_id=101,
        strategy_category="explanation",
        concept_domain="machine_learning"
    )
    assert chosen_type == "analogy"

    # For scaffolding category, default order is ["leading_question", "subgoal_breakdown", ...]
    chosen_scaff = engine.resolve_strategy_type_tie(
        user_id=101,
        strategy_category="scaffolding",
        concept_domain="machine_learning"
    )
    assert chosen_scaff == "leading_question"


def test_tie_breaking_prefers_highest_mastery_ratio():
    """Prefers whichever strategy_type has the highest (times_led_to_mastery / times_used) ratio."""
    store = InMemoryStrategyEffectivenessStore()
    engine = StrategyEngine(store=store)

    user_id = 42
    domain = "linear_algebra"
    category = "explanation"

    # Simulate past history for user 42:
    # 1. 'analogy': used 10 times, led to mastery 3 times (ratio = 0.30)
    for _ in range(3):
        engine.record_assess_result(user_id, category, "analogy", domain, assess_result="mastered")
    for _ in range(7):
        engine.record_assess_result(user_id, category, "analogy", domain, assess_result="failed")

    # 2. 'visual': used 4 times, led to mastery 3 times (ratio = 0.75)
    for _ in range(3):
        engine.record_assess_result(user_id, category, "visual", domain, assess_result="mastered")
    engine.record_assess_result(user_id, category, "visual", domain, assess_result="failed")

    # 3. 'worked_example': used 2 times, led to mastery 0 times (ratio = 0.0)
    for _ in range(2):
        engine.record_assess_result(user_id, category, "worked_example", domain, assess_result="failed")

    # Plan should break tie in favor of 'visual' (0.75 > 0.30)
    chosen_type = engine.resolve_strategy_type_tie(
        user_id=user_id,
        strategy_category=category,
        concept_domain=domain
    )
    assert chosen_type == "visual"


def test_assess_step_updates_times_used_and_times_led_to_mastery():
    """After every Assess step, increment times_used, and times_led_to_mastery if assess result was 'mastered'."""
    store = InMemoryStrategyEffectivenessStore()
    engine = StrategyEngine(store=store)

    user_id = 999
    cat = "scaffolding"
    stype = "formula_nudge"
    domain = "python_programming"

    # Initial state
    rec0 = store.get_effectiveness(user_id, cat, stype, domain)
    assert rec0 is None

    # Attempt 1: assess result is NOT mastered (e.g. incorrect or in_progress)
    rec1 = engine.record_assess_result(user_id, cat, stype, domain, assess_result="in_progress")
    assert rec1.times_used == 1
    assert rec1.times_led_to_mastery == 0
    assert rec1.effectiveness_ratio == 0.0

    # Attempt 2: assess result IS mastered
    rec2 = engine.record_assess_result(user_id, cat, stype, domain, assess_result="mastered")
    assert rec2.times_used == 2
    assert rec2.times_led_to_mastery == 1
    assert rec2.effectiveness_ratio == 0.5

    # Attempt 3: boolean True assess result
    rec3 = engine.record_assess_result(user_id, cat, stype, domain, assess_result=True)
    assert rec3.times_used == 3
    assert rec3.times_led_to_mastery == 2
    assert pytest.approx(rec3.effectiveness_ratio, 0.01) == 0.67


def test_user_and_domain_isolation():
    """Effectiveness records are strictly isolated across different users and subject domains."""
    store = InMemoryStrategyEffectivenessStore()
    engine = StrategyEngine(store=store)

    # User A in Machine Learning loves 'analogy'
    engine.record_assess_result(1001, "explanation", "analogy", "machine_learning", assess_result="mastered")
    # User B in Machine Learning loves 'first_principles'
    engine.record_assess_result(1002, "explanation", "first_principles", "machine_learning", assess_result="mastered")
    # User A in Calculus loves 'visual'
    engine.record_assess_result(1001, "explanation", "visual", "calculus", assess_result="mastered")

    assert engine.resolve_strategy_type_tie(1001, "explanation", "machine_learning") == "analogy"
    assert engine.resolve_strategy_type_tie(1002, "explanation", "machine_learning") == "first_principles"
    assert engine.resolve_strategy_type_tie(1001, "explanation", "calculus") == "visual"


def test_plan_with_learner_state_and_concept_graph():
    """StrategyEngine.plan integrates LearnerState, ConceptGraph, and tie-breaking."""
    store = InMemoryStrategyEffectivenessStore()
    engine = StrategyEngine(store=store)
    graph = create_ml_concept_graph()

    # User 500 performs better with 'worked_example' for explanations
    for _ in range(5):
        engine.record_assess_result(500, "explanation", "worked_example", "machine_learning", assess_result="mastered")

    # All prerequisite ancestors are mastered
    learner_state = LearnerState(
        student_id="500",
        concept_mastery={
            "variables": ConceptMastery(concept="variables", mastery=0.95),
            "expressions": ConceptMastery(concept="expressions", mastery=0.95),
            "functions": ConceptMastery(concept="functions", mastery=0.95),
            "linear_algebra": ConceptMastery(concept="linear_algebra", mastery=0.95),
            "calculus_basics": ConceptMastery(concept="calculus_basics", mastery=0.95),
            "chain_rule": ConceptMastery(concept="chain_rule", mastery=0.95),
            "partial_derivatives": ConceptMastery(concept="partial_derivatives", mastery=0.95),
            "probability": ConceptMastery(concept="probability", mastery=0.95),
            "supervised_learning": ConceptMastery(concept="supervised_learning", mastery=0.95),
            "loss_functions": ConceptMastery(concept="loss_functions", mastery=0.95),
            "gradient_descent": ConceptMastery(concept="gradient_descent", mastery=0.95),
            "backpropagation": ConceptMastery(concept="backpropagation", mastery=0.2, attempts=3, correct=0)
        }
    )


    # Plan when student has 2 consecutive failures and prerequisites are satisfied -> triggers 'explanation'
    plan = engine.plan(
        learner_state=learner_state,
        concept_graph=graph,
        target_concept="backpropagation",
        concept_domain="machine_learning",
        user_id=500,
        consecutive_failures=2
    )

    assert plan.strategy_category == "explanation"
    assert plan.strategy_type == "worked_example"
    assert plan.recommendation == StrategyAction.EXPLAIN
    assert plan.target_concept == "backpropagation"
    assert plan.curriculum_position is not None


def test_plan_triggers_remediation_on_root_gap():
    """When a root prerequisite gap is detected, StrategyEngine plans remediation/prerequisite_review."""
    store = InMemoryStrategyEffectivenessStore()
    engine = StrategyEngine(store=store)
    graph = create_ml_concept_graph()

    # Learner struggling with backpropagation, and has not mastered chain_rule
    learner_state = LearnerState(
        student_id="700",
        concept_mastery={
            "chain_rule": ConceptMastery(concept="chain_rule", mastery=0.10),
            "backpropagation": ConceptMastery(concept="backpropagation", mastery=0.20)
        }
    )

    plan = engine.plan(
        learner_state=learner_state,
        concept_graph=graph,
        target_concept="backpropagation",
        concept_domain="machine_learning",
        user_id=700,
        consecutive_failures=1
    )

    assert plan.strategy_category == "remediation"
    assert plan.strategy_type == "prerequisite_review"
    assert plan.root_cause_diagnosis is not None

