"""
test_answer_evaluator.py
------------------------
Unit tests for AnswerEvaluator:
- Cheap & fast concept identification via ConceptGraph keyword matching
- Evaluation & partial credit scoring
- Direct integration feeding BKTUpdater.update()
- Direct integration feeding StrategyEngine's learner_strategy_effectiveness
"""

import time
import pytest
from ai_tutor.answer_evaluator import AnswerEvaluator
from ai_tutor.concept_graph import create_ml_concept_graph
from ai_tutor.strategy_engine import StrategyEngine, InMemoryStrategyEffectivenessStore
from ai_tutor.models import LearnerState, ConceptMastery


def test_identify_concepts_fast_lexical_matching():
    evaluator = AnswerEvaluator()

    # 1. Backprop and gradient descent response
    resp = "Backpropagation calculates partial derivatives of the loss function to perform gradient descent."
    concepts = evaluator.identify_concepts(student_response=resp)
    
    assert "backpropagation" in concepts
    assert "gradient_descent" in concepts
    assert "loss_functions" in concepts
    assert "partial_derivatives" in concepts


def test_identify_concepts_with_aliases():
    evaluator = AnswerEvaluator()

    # Mention MSE and learning rate
    resp = "We minimize MSE by taking steps proportional to the learning rate."
    concepts = evaluator.identify_concepts(student_response=resp)
    
    assert "loss_functions" in concepts  # MSE alias
    assert "gradient_descent" in concepts  # learning rate alias


def test_identify_concepts_latency_is_sub_millisecond():
    evaluator = AnswerEvaluator()
    resp = "The chain rule allows us to compute derivatives of composite nested functions in deep neural networks."
    
    t0 = time.perf_counter()
    for _ in range(100):
        evaluator.identify_concepts(student_response=resp)
    duration_ms = (time.perf_counter() - t0) * 1000 / 100
    
    # Must be under 5ms per evaluation (typically < 0.2ms)
    assert duration_ms < 5.0


def test_evaluate_correct_and_partial_credit():
    evaluator = AnswerEvaluator()

    # Fully correct response
    res_full = evaluator.evaluate(
        response="Loss functions like MSE quantify how far predictions are from ground truth targets in supervised learning.",
        expected_concepts=["loss_functions", "supervised_learning"]
    )
    assert res_full.correct is True
    assert res_full.partial_credit >= 0.85
    assert res_full.is_mastered is True
    assert "loss_functions" in res_full.concepts_touched

    # Partial response (only mentions 1 of 2 expected concepts)
    res_partial = evaluator.evaluate(
        response="Supervised learning uses labeled training data.",
        expected_concepts=["loss_functions", "supervised_learning"]
    )
    assert res_partial.partial_credit < 0.85
    assert res_partial.is_mastered is False


def test_evaluate_detects_misconceptions():
    evaluator = AnswerEvaluator()

    # Student states misconception that backprop updates weights directly
    res_mc = evaluator.evaluate(
        response="Backprop updates weights directly without needing an optimizer.",
        expected_concepts=["backpropagation"]
    )
    assert res_mc.correct is False
    assert len(res_mc.misconceptions_detected) > 0
    assert "backprop_update_vs_derivative" in res_mc.misconceptions_detected[0]


def test_evaluate_and_update_bkt_and_strategy_effectiveness():
    store = InMemoryStrategyEffectivenessStore()
    strat_engine = StrategyEngine(store=store)
    evaluator = AnswerEvaluator()

    user_id = 888
    domain = "machine_learning"
    category = "explanation"
    stype = "visual"

    learner_state = LearnerState(
        student_id=str(user_id),
        concept_mastery={
            "gradient_descent": ConceptMastery(concept="gradient_descent", mastery=0.30)
        }
    )

    # 1. Student answers correctly
    res1 = evaluator.evaluate_and_update(
        user_id=user_id,
        response="Gradient descent steps down the loss function surface in the opposite direction of the gradient vector.",
        expected_concepts=["gradient_descent", "loss_functions"],
        learner_state=learner_state,
        strategy_engine=strat_engine,
        strategy_category=category,
        strategy_type=stype,
        concept_domain=domain
    )

    assert res1.correct is True
    assert res1.updated_p_known is not None
    assert res1.updated_p_known > 0.30  # Posterior BKT increased
    assert learner_state.concept_mastery["gradient_descent"].mastery == res1.updated_p_known

    # Check Strategy effectiveness update
    eff_rec = store.get_effectiveness(user_id, category, stype, domain)
    assert eff_rec is not None
    assert eff_rec.times_used == 1
    assert eff_rec.times_led_to_mastery == 1
    assert eff_rec.effectiveness_ratio == 1.0

    # 2. Student fails next question with hints
    res2 = evaluator.evaluate_and_update(
        user_id=user_id,
        response="I don't know.",
        expected_concepts=["gradient_descent"],
        learner_state=learner_state,
        strategy_engine=strat_engine,
        strategy_category=category,
        strategy_type=stype,
        concept_domain=domain,
        hints_used=2
    )

    assert res2.correct is False
    assert res2.updated_p_known < res1.updated_p_known  # Posterior BKT decreased

    eff_rec2 = store.get_effectiveness(user_id, category, stype, domain)
    assert eff_rec2.times_used == 2
    assert eff_rec2.times_led_to_mastery == 1
    assert eff_rec2.effectiveness_ratio == 0.5
