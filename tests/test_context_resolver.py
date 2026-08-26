"""
tests/test_context_resolver.py
------------------------------
Unit tests for ContextResolver and TeachingStrategy recommendations.
"""

import pytest
from ai_tutor.models import (
    ConceptMastery,
    LearnerState,
    Misconception,
    StrategyAction,
)
from ai_tutor.context_resolver import ContextResolver


class TestContextResolver:

    def setup_method(self):
        self.resolver = ContextResolver()

    def test_explain_recommended_when_2_plus_wrong_attempts(self):
        """Rule: recommend 'explain' if 2+ wrong attempts on target_concept."""
        state = LearnerState(student_id="student_1")
        strategy = self.resolver.resolve(
            learner_state=state,
            target_concept="Backpropagation",
            last_answer_correct=False,
            consecutive_failures=2,
            hint_budget_remaining=3
        )
        assert strategy.recommendation == StrategyAction.EXPLAIN
        assert strategy.recommendation == "explain"
        assert strategy.consecutive_failures == 2
        assert "direct explanation" in strategy.rationale.lower()

    def test_explain_recommended_on_3_or_more_failures(self):
        state = LearnerState(student_id="student_1")
        strategy = self.resolver.resolve(
            learner_state=state,
            target_concept="Backpropagation",
            last_answer_correct=False,
            consecutive_failures=3,
            hint_budget_remaining=1
        )
        assert strategy.recommendation == StrategyAction.EXPLAIN

    def test_hint_recommended_when_wrong_and_budget_remains(self):
        """Rule: recommend 'hint' if hint_budget_remaining > 0 and last answer was wrong."""
        state = LearnerState(student_id="student_1")
        strategy = self.resolver.resolve(
            learner_state=state,
            target_concept="Gradient Descent",
            last_answer_correct=False,
            consecutive_failures=1,
            hint_budget_remaining=2
        )
        assert strategy.recommendation == StrategyAction.HINT
        assert strategy.recommendation == "hint"
        assert strategy.hint_budget_remaining == 2
        assert "scaffolding hint" in strategy.rationale.lower()

    def test_quiz_recommended_when_mastery_above_point_8(self):
        """Rule: recommend 'quiz' if mastery on target_concept > 0.8 (check for real understanding)."""
        state = LearnerState(student_id="student_1")
        state.concept_mastery["Loss Functions"] = ConceptMastery(
            concept="Loss Functions",
            mastery=0.85
        )
        strategy = self.resolver.resolve(
            learner_state=state,
            target_concept="Loss Functions",
            last_answer_correct=True,
            consecutive_failures=0
        )
        assert strategy.recommendation == StrategyAction.QUIZ
        assert strategy.recommendation == "quiz"
        assert strategy.target_mastery == 0.85
        assert "formative quiz" in strategy.rationale.lower()

    def test_challenge_recommended_when_mastery_above_point_9(self):
        """Rule: recommend 'challenge' if mastery > 0.9 (push to next concept)."""
        state = LearnerState(student_id="student_1")
        state.concept_mastery["Neural Networks"] = ConceptMastery(
            concept="Neural Networks",
            mastery=0.95
        )
        strategy = self.resolver.resolve(
            learner_state=state,
            target_concept="Neural Networks",
            last_answer_correct=True,
            consecutive_failures=0
        )
        assert strategy.recommendation == StrategyAction.CHALLENGE
        assert strategy.recommendation == "challenge"
        assert strategy.target_mastery == 0.95
        assert "pushing learner" in strategy.rationale.lower()

    def test_default_guide_recommendation(self):
        """Default: Socratic diagnostic guide when no trigger applies."""
        state = LearnerState(student_id="student_1")
        state.concept_mastery["Overfitting"] = ConceptMastery(
            concept="Overfitting",
            mastery=0.4
        )
        strategy = self.resolver.resolve(
            learner_state=state,
            target_concept="Overfitting",
            last_answer_correct=None,
            consecutive_failures=0
        )
        assert strategy.recommendation == StrategyAction.GUIDE
        assert strategy.recommendation == "guide"

    def test_surfaces_highest_confidence_unresolved_misconception(self):
        """Rule: Always surface the highest-confidence unresolved misconception as misconception_to_address."""
        state = LearnerState(student_id="student_1")
        m1 = Misconception(
            key="misc_low",
            description="Low confidence issue",
            concept="Gradient Descent",
            confidence=0.4,
            hit_count=1
        )
        m2 = Misconception(
            key="misc_high",
            description="High confidence misconception",
            concept="Gradient Descent",
            confidence=0.85,
            hit_count=3
        )
        state.misconceptions.extend([m1, m2])

        strategy = self.resolver.resolve(
            learner_state=state,
            target_concept="Gradient Descent"
        )
        assert strategy.misconception_to_address is not None
        assert strategy.misconception_to_address.key == "misc_high"
        assert strategy.misconception_to_address.confidence == 0.85

    def test_surfaces_concept_specific_misconception_if_available(self):
        state = LearnerState(student_id="student_1")
        m_other = Misconception(
            key="misc_other",
            description="Other concept issue",
            concept="Overfitting",
            confidence=0.9,
            hit_count=2
        )
        m_target = Misconception(
            key="misc_target",
            description="Target concept issue",
            concept="Gradient Descent",
            confidence=0.75,
            hit_count=1
        )
        state.misconceptions.extend([m_other, m_target])

        strategy = self.resolver.resolve(
            learner_state=state,
            target_concept="Gradient Descent"
        )
        assert strategy.misconception_to_address is not None
        assert strategy.misconception_to_address.key == "misc_target"

    def test_handles_no_learner_state_gracefully(self):
        strategy = self.resolver.resolve(
            learner_state=None,
            target_concept="Supervised Learning"
        )
        assert strategy.recommendation == StrategyAction.GUIDE
        assert strategy.misconception_to_address is None
        assert strategy.target_mastery is None
        assert strategy.hint_budget_remaining == 3
