"""
tests/test_quiz_and_assessment.py
---------------------------------
Unit tests for QuizAgent (difficulty adjustment + 500-1200 token budget)
and AssessmentAgent (grading + LearningEvent emission feeding Step 2 LearnerModelEngine).
"""

import pytest
from ai_tutor.models import (
    AssessmentGrade,
    Chunk,
    ConceptMastery,
    KnowledgeContext,
    LearnerState,
    LearningContext,
    LearningEvent,
    LearningEventType,
    Misconception,
    OrchestratedContext,
    PedagogyState,
    QuizQuestion,
    SessionContext,
    StrategyAction,
    TeachingStrategy,
)
from ai_tutor.event_bus import InMemoryEventBus
from ai_tutor.learner_store import InMemoryLearnerStateStore
from ai_tutor.learner_model import MisconceptionEngine, LearnerModelEngine
from ai_tutor.quiz_agent import QuizAgent
from ai_tutor.assessment_agent import AssessmentAgent


# ---------------------------------------------------------------------------
# 1. QuizAgent Tests
# ---------------------------------------------------------------------------

class TestQuizAgent:
    def setup_method(self):
        self.quiz_agent = QuizAgent()

    def test_generates_within_500_to_1200_token_budget(self):
        strategy = TeachingStrategy(
            recommendation=StrategyAction.QUIZ,
            target_concept="Gradient Descent",
            target_mastery=0.85,
            rationale="Formative assessment check"
        )
        quiz: QuizQuestion = self.quiz_agent.generate(
            target_concept="Gradient Descent",
            strategy=strategy
        )

        assert isinstance(quiz, QuizQuestion)
        assert quiz.concept == "Gradient Descent"
        # Verify 500 - 1200 token budget envelope
        assert 500 <= quiz.token_count <= 1200, f"Token count {quiz.token_count} outside [500, 1200] range"
        assert len(quiz.question) > 0

    def test_difficulty_adjustment_easy(self):
        strategy = TeachingStrategy(
            recommendation=StrategyAction.QUIZ,
            target_concept="Linear Regression",
            difficulty_adjustment="easy",
            rationale="Introductory check"
        )
        quiz = self.quiz_agent.generate(strategy=strategy)
        assert quiz.difficulty == "easy"
        assert "Foundational" in quiz.question

    def test_difficulty_adjustment_hard(self):
        strategy = TeachingStrategy(
            recommendation=StrategyAction.CHALLENGE,
            target_concept="Backpropagation",
            difficulty_adjustment="hard",
            target_mastery=0.95,
            rationale="Advanced optimization challenge"
        )
        quiz = self.quiz_agent.generate(strategy=strategy)
        assert quiz.difficulty == "hard"
        assert "Advanced" in quiz.question or "Optimization" in quiz.question

    def test_targets_active_misconception_specifically(self):
        misc = Misconception(
            key="gd_local_minimum_paralysis",
            description="Believes gradient descent gets permanently stuck on local minima in convex problems.",
            concept="Gradient Descent",
            confidence=0.8
        )
        strategy = TeachingStrategy(
            recommendation=StrategyAction.QUIZ,
            target_concept="Gradient Descent",
            misconception_to_address=misc,
            rationale="Expose misconception"
        )
        quiz = self.quiz_agent.generate(strategy=strategy)
        assert quiz.misconception_targeted == "gd_local_minimum_paralysis"
        assert "Believes gradient descent gets permanently stuck" in quiz.question

    def test_incorporates_grounded_rag_context(self):
        chunks = [
            Chunk(content="Momentum replaces standard SGD updates by maintaining an exponential velocity vector.", source_title="Optimization Lecture", source_id=1)
        ]
        context = OrchestratedContext(
            student_message="Quiz me",
            session_context=SessionContext(session_id="s1"),
            learning_context=LearningContext(),
            knowledge_context=KnowledgeContext(chunks=chunks)
        )
        strategy = TeachingStrategy(
            recommendation=StrategyAction.QUIZ,
            target_concept="SGD with Momentum",
            rationale="Test momentum"
        )
        quiz = self.quiz_agent.generate(strategy=strategy, context=context)
        assert "Momentum replaces standard SGD" in quiz.question


# ---------------------------------------------------------------------------
# 2. AssessmentAgent Tests & Event Emission
# ---------------------------------------------------------------------------

class TestAssessmentAgent:
    def setup_method(self):
        self.bus = InMemoryEventBus()
        self.assessment_agent = AssessmentAgent(event_bus=self.bus)

    def test_grades_correct_response_and_emits_event(self):
        grade: AssessmentGrade = self.assessment_agent.grade(
            student_answer="The learning rate determines the step size taken along the negative gradient direction to reach the minimum.",
            expected_concept="Gradient Descent",
            hints_used=1,
            student_id="student_101",
            session_id="sess_abc"
        )

        assert grade.correct is True
        assert grade.score == 1.0
        assert grade.hints_used == 1
        assert "correct" in grade.feedback.lower()

        # Verify emitted LearningEvent
        assert len(self.bus.log) == 1
        emitted: LearningEvent = self.bus.log[0]
        assert emitted.event_type == LearningEventType.ANSWER_SUBMITTED.value
        assert emitted.student_id == "student_101"
        assert emitted.session_id == "sess_abc"
        assert emitted.concept == "Gradient Descent"
        assert emitted.hint_level == 1
        assert emitted.payload["correct"] is True
        assert emitted.payload["hints_used"] == 1
        assert emitted.payload["concept"] == "Gradient Descent"

    def test_grades_misconception_response_and_emits_event(self):
        grade: AssessmentGrade = self.assessment_agent.grade(
            student_answer="Gradient descent is always stuck in local minima and can never escape.",
            expected_concept="Gradient Descent",
            hints_used=0,
            student_id="student_102",
            session_id="sess_xyz"
        )

        assert grade.correct is False
        assert grade.score == 0.0
        assert grade.misconception_detected == "gd_local_minimum_paralysis"
        assert "misconception" in grade.feedback.lower()

        # Verify emitted event captures misconception
        assert len(self.bus.log) == 1
        emitted: LearningEvent = self.bus.log[0]
        assert emitted.payload["correct"] is False
        assert emitted.payload["misconception_detected"] == "gd_local_minimum_paralysis"

    def test_grades_give_up_response(self):
        grade = self.assessment_agent.grade(
            student_answer="I don't know, tell me the answer please",
            expected_concept="Backpropagation",
            hints_used=2,
            student_id="student_103"
        )
        assert grade.correct is False
        assert grade.score == 0.0
        assert "together" in grade.feedback.lower() or "work through" in grade.feedback.lower()


# ---------------------------------------------------------------------------
# 3. Closed-Loop Integration: AssessmentAgent -> EventBus -> LearnerModelEngine (Step 2)
# ---------------------------------------------------------------------------

class TestClosedLoopLearnerModelIntegration:
    def test_assessment_event_updates_learner_model_state(self):
        """
        Closed-loop verification:
        AssessmentAgent grades -> emits ANSWER_SUBMITTED event ->
        LearnerModelEngine (Step 2) receives event via EventBus ->
        KnowledgeTracer (BKT) + MisconceptionEngine update LearnerState in store!
        """
        bus = InMemoryEventBus()
        store = InMemoryLearnerStateStore()

        # Initialize Step 2 LearnerModelEngine subscribed to the event bus
        engine = LearnerModelEngine(store=store, bus=bus)

        # Initialize AssessmentAgent connected to the same event bus
        assessor = AssessmentAgent(event_bus=bus)

        # 1. Student answers correctly on 'Gradient Descent' with 1 hint used
        grade1 = assessor.grade(
            student_answer="The learning rate scales the magnitude of the negative gradient step.",
            expected_concept="Gradient Descent",
            hints_used=1,
            student_id="student_loop_test",
            session_id="sess_loop_1"
        )
        assert grade1.correct is True

        # Check that LearnerState was automatically updated in the store
        state1 = store.load("student_loop_test")
        assert state1 is not None
        assert "Gradient Descent" in state1.concept_mastery
        assert state1.concept_mastery["Gradient Descent"].attempts == 1
        assert state1.concept_mastery["Gradient Descent"].correct == 1
        assert state1.concept_mastery["Gradient Descent"].mastery > 0.10

        # 2. Student submits an answer with a misconception
        grade2 = assessor.grade(
            student_answer="The chain rule gradients flow forward from input to output layer.",
            expected_concept="Backpropagation",
            hints_used=0,
            student_id="student_loop_test",
            session_id="sess_loop_1"
        )
        assert grade2.correct is False
        assert grade2.misconception_detected == "chain_rule_layer_order"

        # Check that LearnerState now reflects the new concept and detected misconception
        state2 = store.load("student_loop_test")
        assert "Backpropagation" in state2.concept_mastery
        assert state2.concept_mastery["Backpropagation"].attempts == 1
        assert state2.concept_mastery["Backpropagation"].correct == 0

        # Verify misconception tracked in LearnerState
        assert len(state2.misconceptions) == 1
        assert state2.misconceptions[0].key == "chain_rule_layer_order"
        assert state2.misconceptions[0].concept == "Backpropagation"
        assert state2.misconceptions[0].confidence >= 0.5
