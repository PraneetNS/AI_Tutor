"""
tests/test_tutor_reasoner.py
----------------------------
Unit tests for TutorReasoner ('decide -> teach -> assess -> adapt' loop)
and the three sub-agents (TutorAgent, QuizAgent, AssessmentAgent).
"""

import pytest
from unittest.mock import MagicMock
from ai_tutor.models import (
    ChatMessage,
    Chunk,
    ConceptMastery,
    KnowledgeContext,
    LearnerState,
    LearningContext,
    Misconception,
    OrchestratedContext,
    PedagogyMode,
    PedagogyState,
    ReasonerResult,
    Role,
    SessionContext,
    SourceCitation,
    StrategyAction,
    TeachingStrategy,
)
from ai_tutor.quiz_agent import QuizAgent
from ai_tutor.assessment_agent import AssessmentAgent
from ai_tutor.tutor_reasoner import (
    BaseTutorSubAgent,
    TutorAgent,
    TutorReasoner,
)


def _build_mock_context(
    student_message: str = "Explain gradient descent",
    recent_messages: list = None,
    target_concept: str = "Gradient Descent",
    pedagogy_state: PedagogyState = None,
    misconceptions: list = None
) -> OrchestratedContext:
    session_ctx = SessionContext(
        session_id="sess_test",
        recent_messages=recent_messages or [],
        pedagogy_state=pedagogy_state or PedagogyState(),
        detected_mood="neutral"
    )
    learning_ctx = LearningContext(
        student_id="student_1",
        target_concept=target_concept,
        active_misconceptions=misconceptions or []
    )
    knowledge_ctx = KnowledgeContext(
        chunks=[Chunk(content="Gradient descent is an iterative algorithm.", source_title="ML Lecture", source_id=1)],
        citations=[SourceCitation(lecture_id=1, title="ML Lecture", snippet="Gradient descent...")]
    )
    return OrchestratedContext(
        student_message=student_message,
        session_context=session_ctx,
        learning_context=learning_ctx,
        knowledge_context=knowledge_ctx
    )


# ---------------------------------------------------------------------------
# 1. Sub-Agent Unit Tests
# ---------------------------------------------------------------------------

class TestTutorAgent:
    def test_tutor_agent_generates_response(self):
        agent = TutorAgent()
        context = _build_mock_context()
        strategy = TeachingStrategy(
            recommendation=StrategyAction.GUIDE,
            target_concept="Gradient Descent",
            rationale="Guide Socratic dialogue"
        )
        answer = agent.generate(context, strategy)
        assert isinstance(answer, str)
        assert len(answer) > 0


class TestQuizAgent:
    def test_quiz_agent_targets_concept_and_misconception(self):
        agent = QuizAgent()
        misc = Misconception(
            key="gd_local_min",
            description="Believes GD is always stuck in local minima",
            concept="Gradient Descent"
        )
        context = _build_mock_context(misconceptions=[misc])
        strategy = TeachingStrategy(
            recommendation=StrategyAction.QUIZ,
            target_concept="Gradient Descent",
            misconception_to_address=misc,
            rationale="Formative check on GD"
        )
        quiz_text = agent.generate_quiz(context, strategy)
        assert "Gradient Descent" in quiz_text
        assert "Believes GD is always stuck" in quiz_text


class TestAssessmentAgent:
    def test_evaluates_correct_response(self):
        agent = AssessmentAgent()
        context = _build_mock_context(student_message="The learning rate scales the step size taken in the negative gradient direction.")
        strategy = TeachingStrategy(
            recommendation=StrategyAction.GUIDE,
            target_concept="Gradient Descent",
            consecutive_failures=0,
            rationale="Evaluate answer"
        )
        feedback, meta = agent.evaluate(context, strategy)
        assert meta["correct"] is True
        assert "accurate" in feedback.lower() or "intuition" in feedback.lower()

    def test_evaluates_incorrect_response(self):
        agent = AssessmentAgent()
        context = _build_mock_context(student_message="It is always stuck in local minima forever.")
        strategy = TeachingStrategy(
            recommendation=StrategyAction.GUIDE,
            target_concept="Gradient Descent",
            consecutive_failures=0,
            rationale="Evaluate answer"
        )
        feedback, meta = agent.evaluate(context, strategy)
        assert meta["correct"] is False
        assert "misconception" in feedback.lower() or "not quite" in feedback.lower()


# ---------------------------------------------------------------------------
# 2. TutorReasoner DECIDE Stage Tests
# ---------------------------------------------------------------------------

class TestTutorReasonerDecide:
    def setup_method(self):
        self.reasoner = TutorReasoner()

    def test_decide_picks_quiz_agent_when_strategy_is_quiz(self):
        context = _build_mock_context()
        strategy = TeachingStrategy(
            recommendation=StrategyAction.QUIZ,
            target_concept="Loss Functions",
            rationale="Mastery > 0.8"
        )
        agent_name = self.reasoner.decide(context, strategy)
        assert agent_name == "QuizAgent"

    def test_decide_picks_quiz_agent_when_student_asks_for_quiz(self):
        context = _build_mock_context(student_message="Can you quiz me on backpropagation?")
        strategy = TeachingStrategy(
            recommendation=StrategyAction.GUIDE,
            target_concept="Backpropagation",
            rationale="General guide"
        )
        agent_name = self.reasoner.decide(context, strategy)
        assert agent_name == "QuizAgent"

    def test_decide_picks_assessment_agent_when_student_answers_prior_question(self):
        recent = [
            ChatMessage(role=Role.ASSISTANT, content="What happens when the learning rate is too large?")
        ]
        context = _build_mock_context(
            student_message="The model will overshoot the minimum and oscillate.",
            recent_messages=recent
        )
        strategy = TeachingStrategy(
            recommendation=StrategyAction.GUIDE,
            target_concept="Gradient Descent",
            rationale="Awaiting response"
        )
        agent_name = self.reasoner.decide(context, strategy)
        assert agent_name == "AssessmentAgent"

    def test_decide_picks_tutor_agent_for_hinting_and_explaining(self):
        context = _build_mock_context(student_message="I don't understand loss functions.")
        strategy = TeachingStrategy(
            recommendation=StrategyAction.HINT,
            target_concept="Loss Functions",
            rationale="Student needs scaffolding"
        )
        agent_name = self.reasoner.decide(context, strategy)
        assert agent_name == "TutorAgent"


# ---------------------------------------------------------------------------
# 3. TutorReasoner ADAPT Stage Tests
# ---------------------------------------------------------------------------

class TestTutorReasonerAdapt:
    def setup_method(self):
        self.reasoner = TutorReasoner()

    def test_adapt_increments_hint_level_on_hint_strategy(self):
        current_state = PedagogyState(hint_level=1, stuck=False)
        strategy = TeachingStrategy(
            recommendation=StrategyAction.HINT,
            target_concept="Overfitting",
            rationale="Provide hint"
        )
        updated = self.reasoner.adapt(
            selected_agent="TutorAgent",
            current_state=current_state,
            strategy=strategy,
            assessment_result=None,
            student_message="Give me a hint please",
            detected_mood="neutral"
        )
        assert updated.hint_level == 2
        assert updated.stuck is True
        assert updated.topic == "Overfitting"

    def test_adapt_switches_to_direct_mode_on_explain(self):
        current_state = PedagogyState(hint_level=3, stuck=True)
        strategy = TeachingStrategy(
            recommendation=StrategyAction.EXPLAIN,
            target_concept="Backpropagation",
            consecutive_failures=2,
            rationale="Rule 5: direct explanation"
        )
        updated = self.reasoner.adapt(
            selected_agent="TutorAgent",
            current_state=current_state,
            strategy=strategy,
            assessment_result=None,
            student_message="I still don't get it",
            detected_mood="frustrated"
        )
        assert updated.pedagogy_mode == PedagogyMode.DIRECT
        assert updated.stuck is True

    def test_adapt_resets_hint_level_on_correct_assessment(self):
        current_state = PedagogyState(hint_level=2, stuck=True)
        strategy = TeachingStrategy(
            recommendation=StrategyAction.GUIDE,
            target_concept="Gradient Descent",
            rationale="Check understanding"
        )
        updated = self.reasoner.adapt(
            selected_agent="AssessmentAgent",
            current_state=current_state,
            strategy=strategy,
            assessment_result={"correct": True, "score": 1.0},
            student_message="It moves opposite the gradient.",
            detected_mood="confident"
        )
        assert updated.hint_level == 0
        assert updated.stuck is False


# ---------------------------------------------------------------------------
# 4. End-to-End reason_turn Integration Tests
# ---------------------------------------------------------------------------

class TestTutorReasonerFullTurn:
    def test_full_turn_loop_execution(self):
        reasoner = TutorReasoner()
        context = _build_mock_context(student_message="What is the role of the loss function?")
        strategy = TeachingStrategy(
            recommendation=StrategyAction.GUIDE,
            target_concept="Loss Functions",
            rationale="Socratic diagnostic"
        )

        result: ReasonerResult = reasoner.reason_turn(context=context, strategy_override=strategy)

        assert isinstance(result, ReasonerResult)
        assert result.selected_agent == "TutorAgent"
        assert len(result.answer) > 0
        assert result.pedagogy_state.topic == "Loss Functions"
        assert len(result.sources) == 1
        assert result.strategy_applied.target_concept == "Loss Functions"

    def test_full_turn_with_quiz_agent(self):
        reasoner = TutorReasoner()
        context = _build_mock_context(student_message="Quiz me on Overfitting")
        strategy = TeachingStrategy(
            recommendation=StrategyAction.QUIZ,
            target_concept="Overfitting",
            rationale="Formative quiz check"
        )

        result: ReasonerResult = reasoner.reason_turn(context=context, strategy_override=strategy)

        assert result.selected_agent == "QuizAgent"
        assert "Overfitting" in result.answer
        assert result.pedagogy_state.topic == "Overfitting"

    def test_mockable_sub_agents_in_reasoner(self):
        mock_tutor = MagicMock(spec=BaseTutorSubAgent)
        mock_tutor.generate.return_value = "Mocked tutor answer"

        reasoner = TutorReasoner(tutor_agent=mock_tutor)
        context = _build_mock_context(student_message="Teach me")
        strategy = TeachingStrategy(
            recommendation=StrategyAction.GUIDE,
            target_concept="Topic A",
            rationale="Test"
        )

        result = reasoner.reason_turn(context=context, strategy_override=strategy)
        assert result.answer == "Mocked tutor answer"
        mock_tutor.generate.assert_called_once()
