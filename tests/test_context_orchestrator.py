"""
tests/test_context_orchestrator.py
----------------------------------
Unit tests for ContextOrchestrator, ContextAgent, LearningAgent,
KnowledgeAgent, and OrchestratedContext.
"""

import pytest
from unittest.mock import MagicMock
from ai_tutor.models import (
    AIChatRequest,
    ChatMessage,
    Chunk,
    ConceptMastery,
    KnowledgeContext,
    LearnerState,
    LearningContext,
    Misconception,
    OrchestratedContext,
    PedagogyState,
    Role,
    SessionContext,
    SourceCitation,
    StrategyAction,
    TeachingStrategy,
)
from ai_tutor.session_store import InMemorySessionStore
from ai_tutor.learner_store import InMemoryLearnerStateStore
from ai_tutor.knowledge_source import MockKnowledgeSource
from ai_tutor.context_orchestrator import (
    BaseContextAgent,
    BaseLearningAgent,
    BaseKnowledgeAgent,
    ContextAgent,
    LearningAgent,
    KnowledgeAgent,
    ContextOrchestrator,
)


# ---------------------------------------------------------------------------
# 1. ContextAgent Tests (Isolated)
# ---------------------------------------------------------------------------

class TestContextAgent:
    def test_reads_session_history_and_turn_count(self):
        store = InMemorySessionStore()
        store.append_message("sess_1", Role.USER, "Hello!")
        store.append_message("sess_1", Role.ASSISTANT, "Hi there!")

        agent = ContextAgent(session_store=store)
        ctx = agent.run(session_id="sess_1", student_message="I have a question.")

        assert ctx.session_id == "sess_1"
        assert ctx.turn_count == 2
        assert len(ctx.recent_messages) == 2
        assert ctx.detected_mood == "neutral"

    def test_detects_mood_signals(self):
        agent = ContextAgent()

        # Confused
        ctx_confused = agent.run(session_id="s1", student_message="I am totally confused and stuck")
        assert ctx_confused.detected_mood == "confused"

        # Frustrated
        ctx_frustrated = agent.run(session_id="s1", student_message="This is stupid, I hate this")
        assert ctx_frustrated.detected_mood == "frustrated"

        # Confident
        ctx_confident = agent.run(session_id="s1", student_message="That makes sense, I got it!")
        assert ctx_confident.detected_mood == "confident"

        # Curious
        ctx_curious = agent.run(session_id="s1", student_message="Why does the gradient point uphill?")
        assert ctx_curious.detected_mood == "curious"

    def test_respects_history_override(self):
        agent = ContextAgent()
        history = [ChatMessage(role=Role.USER, content="Prior question")]
        ctx = agent.run(
            session_id="sess_custom",
            student_message="Follow up",
            history_override=history
        )
        assert len(ctx.recent_messages) == 1
        assert ctx.recent_messages[0].content == "Prior question"


# ---------------------------------------------------------------------------
# 2. LearningAgent Tests (Isolated)
# ---------------------------------------------------------------------------

class TestLearningAgent:
    def test_reads_learner_state_and_resolves_strategy(self):
        store = InMemoryLearnerStateStore()
        state = LearnerState(student_id="user_123")
        state.concept_mastery["Gradient Descent"] = ConceptMastery(
            concept="Gradient Descent",
            mastery=0.85
        )
        state.misconceptions.append(
            Misconception(
                key="gd_local_min",
                description="Local minimum trap",
                concept="Gradient Descent",
                confidence=0.8
            )
        )
        store.save(state)

        agent = LearningAgent(learner_store=store)
        ctx = agent.run(
            student_id="user_123",
            target_concept="Gradient Descent"
        )

        assert ctx.student_id == "user_123"
        assert ctx.target_concept == "Gradient Descent"
        assert ctx.target_mastery == 0.85
        assert len(ctx.active_misconceptions) == 1
        assert ctx.teaching_strategy is not None
        assert ctx.teaching_strategy.recommendation == StrategyAction.QUIZ

    def test_handles_nonexistent_student_gracefully(self):
        store = InMemoryLearnerStateStore()
        agent = LearningAgent(learner_store=store)
        ctx = agent.run(student_id="unknown_student", target_concept="Overfitting")

        assert ctx.learner_state is None
        assert ctx.teaching_strategy.recommendation == StrategyAction.GUIDE


# ---------------------------------------------------------------------------
# 3. KnowledgeAgent Tests (Isolated)
# ---------------------------------------------------------------------------

class TestKnowledgeAgent:
    def test_retrieves_chunks_and_builds_citations(self):
        ks = MockKnowledgeSource()
        agent = KnowledgeAgent(knowledge_source=ks)

        ctx = agent.run(query="gradient descent", course_id=101, lecture_id=60)
        assert len(ctx.chunks) > 0
        assert len(ctx.citations) > 0
        assert ctx.knowledge_source_used == "MockKnowledgeSource"
        assert ctx.query == "gradient descent"

    def test_handles_empty_or_none_knowledge_source(self):
        agent = KnowledgeAgent(knowledge_source=None)
        ctx = agent.run(query="anything")
        assert len(ctx.chunks) == 0
        assert len(ctx.citations) == 0
        assert ctx.knowledge_source_used is None


# ---------------------------------------------------------------------------
# 4. ContextOrchestrator Parallel Fan-Out & Merge Tests
# ---------------------------------------------------------------------------

class TestContextOrchestrator:
    def test_orchestrates_all_three_agents_in_parallel(self):
        session_store = InMemorySessionStore()
        session_store.append_message("sess_abc", Role.USER, "What is learning rate?")

        learner_store = InMemoryLearnerStateStore()
        state = LearnerState(student_id="student_xyz")
        state.concept_mastery["Gradient Descent"] = ConceptMastery(concept="Gradient Descent", mastery=0.95)
        learner_store.save(state)

        knowledge_source = MockKnowledgeSource()

        orchestrator = ContextOrchestrator(
            context_agent=ContextAgent(session_store=session_store),
            learning_agent=LearningAgent(learner_store=learner_store),
            knowledge_agent=KnowledgeAgent(knowledge_source=knowledge_source)
        )

        orch_ctx: OrchestratedContext = orchestrator.orchestrate(
            student_message="Why does learning rate matter in gradient descent?",
            session_id="sess_abc",
            student_id="student_xyz",
            course_id=101,
            lecture_id=60,
            target_concept="Gradient Descent"
        )

        # Verify Session Context merged
        assert orch_ctx.session_context.session_id == "sess_abc"
        assert orch_ctx.session_context.turn_count == 1
        assert orch_ctx.session_context.detected_mood == "curious"

        # Verify Learning Context merged
        assert orch_ctx.learning_context.student_id == "student_xyz"
        assert orch_ctx.learning_context.target_mastery == 0.95
        assert orch_ctx.learning_context.teaching_strategy.recommendation == StrategyAction.CHALLENGE

        # Verify Knowledge Context merged
        assert len(orch_ctx.knowledge_context.chunks) > 0
        assert orch_ctx.knowledge_context.knowledge_source_used == "MockKnowledgeSource"

        # Verify Orchestrated Prompt Sections Generation
        sections = orch_ctx.to_prompt_sections()
        assert "learner_state" in sections
        assert "rag_knowledge" in sections
        assert "teaching_strategy" in sections
        assert "conversation_history" in sections
        assert "conversation_summary" in sections

    def test_orchestrate_with_mocked_agents(self):
        """Verify each agent is independently mockable."""
        mock_context_agent = MagicMock(spec=BaseContextAgent)
        mock_context_agent.run.return_value = SessionContext(
            session_id="mock_sess",
            detected_mood="confident"
        )

        mock_learning_agent = MagicMock(spec=BaseLearningAgent)
        mock_learning_agent.run.return_value = LearningContext(
            student_id="mock_student",
            target_concept="MockConcept",
            teaching_strategy=TeachingStrategy(
                recommendation=StrategyAction.HINT,
                rationale="Mock rationale"
            )
        )

        mock_knowledge_agent = MagicMock(spec=BaseKnowledgeAgent)
        mock_knowledge_agent.run.return_value = KnowledgeContext(
            chunks=[Chunk(content="Mock chunk", source_title="Mock Source", source_id=1)],
            knowledge_source_used="MockKS"
        )

        orchestrator = ContextOrchestrator(
            context_agent=mock_context_agent,
            learning_agent=mock_learning_agent,
            knowledge_agent=mock_knowledge_agent
        )

        orch_ctx = orchestrator.orchestrate(
            student_message="Test prompt",
            session_id="mock_sess",
            student_id="mock_student"
        )

        assert orch_ctx.session_context.detected_mood == "confident"
        assert orch_ctx.learning_context.target_concept == "MockConcept"
        assert orch_ctx.knowledge_context.chunks[0].content == "Mock chunk"

        mock_context_agent.run.assert_called_once()
        mock_learning_agent.run.assert_called_once()
        mock_knowledge_agent.run.assert_called_once()

    def test_resilience_to_agent_failure(self):
        """If one agent throws an unhandled exception, the orchestrator should not crash."""
        failing_knowledge_agent = MagicMock(spec=BaseKnowledgeAgent)
        failing_knowledge_agent.run.side_effect = RuntimeError("Knowledge DB disconnected")

        orchestrator = ContextOrchestrator(
            context_agent=ContextAgent(),
            learning_agent=LearningAgent(),
            knowledge_agent=failing_knowledge_agent
        )

        orch_ctx = orchestrator.orchestrate(
            student_message="Hello",
            session_id="sess_1"
        )

        assert orch_ctx is not None
        assert orch_ctx.knowledge_context.chunks == []
        assert orch_ctx.session_context.session_id == "sess_1"

    def test_orchestrate_request_helper(self):
        orchestrator = ContextOrchestrator()
        req = AIChatRequest(
            message="Explain SGD",
            session_id="req_sess",
            student_id="req_stu",
            course_id=10
        )
        orch_ctx = orchestrator.orchestrate_request(req)
        assert orch_ctx.student_message == "Explain SGD"
        assert orch_ctx.session_context.session_id == "req_sess"
        assert orch_ctx.learning_context.student_id == "req_stu"
