"""
context_orchestrator.py
-----------------------
Parallel context orchestration engine.

Fans out to three specialized sub-agents concurrently:
1. ContextAgent: Reads short-term session memory, recent turns, and detects student mood.
2. LearningAgent: Reads long-term LearnerState, concept masteries, active misconceptions, and resolves TeachingStrategy.
3. KnowledgeAgent: Retrieves relevant curriculum grounding chunks via KnowledgeSource RAG interface.

Merges their outputs into a single unified `OrchestratedContext` object ready
for consumption by the Tutor Reasoner and BudgetManager.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

from .context_resolver import ContextResolver
from .knowledge_source import KnowledgeSource
from .learner_store import BaseLearnerStateStore, InMemoryLearnerStateStore
from .models import (
    AIChatRequest,
    ChatMessage,
    Chunk,
    KnowledgeContext,
    LearnerState,
    LearningContext,
    OrchestratedContext,
    PedagogyState,
    SessionContext,
    SourceCitation,
    TeachingStrategy,
)
from .session_store import BaseSessionStore, InMemorySessionStore

logger = logging.getLogger("ai_tutor.context_orchestrator")


# ---------------------------------------------------------------------------
# 1. Base and Concrete ContextAgent
# ---------------------------------------------------------------------------

class BaseContextAgent(ABC):
    """Abstract interface for short-term session context retrieval and analysis."""

    @abstractmethod
    def run(
        self,
        session_id: str,
        student_message: str,
        history_override: Optional[List[ChatMessage]] = None,
        pedagogy_state_override: Optional[PedagogyState] = None
    ) -> SessionContext:
        pass


class ContextAgent(BaseContextAgent):
    """
    Retrieves recent session history, tracks turn counts, and computes
    heuristic tone/mood detection from incoming signals.
    """

    def __init__(
        self,
        session_store: Optional[BaseSessionStore] = None,
        max_recent_turns: int = 10
    ) -> None:
        self.session_store = session_store or InMemorySessionStore()
        self.max_recent_turns = max_recent_turns

    def run(
        self,
        session_id: str,
        student_message: str,
        history_override: Optional[List[ChatMessage]] = None,
        pedagogy_state_override: Optional[PedagogyState] = None
    ) -> SessionContext:
        # 1. Load session data
        if history_override is not None:
            messages = list(history_override)
            state = pedagogy_state_override or PedagogyState()
        else:
            session_data = self.session_store.get_session(session_id)
            messages = session_data.messages
            state = pedagogy_state_override or session_data.pedagogy_state

        recent = messages[-self.max_recent_turns:] if len(messages) > self.max_recent_turns else messages

        # 2. Detect student mood / sentiment heuristic
        mood = self._detect_mood(student_message, recent)

        return SessionContext(
            session_id=session_id,
            recent_messages=recent,
            pedagogy_state=state,
            detected_mood=mood,
            turn_count=len(messages)
        )

    def _detect_mood(self, current_msg: str, recent_messages: List[ChatMessage]) -> str:
        msg_lower = (current_msg or "").lower()

        # Frustration signals
        frustrated_keywords = ["hate", "stupid", "impossible", "give up", "makes no sense", "annoying", "ugh"]
        if any(w in msg_lower for w in frustrated_keywords):
            return "frustrated"

        # Confusion / stuck signals
        confused_keywords = ["confused", "stuck", "don't understand", "dont get it", "lost", "what do you mean"]
        if any(w in msg_lower for w in confused_keywords):
            return "confused"

        # Confidence signals
        confident_keywords = ["easy", "i got it", "makes sense", "understood", "clear now", "solved"]
        if any(w in msg_lower for w in confident_keywords):
            return "confident"

        # Curiosity signals
        curious_keywords = ["why", "how does", "what if", "can you explain", "tell me more", "wondering"]
        if any(w in msg_lower for w in curious_keywords):
            return "curious"

        return "neutral"


# ---------------------------------------------------------------------------
# 2. Base and Concrete LearningAgent
# ---------------------------------------------------------------------------

class BaseLearningAgent(ABC):
    """Abstract interface for long-term learner model state retrieval and strategy."""

    @abstractmethod
    def run(
        self,
        student_id: Optional[str],
        target_concept: Optional[str] = None,
        course_id: Optional[int] = None,
        lecture_id: Optional[int] = None,
        last_answer_correct: Optional[bool] = None,
        consecutive_failures: int = 0,
        hint_budget_remaining: Optional[int] = None
    ) -> LearningContext:
        pass


class LearningAgent(BaseLearningAgent):
    """
    Reads persistent LearnerState from LearnerStore and resolves a
    grounded TeachingStrategy using ContextResolver.
    """

    def __init__(
        self,
        learner_store: Optional[BaseLearnerStateStore] = None,
        context_resolver: Optional[ContextResolver] = None
    ) -> None:
        self.learner_store = learner_store or InMemoryLearnerStateStore()
        self.context_resolver = context_resolver or ContextResolver()

    def run(
        self,
        student_id: Optional[str],
        target_concept: Optional[str] = None,
        course_id: Optional[int] = None,
        lecture_id: Optional[int] = None,
        last_answer_correct: Optional[bool] = None,
        consecutive_failures: int = 0,
        hint_budget_remaining: Optional[int] = None
    ) -> LearningContext:
        learner_state: Optional[LearnerState] = None
        target_mastery: Optional[float] = None
        active_misconceptions = []
        behavior_summary = None

        if student_id:
            learner_state = self.learner_store.load(str(student_id))

        if learner_state:
            active_misconceptions = list(learner_state.misconceptions)
            if target_concept and target_concept in learner_state.concept_mastery:
                target_mastery = learner_state.concept_mastery[target_concept].mastery

            behavior_summary = {
                "hints_per_session": learner_state.behavior.hints_per_session,
                "avg_persistence": learner_state.behavior.avg_persistence,
                "engagement_score": learner_state.behavior.engagement_score,
                "sessions_total": learner_state.behavior.sessions_total,
            }

        # Resolve teaching strategy
        resolved = self.context_resolver.resolve(
            learner_state=learner_state,
            target_concept=target_concept,
            course_id=course_id,
            lecture_id=lecture_id,
            last_answer_correct=last_answer_correct,
            consecutive_failures=consecutive_failures,
            hint_budget_remaining=hint_budget_remaining
        )
        strategy = resolved.learning_context.teaching_strategy if hasattr(resolved, "learning_context") and resolved.learning_context else resolved

        return LearningContext(
            student_id=str(student_id) if student_id else None,
            learner_state=learner_state,
            target_concept=target_concept,
            target_mastery=target_mastery,
            teaching_strategy=strategy,
            active_misconceptions=active_misconceptions,
            behavior_summary=behavior_summary
        )



# ---------------------------------------------------------------------------
# 3. Base and Concrete KnowledgeAgent
# ---------------------------------------------------------------------------

class BaseKnowledgeAgent(ABC):
    """Abstract interface for RAG curriculum retrieval."""

    @abstractmethod
    def run(
        self,
        query: str,
        course_id: Optional[int] = None,
        lecture_id: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> KnowledgeContext:
        pass


class KnowledgeAgent(BaseKnowledgeAgent):
    """
    Queries KnowledgeSource for relevant textbook/lecture chunks and produces
    citations for grounded tutor responses.
    """

    def __init__(
        self,
        knowledge_source: Optional[KnowledgeSource] = None
    ) -> None:
        self.knowledge_source = knowledge_source

    def run(
        self,
        query: str,
        course_id: Optional[int] = None,
        lecture_id: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> KnowledgeContext:
        if not self.knowledge_source or not query:
            return KnowledgeContext(
                chunks=[],
                citations=[],
                knowledge_source_used=None,
                query=query
            )

        combined_filters = dict(filters or {})
        if course_id is not None:
            combined_filters["course_id"] = course_id
        if lecture_id is not None:
            combined_filters["lecture_id"] = lecture_id

        try:
            chunks = self.knowledge_source.retrieve(query=query, filters=combined_filters)
            source_name = type(self.knowledge_source).__name__
        except Exception as exc:
            logger.error("Knowledge retrieval error: %s", exc)
            return KnowledgeContext(chunks=[], citations=[], knowledge_source_used=None, query=query)

        citations: List[SourceCitation] = []
        for chunk in chunks:
            meta = chunk.metadata or {}
            lec_id = meta.get(
                "lecture_id",
                int(chunk.source_id) if isinstance(chunk.source_id, int) or (isinstance(chunk.source_id, str) and chunk.source_id.isdigit()) else (lecture_id or 0)
            )
            snippet = chunk.content[:150] + "..." if len(chunk.content) > 150 else chunk.content
            citations.append(
                SourceCitation(
                    lecture_id=lec_id,
                    title=chunk.source_title,
                    chunk_id=meta.get("chunk_id"),
                    snippet=snippet,
                    relevance_score=meta.get("relevance_score") or meta.get("hybrid_score")
                )
            )

        return KnowledgeContext(
            chunks=chunks,
            citations=citations,
            knowledge_source_used=source_name,
            query=query
        )


# ---------------------------------------------------------------------------
# 4. ContextOrchestrator
# ---------------------------------------------------------------------------

class ContextOrchestrator:
    """
    Coordinates concurrent execution of ContextAgent, LearningAgent, and KnowledgeAgent.
    Merges their independent outputs into a single consolidated `OrchestratedContext`.
    """

    def __init__(
        self,
        context_agent: Optional[BaseContextAgent] = None,
        learning_agent: Optional[BaseLearningAgent] = None,
        knowledge_agent: Optional[BaseKnowledgeAgent] = None,
        max_workers: int = 3
    ) -> None:
        self.context_agent = context_agent or ContextAgent()
        self.learning_agent = learning_agent or LearningAgent()
        self.knowledge_agent = knowledge_agent or KnowledgeAgent()
        self.max_workers = max_workers

    def orchestrate(
        self,
        student_message: str,
        session_id: str,
        student_id: Optional[str] = None,
        course_id: Optional[int] = None,
        lecture_id: Optional[int] = None,
        target_concept: Optional[str] = None,
        history_override: Optional[List[ChatMessage]] = None,
        pedagogy_state_override: Optional[PedagogyState] = None,
        last_answer_correct: Optional[bool] = None,
        consecutive_failures: int = 0,
        hint_budget_remaining: Optional[int] = None,
        knowledge_filters: Optional[Dict[str, Any]] = None,
    ) -> OrchestratedContext:
        """
        Executes all three sub-agents in parallel and merges into OrchestratedContext.
        """
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 1. Dispatch ContextAgent task
            future_context = executor.submit(
                self.context_agent.run,
                session_id=session_id,
                student_message=student_message,
                history_override=history_override,
                pedagogy_state_override=pedagogy_state_override
            )

            # 2. Dispatch LearningAgent task
            future_learning = executor.submit(
                self.learning_agent.run,
                student_id=student_id,
                target_concept=target_concept,
                course_id=course_id,
                lecture_id=lecture_id,
                last_answer_correct=last_answer_correct,
                consecutive_failures=consecutive_failures,
                hint_budget_remaining=hint_budget_remaining
            )

            # 3. Dispatch KnowledgeAgent task
            future_knowledge = executor.submit(
                self.knowledge_agent.run,
                query=student_message,
                course_id=course_id,
                lecture_id=lecture_id,
                filters=knowledge_filters
            )

            # Gather results with resilient fallback
            try:
                session_ctx = future_context.result()
            except Exception as exc:
                logger.error("ContextAgent failed during orchestration: %s", exc)
                session_ctx = SessionContext(session_id=session_id)

            try:
                learning_ctx = future_learning.result()
            except Exception as exc:
                logger.error("LearningAgent failed during orchestration: %s", exc)
                learning_ctx = LearningContext(student_id=student_id)

            try:
                knowledge_ctx = future_knowledge.result()
            except Exception as exc:
                logger.error("KnowledgeAgent failed during orchestration: %s", exc)
                knowledge_ctx = KnowledgeContext(query=student_message)

        return OrchestratedContext(
            student_message=student_message,
            session_context=session_ctx,
            learning_context=learning_ctx,
            knowledge_context=knowledge_ctx,
            course_id=course_id,
            lecture_id=lecture_id,
        )

    def orchestrate_request(self, request: AIChatRequest) -> OrchestratedContext:
        """Convenience method accepting an AIChatRequest."""
        session_id = request.session_id or "default_session"
        student_id = str(request.student_id) if request.student_id is not None else None

        return self.orchestrate(
            student_message=request.message,
            session_id=session_id,
            student_id=student_id,
            course_id=request.course_id,
            lecture_id=request.lecture_id,
            history_override=request.conversation_history,
            hint_budget_remaining=request.hint_level
        )
