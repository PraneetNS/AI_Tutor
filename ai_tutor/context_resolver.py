"""
context_resolver.py
--------------------
ContextResolver: High-concurrency context resolution and pedagogical strategy engine.

Key Features:
1. Concurrent 3-way fan-out fetch:
   - `get_learner_state(user_id)`: Postgres/Redis-cached long-term mastery.
   - `get_session_state(session_id)`: Redis short-term session state.
   - `get_lesson_metadata(course_id)`: Course/Lesson curriculum catalog.
2. Per-fetch timeouts (~150ms):
   - If a fetch times out or errors, falls back to the last cached snapshot.
   - Marks the merged context with `possibly_stale: {source: True}`.
3. Conservative Strategy Adaptation:
   - Downstream StrategyEngine suppresses 'challenge' / 'advance' when mastery is stale.
4. Conditional RAG Gating:
   - Queries KnowledgeSource RAG ONLY if `course_id` is present.
5. Final Prompt Assembly via BudgetManager:
   - Compacts assembled prompt within token limits.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple, Union

from .budget_manager import BudgetManager
from .knowledge_source import KnowledgeSource, MockKnowledgeSource
from .learner_store import BaseLearnerStateStore, InMemoryLearnerStateStore
from .models import (
    ChatMessage,
    Chunk,
    CurriculumPosition,
    KnowledgeContext,
    LearnerState,
    LearningContext,
    Misconception,
    OrchestratedContext,
    PedagogyState,
    RootCauseDiagnosis,
    SessionContext,
    SourceCitation,
    StrategyAction,
    TeachingStrategy,
)
from .session_store import BaseSessionStore, InMemorySessionStore
from .tutor_core import DEFAULT_SYSTEM_PROMPT

if TYPE_CHECKING:
    from .concept_graph import ConceptGraph
    from .strategy_engine import StrategyEngine

logger = logging.getLogger("ai_tutor.context_resolver")


class ContextResolver:
    """
    Orchestrates high-concurrency context resolution with bounded timeouts,
    staleness detection, conservative pedagogical fallbacks, conditional RAG,
    and BudgetManager prompt assembly.
    """

    def __init__(
        self,
        learner_store: Optional[BaseLearnerStateStore] = None,
        session_store: Optional[BaseSessionStore] = None,
        knowledge_source: Optional[KnowledgeSource] = None,
        concept_graph: Optional["ConceptGraph"] = None,
        strategy_engine: Optional["StrategyEngine"] = None,
        budget_manager: Optional[BudgetManager] = None,
        fetch_timeout_ms: int = 150,
        default_hint_budget: int = 3,
        quiz_mastery_threshold: float = 0.8,
        challenge_mastery_threshold: float = 0.9,
        explain_failure_threshold: int = 2,
    ) -> None:
        self.learner_store = learner_store or InMemoryLearnerStateStore()
        self.session_store = session_store or InMemorySessionStore()
        self.knowledge_source = knowledge_source or MockKnowledgeSource()
        self.concept_graph = concept_graph
        self.strategy_engine = strategy_engine
        self.budget_manager = budget_manager or BudgetManager()
        self.fetch_timeout_s = max(0.01, fetch_timeout_ms / 1000.0)

        self.default_hint_budget = default_hint_budget
        self.quiz_mastery_threshold = quiz_mastery_threshold
        self.challenge_mastery_threshold = challenge_mastery_threshold
        self.explain_failure_threshold = explain_failure_threshold

        # Local caches for timeout / error fallbacks
        self._cached_learner_states: Dict[str, LearnerState] = {}
        self._cached_session_states: Dict[str, SessionContext] = {}
        self._cached_lesson_metadata: Dict[int, Dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # 1. Individual Concurrent Fetch Routines
    # ------------------------------------------------------------------

    def get_learner_state(
        self,
        user_id: Union[int, str],
        simulated_delay: float = 0.0
    ) -> LearnerState:
        """Fetch long-term learner state (Postgres / Redis cached)."""
        if simulated_delay > 0:
            time.sleep(simulated_delay)

        uid_str = str(user_id)
        state = self.learner_store.load(uid_str)
        if state is None:
            state = LearnerState(student_id=uid_str)

        # Update last known cache
        self._cached_learner_states[uid_str] = state
        return state


    def get_session_state(
        self,
        session_id: str,
        simulated_delay: float = 0.0
    ) -> SessionContext:
        """Fetch short-term session state (Redis)."""
        if simulated_delay > 0:
            time.sleep(simulated_delay)

        session_data = self.session_store.get_session(session_id)
        ctx = SessionContext(
            session_id=session_id,
            recent_messages=session_data.messages[-10:],
            pedagogy_state=session_data.pedagogy_state,
            detected_mood="neutral",
            turn_count=len(session_data.messages),
        )


        # Update last known cache
        self._cached_session_states[session_id] = ctx
        return ctx

    def get_lesson_metadata(
        self,
        course_id: Optional[int],
        simulated_delay: float = 0.0
    ) -> Dict[str, Any]:
        """Fetch course & lesson curriculum metadata (Course DB)."""
        if course_id is None:
            return {}

        if simulated_delay > 0:
            time.sleep(simulated_delay)

        meta = {
            "course_id": course_id,
            "title": f"Course {course_id} Syllabus",
            "active_module": "Foundations",
        }
        self._cached_lesson_metadata[course_id] = meta
        return meta

    # ------------------------------------------------------------------
    # 2. Concurrent Resolution Orchestrator
    # ------------------------------------------------------------------

    def resolve(
        self,
        user_id: Optional[Union[int, str]] = None,
        session_id: Optional[str] = None,
        course_id: Optional[int] = None,
        student_message: str = "",
        target_concept: Optional[str] = None,
        lecture_id: Optional[int] = None,
        last_answer_correct: Optional[bool] = None,
        consecutive_failures: int = 0,
        hint_budget_remaining: Optional[int] = None,
        learner_state: Optional[LearnerState] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> OrchestratedContext:
        """
        Fires three concurrent fetches (~150ms timeout each), handles fallbacks,
        marks possibly_stale flags, executes conservative strategy planning,
        conditionally triggers RAG if course_id is present, and assembles final prompt.
        """
        uid = user_id or (learner_state.student_id if learner_state else "guest_user")
        uid_str = str(uid)
        sid = session_id or f"sess_{uid_str}"
        possibly_stale: Dict[str, bool] = {}

        # --------------------------------------------------------------
        # Step A: 3-Way Concurrent Fetch with Timeout
        # --------------------------------------------------------------
        fetched_learner_state: Optional[LearnerState] = None
        fetched_session_ctx: Optional[SessionContext] = None
        fetched_lesson_meta: Optional[Dict[str, Any]] = None

        if learner_state is not None:
            # Explicit override supplied
            fetched_learner_state = learner_state
            self._cached_learner_states[uid_str] = learner_state

        with ThreadPoolExecutor(max_workers=3) as executor:
            future_learner = None
            if fetched_learner_state is None:
                future_learner = executor.submit(self.get_learner_state, uid)
            future_session = executor.submit(self.get_session_state, sid)
            future_lesson = executor.submit(self.get_lesson_metadata, course_id)

            # 1. Await Learner State
            if future_learner is not None:
                try:
                    fetched_learner_state = future_learner.result(timeout=self.fetch_timeout_s)
                except Exception as e:
                    logger.warning("[ContextResolver] Learner state fetch timed out or failed (%s). Using cache.", e)
                    possibly_stale["learner_state"] = True
                    fetched_learner_state = self._cached_learner_states.get(
                        uid_str,
                        LearnerState(student_id=uid_str)
                    )

            # 2. Await Session State
            try:
                fetched_session_ctx = future_session.result(timeout=self.fetch_timeout_s)
            except Exception as e:
                logger.warning("[ContextResolver] Session state fetch timed out or failed (%s). Using cache.", e)
                possibly_stale["session_state"] = True
                fetched_session_ctx = self._cached_session_states.get(
                    sid,
                    SessionContext(session_id=sid, pedagogy_state=PedagogyState())
                )

            # 3. Await Lesson Metadata
            try:
                fetched_lesson_meta = future_lesson.result(timeout=self.fetch_timeout_s)
            except Exception as e:
                logger.warning("[ContextResolver] Lesson metadata fetch timed out or failed (%s). Using cache.", e)
                possibly_stale["lesson_metadata"] = True
                fetched_lesson_meta = self._cached_lesson_metadata.get(course_id or 0, {})

        # Ensure all states exist
        active_learner_state = fetched_learner_state or LearnerState(student_id=uid_str)
        active_session_ctx = fetched_session_ctx or SessionContext(session_id=sid)

        # --------------------------------------------------------------
        # Step B: Strategy Resolution (Conservative on Staleness)
        # --------------------------------------------------------------
        strategy = self._resolve_strategy(
            learner_state=active_learner_state,
            target_concept=target_concept,
            course_id=course_id,
            lecture_id=lecture_id,
            last_answer_correct=last_answer_correct,
            consecutive_failures=consecutive_failures,
            hint_budget_remaining=hint_budget_remaining,
            possibly_stale=possibly_stale,
            metadata=metadata,
        )

        # --------------------------------------------------------------
        # Step C: Conditional RAG (Only if course_id is present)
        # --------------------------------------------------------------
        chunks: List[Chunk] = []
        citations: List[SourceCitation] = []
        source_used: Optional[str] = None

        if course_id is not None and course_id > 0:
            try:
                search_query = student_message or target_concept or f"Course {course_id} lesson"
                chunks = self.knowledge_source.retrieve(
                    query=search_query,
                    filters={"course_id": course_id, "lecture_id": lecture_id}
                )
                citations = [
                    SourceCitation(
                        lecture_id=c.source_id,
                        title=c.source_title,
                        snippet=c.content[:150]
                    )
                    for c in chunks
                ]
                source_used = type(self.knowledge_source).__name__
            except Exception as e:
                logger.warning("[ContextResolver] RAG retrieval failed: %s", e)


        knowledge_ctx = KnowledgeContext(
            chunks=chunks,
            citations=citations,
            knowledge_source_used=source_used,
            query=student_message if course_id else None
        )

        # --------------------------------------------------------------
        # Step D: Assemble LearningContext & OrchestratedContext
        # --------------------------------------------------------------
        learning_ctx = LearningContext(
            student_id=uid_str,
            learner_state=active_learner_state,
            target_concept=target_concept or strategy.target_concept,
            target_mastery=strategy.target_mastery,
            teaching_strategy=strategy,
            active_misconceptions=active_learner_state.misconceptions,
        )

        orch_context = OrchestratedContext(
            student_message=student_message,
            session_context=active_session_ctx,
            learning_context=learning_ctx,
            knowledge_context=knowledge_ctx,
            course_id=course_id,
            lecture_id=lecture_id,
            possibly_stale=possibly_stale,
        )

        # --------------------------------------------------------------
        # Step E: Prompt Assembly via BudgetManager
        # --------------------------------------------------------------
        try:
            sections = orch_context.to_prompt_sections()
            sections["system"] = DEFAULT_SYSTEM_PROMPT
            if possibly_stale:
                sections["system"] += f"\n[STALENESS WARNING: Sources {list(possibly_stale.keys())} are potentially stale.]"

            budgets = {
                "system": (150, 400),
                "learner_state": (30, 150),
                "rag_knowledge": (100, 500),
                "teaching_strategy": (20, 100),
                "conversation_history": (50, 400),
            }
            assembled_prompt, _ = self.budget_manager.assemble(sections, budgets)
            orch_context.assembled_prompt = assembled_prompt
        except Exception as e:
            logger.warning("[ContextResolver] BudgetManager assembly warning: %s", e)

        return orch_context

    # ------------------------------------------------------------------
    # 3. Strategy Decision Logic (Conservative Fallback)
    # ------------------------------------------------------------------

    def _resolve_strategy(
        self,
        learner_state: LearnerState,
        target_concept: Optional[str] = None,
        course_id: Optional[int] = None,
        lecture_id: Optional[int] = None,
        last_answer_correct: Optional[bool] = None,
        consecutive_failures: int = 0,
        hint_budget_remaining: Optional[int] = None,
        possibly_stale: Optional[Dict[str, bool]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> TeachingStrategy:
        """
        Determines teaching strategy. If learner_state is possibly stale,
        conservatively avoids 'challenge' / 'advance' recommendations.
        """
        budget = hint_budget_remaining if hint_budget_remaining is not None else self.default_hint_budget
        is_stale_mastery = bool(possibly_stale and possibly_stale.get("learner_state"))

        # ConceptGraph computations
        curriculum_position: Optional[CurriculumPosition] = None
        root_cause_diagnosis: Optional[RootCauseDiagnosis] = None

        if self.concept_graph and target_concept:
            try:
                curriculum_position = self.concept_graph.compute_curriculum_position(
                    learner_state=learner_state,
                    current_concept=target_concept
                )
            except Exception:
                pass

            if consecutive_failures >= 1:
                try:
                    diagnosis = self.concept_graph.diagnose_root_cause(
                        struggling_concept=target_concept,
                        learner_state=learner_state
                    )
                    if diagnosis.likely_root_gap:
                        root_cause_diagnosis = diagnosis
                except Exception:
                    pass

        # Extract current mastery
        current_mastery: Optional[float] = None
        if target_concept and target_concept in learner_state.concept_mastery:
            current_mastery = learner_state.concept_mastery[target_concept].mastery

        # Misconception check
        highest_misconception: Optional[Misconception] = None
        if learner_state.misconceptions:
            matched = [
                m for m in learner_state.misconceptions
                if target_concept and (m.concept.lower() in target_concept.lower() or target_concept.lower() in m.concept.lower())
            ]
            candidates = matched if matched else learner_state.misconceptions
            highest_misconception = max(candidates, key=lambda m: (m.confidence, m.hit_count))

        # Pedagogical Decision Rules
        recommendation: StrategyAction
        rationale: str
        strategy_category: str
        strategy_type: str

        if consecutive_failures >= self.explain_failure_threshold:
            recommendation = StrategyAction.EXPLAIN
            strategy_category = "explanation"
            strategy_type = "worked_example"
            rationale = f"Student has {consecutive_failures} consecutive incorrect attempts. Switching to direct explanation."

        elif last_answer_correct is False and budget > 0:
            recommendation = StrategyAction.HINT
            strategy_category = "scaffolding"
            strategy_type = "leading_question"
            rationale = f"Incorrect answer on '{target_concept}'. Providing scaffolding hint (budget remaining: {budget})."

        elif (
            current_mastery is not None
            and current_mastery > self.challenge_mastery_threshold
            and not is_stale_mastery  # CONSERVATIVE: Do not advance on stale data
        ):
            recommendation = StrategyAction.CHALLENGE
            strategy_category = "challenge"
            strategy_type = "transfer_problem"
            rationale = f"High mastery on '{target_concept}' ({current_mastery:.2f} > {self.challenge_mastery_threshold}). Pushing learner to advanced application or next syllabus concept."


        elif current_mastery is not None and current_mastery > self.quiz_mastery_threshold:
            recommendation = StrategyAction.QUIZ
            strategy_category = "assessment"
            strategy_type = "free_response"
            rationale = (
                f"Mastery reached {current_mastery:.2f}."
                + (" (Verifying understanding conservatively due to stale cache.)" if is_stale_mastery else " Administering formative quiz.")
            )

        else:
            recommendation = StrategyAction.GUIDE
            strategy_category = "scaffolding"
            strategy_type = "leading_question"
            rationale = f"Guiding student Socratic inquiry on '{target_concept or 'active topic'}'."
            if is_stale_mastery:
                rationale += " [Mastery data possibly stale; defaulting to conservative guidance]."

        return TeachingStrategy(
            recommendation=recommendation,
            target_concept=target_concept,
            target_mastery=current_mastery,
            misconception_to_address=highest_misconception,
            hint_budget_remaining=budget,
            consecutive_failures=consecutive_failures,
            rationale=rationale,
            course_id=course_id,
            lecture_id=lecture_id,
            strategy_category=strategy_category,
            strategy_type=strategy_type,
            curriculum_position=curriculum_position,
            root_cause_diagnosis=root_cause_diagnosis,
            metadata=metadata or {},
        )
