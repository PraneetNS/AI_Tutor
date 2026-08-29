"""
tutor_reasoner.py
-----------------
TutorReasoner: 7-Stage Pedagogical Cognitive Reasoning Architecture.

Explicit Steps (independently callable / mockable / replayable):
1. Observe(message, history) -> captures raw input, classifies via IntentClassifier router.
2. Model(user_id) -> reads LearnerState + ConceptGraph position (cached read).
3. Plan() -> calls StrategyEngine, returns TeachingStrategy per taxonomy & effectiveness.
4. Teach() -> dispatches to TutorAgent / QuizAgent / AssessmentAgent per strategy_category.
5. Assess() -> calls AnswerEvaluator on student response.
6. Adapt() -> if not mastered, loops back to Plan() with 'not mastered' branch (picks a
   different strategy_type within the category, never repeating a strategy_type that
   failed 2+ times consecutively for this concept).
7. Remember() -> emits learning event, updates LearnerState and learner_strategy_effectiveness
   asynchronously so it never blocks response latency.

Step Execution Traces:
Every step logs structured execution records into `step_trace` for full session replay and eval.
"""

from __future__ import annotations

import logging
import threading
import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Set, Tuple, Union

from .answer_evaluator import AnswerEvaluator
from .budget_manager import BudgetManager
from .classifier import IntentClassifier
from .concept_graph import ConceptGraph, create_ml_concept_graph
from .llm_client import BaseLLMClient, MockLLMClient
from .models import (
    ChatMessage,
    ClassificationResult,
    CurriculumPosition,
    EvaluationResult,
    LearnerState,
    LearningEvent,
    LearningEventType,
    OrchestratedContext,
    PedagogyMode,
    PedagogyState,
    ReasonerResult,
    Role,
    SourceCitation,
    StrategyAction,
    StrategyEffectivenessRecord,
    TeachingStrategy,
)
from .strategy_engine import (
    STRATEGY_TAXONOMY,
    BaseStrategyEffectivenessStore,
    InMemoryStrategyEffectivenessStore,
    StrategyEngine,
)
from .tutor_core import DEFAULT_SYSTEM_PROMPT

if TYPE_CHECKING:
    from .assessment_agent import AssessmentAgent, BaseAssessmentSubAgent
    from .quiz_agent import BaseQuizSubAgent, QuizAgent

logger = logging.getLogger("ai_tutor.tutor_reasoner")


# ---------------------------------------------------------------------------
# 1. Sub-Agent Interfaces and Concrete Implementations
# ---------------------------------------------------------------------------

class BaseTutorSubAgent(ABC):
    """Sub-agent responsible for Socratic dialogue, scaffolding hints, and direct explanations."""

    @abstractmethod
    def generate(
        self,
        context: OrchestratedContext,
        strategy: TeachingStrategy
    ) -> str:
        pass


class TutorAgent(BaseTutorSubAgent):
    """
    Standard teaching agent: delivers Socratic guidance, progressive hints,
    or direct explanations based on strategy recommendation and budget constraints.
    """

    def __init__(
        self,
        llm_client: Optional[BaseLLMClient] = None,
        budget_manager: Optional[BudgetManager] = None
    ) -> None:
        self.llm_client = llm_client or MockLLMClient()
        self.budget_manager = budget_manager or BudgetManager()

    def generate(
        self,
        context: OrchestratedContext,
        strategy: TeachingStrategy
    ) -> str:
        sections = context.to_prompt_sections()
        sections["system"] = DEFAULT_SYSTEM_PROMPT

        budgets = {
            "system": (150, 400),
            "learner_state": (30, 150),
            "rag_knowledge": (100, 500),
            "teaching_strategy": (20, 100),
            "conversation_history": (50, 400),
        }

        assembled_prompt, _ = self.budget_manager.assemble(sections, budgets)

        messages = list(context.session_context.recent_messages) + [
            ChatMessage(role=Role.USER, content=context.student_message)
        ]

        output = self.llm_client.generate(
            system_prompt=assembled_prompt,
            messages=messages,
            current_state=context.session_context.pedagogy_state
        )

        return output.answer


# Import remaining subagents
from .assessment_agent import AssessmentAgent
from .quiz_agent import QuizAgent


# ---------------------------------------------------------------------------
# 2. Step Log Entry for Session Replay
# ---------------------------------------------------------------------------

class StepLog:
    """Audit log entry capturing individual step outputs for debug replay."""

    def __init__(self, step_number: int, step_name: str, payload: Dict[str, Any]) -> None:
        self.step_number = step_number
        self.step_name = step_name
        self.payload = payload
        self.timestamp = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_number": self.step_number,
            "step_name": self.step_name,
            "timestamp": self.timestamp,
            "payload": self.payload,
        }


# ---------------------------------------------------------------------------
# 3. TutorReasoner: 7-Stage Architecture
# ---------------------------------------------------------------------------

class TutorReasoner:
    """
    7-Stage Pedagogical Reasoning Orchestrator:
    1. Observe(message) -> Intent Classification via Router.
    2. Model(user_id) -> Cached state retrieval & CurriculumPosition.
    3. Plan() -> Strategy selection with empirical tie-breaking.
    4. Teach() -> Sub-agent dispatch (Tutor, Quiz, Assessment).
    5. Assess() -> Answer evaluation & partial credit scoring.
    6. Adapt() -> Fallback loop picking alternative strategy_types on non-mastery.
    7. Remember() -> Asynchronous learning event and effectiveness persistence.
    """

    def __init__(
        self,
        tutor_agent: Optional[BaseTutorSubAgent] = None,
        quiz_agent: Optional[Any] = None,
        assessment_agent: Optional[Any] = None,
        router: Optional[IntentClassifier] = None,
        strategy_engine: Optional[StrategyEngine] = None,
        answer_evaluator: Optional[AnswerEvaluator] = None,
        concept_graph: Optional[ConceptGraph] = None,
    ) -> None:
        self.tutor_agent = tutor_agent or TutorAgent()
        self.quiz_agent = quiz_agent or QuizAgent()
        self.assessment_agent = assessment_agent or AssessmentAgent()
        self.router = router or IntentClassifier()
        self.strategy_engine = strategy_engine or StrategyEngine()
        self.answer_evaluator = answer_evaluator or AnswerEvaluator()
        self.concept_graph = concept_graph or create_ml_concept_graph()

        # In-memory cache for Step 2 (Model) to avoid redundant DB reads
        self._learner_state_cache: Dict[str, Tuple[float, LearnerState]] = {}
        self._cache_ttl_seconds = 300.0  # 5 minutes TTL

        # Session step execution trace
        self.step_traces: List[StepLog] = []

    def _log_step(self, step_number: int, step_name: str, payload: Dict[str, Any]) -> StepLog:
        log_entry = StepLog(step_number=step_number, step_name=step_name, payload=payload)
        self.step_traces.append(log_entry)
        logger.debug("[TutorReasoner Step %d: %s] %s", step_number, step_name, payload)
        return log_entry

    def get_session_replay(self) -> List[Dict[str, Any]]:
        """Returns the chronological execution logs of all 7 steps for replay/eval."""
        return [entry.to_dict() for entry in self.step_traces]

    # ------------------------------------------------------------------
    # Step 1: OBSERVE
    # ------------------------------------------------------------------

    def observe(
        self,
        message: str,
        history: Optional[List[ChatMessage]] = None
    ) -> ClassificationResult:
        """
        Step 1: Captures raw input and classifies intent and mood via Router.
        """
        classification = self.router.classify(student_message=message, conversation_history=history)
        self._log_step(1, "Observe", {
            "raw_message": message,
            "label": classification.label,
            "confidence": classification.confidence,
            "rationale": classification.rationale,
            "flagged_for_review": classification.flagged_for_review,
        })
        return classification



    # ------------------------------------------------------------------
    # Step 2: MODEL
    # ------------------------------------------------------------------

    def model(
        self,
        user_id: Union[int, str],
        learner_state_override: Optional[LearnerState] = None,
        target_concept: Optional[str] = None,
    ) -> Tuple[LearnerState, Optional[CurriculumPosition]]:
        """
        Step 2: Reads LearnerState + ConceptGraph position (cached read).
        """
        uid_str = str(user_id)
        now = time.time()

        if learner_state_override:
            state = learner_state_override
            self._learner_state_cache[uid_str] = (now, state)
        elif uid_str in self._learner_state_cache and (now - self._learner_state_cache[uid_str][0]) < self._cache_ttl_seconds:
            state = self._learner_state_cache[uid_str][1]
        else:
            state = LearnerState(student_id=uid_str)
            self._learner_state_cache[uid_str] = (now, state)

        curriculum_pos = None
        if self.concept_graph:
            try:
                curriculum_pos = self.concept_graph.compute_curriculum_position(
                    learner_state=state,
                    current_concept=target_concept
                )
            except Exception as e:
                logger.warning("[Step 2: Model] Curriculum position failed: %s", e)

        self._log_step(2, "Model", {
            "user_id": uid_str,
            "mastery_count": len(state.concept_mastery),
            "misconception_count": len(state.misconceptions),
            "target_concept": target_concept,
            "curriculum_position": curriculum_pos.model_dump() if curriculum_pos else None,
        })
        return state, curriculum_pos

    # ------------------------------------------------------------------
    # Step 3: PLAN
    # ------------------------------------------------------------------

    def plan(
        self,
        learner_state: Optional[LearnerState] = None,
        target_concept: Optional[str] = None,
        concept_domain: str = "general",
        user_id: Optional[Union[int, str]] = None,
        consecutive_failures: int = 0,
        last_answer_correct: Optional[bool] = None,
        hint_budget_remaining: Optional[int] = None,
        excluded_strategy_types: Optional[Set[str]] = None,
        course_id: Optional[int] = None,
        lecture_id: Optional[int] = None,
    ) -> TeachingStrategy:
        """
        Step 3: Calls StrategyEngine, returning TeachingStrategy.
        """
        base_strategy = self.strategy_engine.plan(
            learner_state=learner_state,
            concept_graph=self.concept_graph,
            target_concept=target_concept,
            concept_domain=concept_domain,
            user_id=user_id,
            consecutive_failures=consecutive_failures,
            last_answer_correct=last_answer_correct,
            hint_budget_remaining=hint_budget_remaining,
            course_id=course_id,
            lecture_id=lecture_id,
        )

        # If chosen strategy_type is in excluded list (e.g. failed twice), pick next available
        if excluded_strategy_types and base_strategy.strategy_type in excluded_strategy_types:
            cat = base_strategy.strategy_category or "explanation"
            available = [
                st for st in STRATEGY_TAXONOMY.get(cat, ["default"])
                if st not in excluded_strategy_types
            ]
            if available:
                fallback_type = available[0]
                base_strategy.strategy_type = fallback_type
                base_strategy.rationale += f" (Excluded {excluded_strategy_types}; switched to alternative type '{fallback_type}')."

        self._log_step(3, "Plan", {
            "recommendation": base_strategy.recommendation,
            "strategy_category": base_strategy.strategy_category,
            "strategy_type": base_strategy.strategy_type,
            "target_concept": base_strategy.target_concept,
            "rationale": base_strategy.rationale,
            "excluded_types": list(excluded_strategy_types or []),
        })
        return base_strategy

    # ------------------------------------------------------------------
    # Step 4: TEACH
    # ------------------------------------------------------------------

    def teach(
        self,
        context: OrchestratedContext,
        strategy: TeachingStrategy
    ) -> Tuple[str, Optional[Dict[str, Any]], str]:
        """
        Step 4: Dispatches to Tutor/Quiz/Assessment agent per strategy_category and interaction context.
        """
        selected_agent = self.decide(context=context, strategy=strategy)

        if selected_agent == "QuizAgent":
            answer = self.quiz_agent.generate_quiz(context=context, strategy=strategy)
            meta = None
        elif selected_agent == "AssessmentAgent":
            answer, meta = self.assessment_agent.evaluate(context=context, strategy=strategy)
        else:
            answer = self.tutor_agent.generate(context=context, strategy=strategy)
            meta = None

        self._log_step(4, "Teach", {
            "selected_agent": selected_agent,
            "response_snippet": answer[:120] if answer else "",
            "assessment_meta": meta,
        })
        return answer, meta, selected_agent

    # ------------------------------------------------------------------
    # Step 5: ASSESS
    # ------------------------------------------------------------------

    def assess(
        self,
        student_response: str,
        expected_concepts: Union[str, List[str]],
        question_context: Optional[str] = None,
        hints_used: int = 0
    ) -> EvaluationResult:
        """
        Step 5: Calls AnswerEvaluator on the student's reply.
        """
        eval_result = self.answer_evaluator.evaluate(
            response=student_response,
            expected_concepts=expected_concepts,
            question_context=question_context,
            hints_used=hints_used
        )
        self._log_step(5, "Assess", {
            "correct": eval_result.correct,
            "partial_credit": eval_result.partial_credit,
            "is_mastered": eval_result.is_mastered,
            "concepts_touched": eval_result.concepts_touched,
            "misconceptions_detected": eval_result.misconceptions_detected,
        })
        return eval_result

    # ------------------------------------------------------------------
    # Step 6: ADAPT
    # ------------------------------------------------------------------

    def adapt_strategy(
        self,
        eval_result: EvaluationResult,
        current_strategy: TeachingStrategy,
        failure_history_for_concept: Optional[List[str]] = None,
        learner_state: Optional[LearnerState] = None,
        concept_domain: str = "general",
        user_id: Optional[Union[int, str]] = None,
    ) -> Tuple[PedagogyState, Optional[TeachingStrategy]]:
        """
        Step 6: If not mastered, loops back to Plan() with 'not mastered' branch:
        - Picks a different strategy_type within the category (e.g. analogy -> visual).
        - Never repeats a strategy_type that already failed twice consecutively for this concept.
        """
        history = list(failure_history_for_concept or [])
        if not eval_result.is_mastered and current_strategy.strategy_type:
            history.append(current_strategy.strategy_type)

        # Identify strategy types that failed 2+ times consecutively
        excluded_types: Set[str] = set()
        for stype in set(history):
            # Check trailing consecutive occurrences
            count = 0
            for item in reversed(history):
                if item == stype:
                    count += 1
                else:
                    break
            if count >= 2:
                excluded_types.add(stype)

        new_strategy: Optional[TeachingStrategy] = None
        if not eval_result.is_mastered:
            new_strategy = self.plan(
                learner_state=learner_state,
                target_concept=current_strategy.target_concept,
                concept_domain=concept_domain,
                user_id=user_id,
                consecutive_failures=current_strategy.consecutive_failures + 1,
                last_answer_correct=False,
                hint_budget_remaining=max(0, current_strategy.hint_budget_remaining - 1),
                excluded_strategy_types=excluded_types,
            )

        new_pedagogy_state = PedagogyState(
            hint_level=min(5, current_strategy.consecutive_failures + 1) if not eval_result.is_mastered else 0,
            topic=current_strategy.target_concept or "general",
            stuck=not eval_result.is_mastered,
            pedagogy_mode=PedagogyMode.DIRECT if (current_strategy.consecutive_failures >= 1 and not eval_result.is_mastered) else PedagogyMode.SOCRATIC,
        )

        self._log_step(6, "Adapt", {
            "is_mastered": eval_result.is_mastered,
            "failure_history": history,
            "excluded_types": list(excluded_types),
            "new_strategy_type": new_strategy.strategy_type if new_strategy else None,
            "new_hint_level": new_pedagogy_state.hint_level,
        })
        return new_pedagogy_state, new_strategy

    # ------------------------------------------------------------------
    # Step 7: REMEMBER
    # ------------------------------------------------------------------

    def remember(
        self,
        user_id: Union[int, str],
        strategy: TeachingStrategy,
        eval_result: EvaluationResult,
        concept_domain: str = "general",
        async_exec: bool = True
    ) -> Dict[str, Any]:
        """
        Step 7: Emits learning event, triggers LearnerModel update and
        learner_strategy_effectiveness update asynchronously.
        """
        def _persist_task() -> Dict[str, Any]:
            target_concept = strategy.target_concept or "general"
            uid_str = str(user_id)

            # Update Learner State in cache
            if uid_str in self._learner_state_cache:
                _, state = self._learner_state_cache[uid_str]
                if target_concept in state.concept_mastery:
                    cm = state.concept_mastery[target_concept]
                    cm.attempts += 1
                    if eval_result.correct:
                        cm.correct += 1
                    if eval_result.updated_p_known:
                        cm.mastery = eval_result.updated_p_known

            # Update Strategy Effectiveness
            eff_rec = None
            if strategy.strategy_category and strategy.strategy_type:
                eff_rec = self.strategy_engine.record_assess_result(
                    user_id=user_id,
                    strategy_category=strategy.strategy_category,
                    strategy_type=strategy.strategy_type,
                    concept_domain=concept_domain,
                    assess_result="mastered" if eval_result.is_mastered else "in_progress"
                )

            payload = {
                "user_id": str(user_id),
                "concept": target_concept,
                "is_mastered": eval_result.is_mastered,
                "strategy_type": strategy.strategy_type,
                "effectiveness_ratio": eff_rec.effectiveness_ratio if eff_rec else None,
            }
            self._log_step(7, "Remember", payload)
            return payload

        if async_exec:
            thread = threading.Thread(target=_persist_task, daemon=True)
            thread.start()
            return {"status": "dispatched_async", "user_id": str(user_id)}
        else:
            return _persist_task()

    # ------------------------------------------------------------------
    # Backwards-Compatible Helpers
    # ------------------------------------------------------------------

    def decide(
        self,
        context: OrchestratedContext,
        strategy: TeachingStrategy
    ) -> str:
        """Determines which sub-agent should handle the current interaction."""
        student_msg = context.student_message.lower()

        if any(w in student_msg for w in ["quiz me", "test me", "give me a problem", "practice question"]):
            return "QuizAgent"

        if strategy.recommendation == StrategyAction.QUIZ or strategy.recommendation == "quiz":
            return "QuizAgent"

        prior_messages = context.session_context.recent_messages
        if prior_messages:
            last_assistant_msg = next((m for m in reversed(prior_messages) if m.role == Role.ASSISTANT), None)
            if last_assistant_msg:
                last_content = last_assistant_msg.content.lower()
                is_prior_question = any(q in last_content for q in ["?", "try", "what happens", "in your own words", "check:"])
                if is_prior_question and not any(h in student_msg for h in ["hint", "stuck", "don't know", "tell me", "explain"]):
                    return "AssessmentAgent"

        return "TutorAgent"

    def teach_or_assess(
        self,
        selected_agent: str,
        context: OrchestratedContext,
        strategy: TeachingStrategy
    ) -> Tuple[str, Optional[Dict[str, Any]]]:
        """Executes the chosen sub-agent."""
        if selected_agent == "QuizAgent":
            answer = self.quiz_agent.generate_quiz(context=context, strategy=strategy)
            return answer, None
        elif selected_agent == "AssessmentAgent":
            answer, meta = self.assessment_agent.evaluate(context=context, strategy=strategy)
            return answer, meta
        else:
            answer = self.tutor_agent.generate(context=context, strategy=strategy)
            return answer, None

    def adapt(
        self,
        selected_agent: str,
        current_state: PedagogyState,
        strategy: TeachingStrategy,
        assessment_result: Optional[Dict[str, Any]],
        student_message: str,
        detected_mood: str
    ) -> PedagogyState:
        """Legacy adapt helper for state updating."""
        msg_lower = student_message.lower()
        new_hint_level = current_state.hint_level
        new_stuck = current_state.stuck
        target_topic = strategy.target_concept or current_state.topic

        if strategy.recommendation == StrategyAction.HINT or strategy.recommendation == "hint" or "hint" in msg_lower:
            new_hint_level = min(5, current_state.hint_level + 1)
            new_stuck = True
        elif strategy.recommendation == StrategyAction.EXPLAIN or strategy.recommendation == "explain":
            new_stuck = True
        elif assessment_result and assessment_result.get("correct") is True:
            new_hint_level = 0
            new_stuck = False
        elif detected_mood in ("frustrated", "confused") or strategy.consecutive_failures >= 2:
            new_stuck = True

        new_mode = PedagogyMode.DIRECT if (strategy.recommendation == StrategyAction.EXPLAIN or strategy.recommendation == "explain") else PedagogyMode.SOCRATIC

        return PedagogyState(
            hint_level=new_hint_level,
            topic=target_topic,
            stuck=new_stuck,
            pedagogy_mode=new_mode
        )

    # ------------------------------------------------------------------
    # Full Turn Coordinator
    # ------------------------------------------------------------------

    def reason_turn(
        self,
        context: OrchestratedContext,
        strategy_override: Optional[TeachingStrategy] = None
    ) -> ReasonerResult:
        """Executes the complete reasoning turn through the 7-step architecture."""
        # 1. Observe
        _classification = self.observe(
            message=context.student_message,
            history=context.session_context.recent_messages
        )

        # 2. Model
        learner_state = context.learning_context.learner_state if hasattr(context.learning_context, "learner_state") else None
        state, curriculum_pos = self.model(
            user_id=context.learning_context.student_id,
            learner_state_override=learner_state,
            target_concept=context.learning_context.target_concept
        )

        # 3. Plan
        strategy = (
            strategy_override
            or context.learning_context.teaching_strategy
            or self.plan(
                learner_state=state,
                target_concept=context.learning_context.target_concept,
                user_id=context.learning_context.student_id,
                consecutive_failures=0
            )
        )

        # 4. Teach
        answer, assessment_meta, selected_agent = self.teach(
            context=context,
            strategy=strategy
        )

        # 5. Adapt (Update state)
        updated_state = self.adapt(
            selected_agent=selected_agent,
            current_state=context.session_context.pedagogy_state,
            strategy=strategy,
            assessment_result=assessment_meta,
            student_message=context.student_message,
            detected_mood=context.session_context.detected_mood
        )

        citations = list(context.knowledge_context.citations)

        return ReasonerResult(
            answer=answer,
            selected_agent=selected_agent,
            pedagogy_state=updated_state,
            strategy_applied=strategy,
            sources=citations,
            assessment_result=assessment_meta,
            metadata={
                "detected_mood": context.session_context.detected_mood,
                "hint_budget_remaining": strategy.hint_budget_remaining
            }
        )
