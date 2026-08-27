"""
learner_model.py
----------------
LearnerModelEngine and internal update engines:
1. KnowledgeTracer: Bayesian Knowledge Tracing (BKT) on answer submissions.
2. MisconceptionEngine: Pattern matching against a seed library of known misconceptions.
3. BehavioralModel: Rolling averages for hints used, persistence, and engagement.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from .models import (
    BehaviorProfile,
    BKTParams,
    ConceptMastery,
    LearnerState,
    LearningEvent,
    LearningEventType,
    Misconception,
)
from .event_bus import BaseEventBus
from .learner_store import BaseLearnerStateStore, InMemoryLearnerStateStore

logger = logging.getLogger("ai_tutor.learner_model")


# ---------------------------------------------------------------------------
# 1. Knowledge Tracing (BKTUpdater & KnowledgeTracer)
# ---------------------------------------------------------------------------

class BKTUpdater:
    """
    Pure-math Bayesian Knowledge Tracing (BKT) updater engine.
    Zero dependency on any LLM call — testable in isolation.

    Two-step update formula:
    1. Posterior given evidence:
       - If hints_used > 0, treat as incorrect regardless of `correct` flag.
       - If correct (and hints_used == 0):
         P(L|correct) = [P(L) * (1 - P(S))] / [P(L) * (1 - P(S)) + (1 - P(L)) * P(G)]
       - If incorrect (or hints_used > 0):
         P(L|incorrect) = [P(L) * P(S)] / [P(L) * P(S) + (1 - P(L)) * (1 - P(G))]

    2. Apply learning transition:
       P(L_new) = P(L|evidence) + (1 - P(L|evidence)) * P(T)
    """

    DEFAULT_PARAMS: BKTParams = BKTParams(
        p_l0=0.30,
        p_t=0.10,
        p_g=0.25,
        p_s=0.10,
    )

    def __init__(
        self,
        config_table: Optional[Dict[str, BKTParams]] = None,
        default_params: Optional[BKTParams] = None,
    ) -> None:
        self.default_params = default_params or self.DEFAULT_PARAMS
        self.config_table: Dict[str, BKTParams] = dict(config_table or {})

    def get_params(self, concept_id: Optional[str] = None) -> BKTParams:
        """
        Loads per-concept BKTParams from config table.
        Falls back to defaults (P_L0=0.3, P_T=0.1, P_G=0.25, P_S=0.1)
        if a concept has no tuned params yet.
        """
        if concept_id and concept_id in self.config_table:
            return self.config_table[concept_id]
        return self.default_params

    def set_params(self, concept_id: str, params: BKTParams) -> None:
        """Configures tuned parameters for a specific concept in the config table."""
        self.config_table[concept_id] = params

    @classmethod
    def update(
        cls,
        prior_mastery: float,
        correct: bool,
        hints_used: int = 0,
        params: Optional[BKTParams] = None,
    ) -> float:
        """
        Updates learner mastery P(L) using the two-step BKT formula.

        If hints_used > 0, treats the attempt as incorrect regardless of the `correct` flag.
        """
        if params is None:
            params = cls.DEFAULT_PARAMS

        # Prior mastery bounded in (0, 1) for numerical stability
        p_l = max(0.0001, min(0.9999, float(prior_mastery)))
        p_s = float(params.p_s)
        p_g = float(params.p_g)
        p_t = float(params.p_t)

        # If hints_used > 0, treat as incorrect regardless of `correct` flag
        is_effective_correct = bool(correct) and (int(hints_used) <= 0)

        # 1. Posterior given evidence
        if is_effective_correct:
            numerator = p_l * (1.0 - p_s)
            denominator = numerator + ((1.0 - p_l) * p_g)
        else:
            numerator = p_l * p_s
            denominator = numerator + ((1.0 - p_l) * (1.0 - p_g))

        posterior_p_l = numerator / max(1e-9, denominator)

        # 2. Learning transition
        p_l_new = posterior_p_l + ((1.0 - posterior_p_l) * p_t)

        return max(0.0, min(1.0, round(p_l_new, 4)))


class KnowledgeTracer:
    """
    Standard Bayesian Knowledge Tracing (BKT) engine.
    Wraps BKTUpdater and manages state updates for learner events.
    """

    def __init__(
        self,
        p_l0: float = 0.10,
        p_t: float = 0.30,
        p_s: float = 0.10,
        p_g: float = 0.20,
        updater: Optional[BKTUpdater] = None,
        config_table: Optional[Dict[str, BKTParams]] = None,
    ) -> None:
        self.p_l0 = p_l0
        self.p_t = p_t
        self.p_s = p_s
        self.p_g = p_g
        default_params = BKTParams(p_l0=p_l0, p_t=p_t, p_s=p_s, p_g=p_g)
        self.updater = updater or BKTUpdater(config_table=config_table, default_params=default_params)

    def update(
        self,
        current_mastery: float,
        is_correct: bool,
        hints_used: int = 0,
        params: Optional[BKTParams] = None,
    ) -> float:
        """
        Calculates posterior mastery P(L_t) given observation and transit to P(L_{t+1}).
        """
        if params is None:
            params = BKTParams(p_l0=self.p_l0, p_t=self.p_t, p_s=self.p_s, p_g=self.p_g)
        return self.updater.update(
            prior_mastery=current_mastery,
            correct=is_correct,
            hints_used=hints_used,
            params=params,
        )

    def process_event(
        self,
        state: LearnerState,
        event: LearningEvent
    ) -> bool:
        """
        Processes answer_submitted / quiz_submitted / answer_revealed events.
        Returns True if state was mutated.
        """
        event_type = event.event_type if isinstance(event.event_type, str) else event.event_type.value

        is_answer_event = event_type in (
            LearningEventType.ANSWER_SUBMITTED.value,
            LearningEventType.QUIZ_SUBMITTED.value,
            LearningEventType.ANSWER_REVEALED.value,
        )
        if not is_answer_event:
            return False

        concept = event.concept or event.payload.get("concept")
        if not concept:
            return False

        hints_used = event.payload.get("hints_used", 0)
        if not hints_used and event.hint_level is not None:
            hints_used = event.hint_level

        # Determine correctness
        if event_type == LearningEventType.ANSWER_REVEALED.value:
            # Tutor revealed answer because student was stuck -> treat as failed attempt
            is_correct = False
            hints_used = max(hints_used, 1)
        else:
            is_correct = bool(event.payload.get("correct", False))

        params = self.updater.get_params(concept)

        existing = state.concept_mastery.get(concept)
        if existing:
            new_score = self.updater.update(
                prior_mastery=existing.mastery,
                correct=is_correct,
                hints_used=hints_used,
                params=params,
            )
            existing.mastery = new_score
            existing.attempts += 1
            if is_correct:
                existing.correct += 1
            existing.last_updated = datetime.now(timezone.utc).isoformat()
        else:
            initial_p_l = params.p_l0
            new_score = self.updater.update(
                prior_mastery=initial_p_l,
                correct=is_correct,
                hints_used=hints_used,
                params=params,
            )
            state.concept_mastery[concept] = ConceptMastery(
                concept=concept,
                mastery=new_score,
                attempts=1,
                correct=1 if is_correct else 0,
                last_updated=datetime.now(timezone.utc).isoformat()
            )

        return True



# ---------------------------------------------------------------------------
# 2. Misconception Engine
# ---------------------------------------------------------------------------

SEED_MISCONCEPTIONS: Dict[str, List[Dict[str, Any]]] = {
    "Gradient Descent": [
        {
            "key": "gd_local_minimum_paralysis",
            "description": "Believes gradient descent gets permanently stuck on saddle points or local minima in convex problems.",
            "patterns": ["always stuck", "local minima always", "cannot escape local", "saddle point trap"]
        },
        {
            "key": "learning_rate_direction_confusion",
            "description": "Confuses learning rate with gradient direction (thinks large LR reverses search direction).",
            "patterns": ["large lr changes direction", "high learning rate moves opposite", "learning rate determines direction"]
        }
    ],
    "Backpropagation": [
        {
            "key": "backprop_update_vs_derivative",
            "description": "Confuses computing gradients with parameter updates (conflates backpropagation with optimizer).",
            "patterns": ["backprop updates weights directly", "backprop changes weights", "backprop does gradient descent"]
        },
        {
            "key": "chain_rule_layer_order",
            "description": "Believes gradients flow forward from input to output.",
            "patterns": ["gradients flow forward", "input to output gradient", "chain rule starts at input"]
        }
    ],
    "Loss Functions": [
        {
            "key": "loss_vs_accuracy_confusion",
            "description": "Equates minimizing loss with maximizing classification accuracy directly.",
            "patterns": ["loss is accuracy", "accuracy equals loss", "loss is the same as accuracy"]
        }
    ],
    "Overfitting": [
        {
            "key": "training_loss_zero_ideal",
            "description": "Believes zero training error is always the ultimate goal of learning.",
            "patterns": ["zero error is best", "goal is 0 loss", "always aim for 0 error"]
        }
    ],
    "Supervised Learning": [
        {
            "key": "unlabeled_data_supervised",
            "description": "Believes supervised models can learn without ground truth target labels.",
            "patterns": ["supervised doesn't need labels", "supervised learns from unlabelled", "no labels needed in supervised"]
        }
    ]
}


class MisconceptionEngine:
    """
    Pattern-matches incorrect responses against seed library of known misconceptions.
    Increments confidence and hit count when detected.
    """

    def __init__(
        self,
        library: Optional[Dict[str, List[Dict[str, Any]]]] = None,
        confidence_step: float = 0.15
    ) -> None:
        self.library = library or SEED_MISCONCEPTIONS
        self.confidence_step = confidence_step

    def match(
        self,
        concept: str,
        text: str
    ) -> Optional[Tuple[str, str]]:
        """
        Searches the library for matching misconceptions for the given concept.
        Returns (misconception_key, description) if matched, else None.
        """
        if not concept or not text:
            return None

        # Check exact concept and normalized concept
        patterns_to_check = []
        for lib_concept, items in self.library.items():
            if lib_concept.lower() in concept.lower() or concept.lower() in lib_concept.lower():
                patterns_to_check.extend(items)

        text_lower = text.lower()
        for item in patterns_to_check:
            for pat in item["patterns"]:
                if pat.lower() in text_lower:
                    return (item["key"], item["description"])

        return None

    def process_event(
        self,
        state: LearnerState,
        event: LearningEvent
    ) -> bool:
        """
        Inspects message / answer events for misconception patterns.
        """
        concept = event.concept or event.payload.get("concept")
        if not concept:
            return False

        # Extract text to inspect
        text = event.payload.get("response") or event.payload.get("message") or event.payload.get("answer") or ""
        if not text:
            return False

        # Only evaluate if the submission was incorrect or signaled misconception
        is_incorrect = event.payload.get("correct") is False or event.event_type in (
            LearningEventType.MISCONCEPTION_FOUND.value,
            LearningEventType.MISCONCEPTION_FOUND,
        )
        if not is_incorrect:
            return False

        matched = self.match(concept, text)
        if not matched:
            return False

        m_key, m_desc = matched
        now_iso = datetime.now(timezone.utc).isoformat()

        # Check if already present in student's state
        existing = next((m for m in state.misconceptions if m.key == m_key), None)
        if existing:
            existing.confidence = min(1.0, round(existing.confidence + self.confidence_step, 4))
            existing.hit_count += 1
            existing.last_seen_at = now_iso
        else:
            state.misconceptions.append(
                Misconception(
                    key=m_key,
                    description=m_desc,
                    concept=concept,
                    confidence=0.5,
                    hit_count=1,
                    detected_at=now_iso,
                    last_seen_at=now_iso
                )
            )

        return True


# ---------------------------------------------------------------------------
# 3. Behavioral Model
# ---------------------------------------------------------------------------

class BehavioralModel:
    """
    Tracks rolling averages for:
    - Hints per session
    - Persistence (interaction turns before asking for hints/answers)
    - Engagement score
    """

    def process_event(
        self,
        state: LearnerState,
        event: LearningEvent
    ) -> bool:
        b = state.behavior
        now_iso = datetime.now(timezone.utc).isoformat()
        b.last_active_at = now_iso

        event_type = event.event_type if isinstance(event.event_type, str) else event.event_type.value

        if event_type == LearningEventType.SESSION_STARTED.value:
            b.sessions_total += 1
            return True

        if event_type == LearningEventType.HINT_REQUESTED.value:
            b.total_hints_used += 1
            if b.sessions_total > 0:
                b.hints_per_session = round(b.total_hints_used / b.sessions_total, 2)
            else:
                b.hints_per_session = float(b.total_hints_used)
            return True

        if event_type in (LearningEventType.MESSAGE_SENT.value, LearningEventType.ANSWER_SUBMITTED.value):
            b.total_turns += 1
            if b.sessions_total > 0:
                b.avg_persistence = round(b.total_turns / b.sessions_total, 2)
            else:
                b.avg_persistence = float(b.total_turns)

            if event.payload.get("correct") is True:
                b.sessions_active = max(b.sessions_active, b.sessions_total)

            # Engagement: ratio of active to total sessions
            if b.sessions_total > 0:
                b.engagement_score = round(max(0.1, min(1.0, b.sessions_active / max(1, b.sessions_total))), 2)
            return True

        if event_type == LearningEventType.ANSWER_REVEALED.value:
            # Revealing answer lowers persistence slightly
            b.avg_persistence = max(0.0, round(b.avg_persistence * 0.9, 2))
            return True

        return False


# ---------------------------------------------------------------------------
# LearnerModelEngine
# ---------------------------------------------------------------------------

class LearnerModelEngine:
    """
    Coordinates event consumption and state persistence for learner modeling.
    Subscribes to BaseEventBus and maintains LearnerState in BaseLearnerStateStore.
    """

    def __init__(
        self,
        store: Optional[BaseLearnerStateStore] = None,
        bus: Optional[BaseEventBus] = None,
        knowledge_tracer: Optional[KnowledgeTracer] = None,
        misconception_engine: Optional[MisconceptionEngine] = None,
        behavioral_model: Optional[BehavioralModel] = None,
    ) -> None:
        self.store = store or InMemoryLearnerStateStore()
        self.bus = bus
        self.knowledge_tracer = knowledge_tracer or KnowledgeTracer()
        self.misconception_engine = misconception_engine or MisconceptionEngine()
        self.behavioral_model = behavioral_model or BehavioralModel()

        if self.bus:
            self.bus.subscribe(self.process_event)

    def process_event(self, event: LearningEvent) -> Optional[LearnerState]:
        """
        Dispatches event through all three updaters and persists the resulting LearnerState.
        """
        student_id = event.student_id
        if not student_id:
            logger.debug("Event %s has no student_id; skipping learner model update.", event.event_id)
            return None

        student_id_str = str(student_id)

        # 1. Load or initialize LearnerState
        state = self.store.load(student_id_str)
        if not state:
            state = LearnerState(student_id=student_id_str)

        # 2. Run updaters
        kt_updated = self.knowledge_tracer.process_event(state, event)
        mc_updated = self.misconception_engine.process_event(state, event)
        bm_updated = self.behavioral_model.process_event(state, event)

        # 3. Always stamp updated_at and save
        state.updated_at = datetime.now(timezone.utc).isoformat()
        self.store.save(state)

        logger.debug(
            "LearnerState updated for student %s (KT: %s, MC: %s, BM: %s)",
            student_id_str, kt_updated, mc_updated, bm_updated
        )
        return state

    def get_learner_state(self, student_id: str) -> Optional[LearnerState]:
        """Convenience query for a student's active LearnerState."""
        return self.store.load(str(student_id))
