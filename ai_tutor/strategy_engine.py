"""
strategy_engine.py
------------------
StrategyEngine implementing the pedagogical Plan step:
1. Taxonomy-driven strategy selection (strategy_category and strategy_type).
2. Effectiveness-based tie breaking via `learner_strategy_effectiveness` records.
3. Multi-Armed Bandit style preference: chooses strategy_type with highest
   (times_led_to_mastery / times_used) ratio for (user_id, concept_domain),
   falling back to a fixed default taxonomy order when no history exists.
4. Assess step feedback loop: increments `times_used` on every attempt,
   and `times_led_to_mastery` when assessment reaches 'mastered'.
"""

from __future__ import annotations

import logging
import threading
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

from .models import (
    CurriculumPosition,
    LearnerState,
    Misconception,
    RootCauseDiagnosis,
    StrategyAction,
    StrategyEffectivenessRecord,
    TeachingStrategy,
)

if TYPE_CHECKING:
    from .concept_graph import ConceptGraph

logger = logging.getLogger("ai_tutor.strategy_engine")


# =============================================================================
# STRATEGY TAXONOMY
# =============================================================================
# Standard pedagogical strategy taxonomy mapping categories to ordered subtypes:
STRATEGY_TAXONOMY: Dict[str, List[str]] = {
    "explanation": [
        "analogy",            # Real-world intuitive metaphor
        "visual",             # Diagrammatic mental model
        "worked_example",     # Concrete step-by-step example
        "first_principles",   # Formal axiomatic definition/derivation
    ],
    "scaffolding": [
        "leading_question",   # Socratic diagnostic prompt
        "subgoal_breakdown",  # Problem decomposition
        "formula_nudge",      # Key theorem / rule reminder
        "near_solution",      # High-specificity next-step hint
    ],
    "assessment": [
        "conceptual_mcq",     # Conceptual multiple-choice check
        "free_response",      # In-depth open-ended diagnostic probe
        "counterexample_probe", # Boundary condition verification
        "calculation",        # Direct mathematical computation
    ],
    "challenge": [
        "transfer_problem",   # Novel application outside training domain
        "edge_case_debugging",# Error identification in adversarial code/math
        "synthesis",          # Multi-concept cross-domain integration
    ],
    "remediation": [
        "prerequisite_review",     # Step back to address unmastered ancestor gap
        "misconception_refutation",# Cognitive conflict addressing active misconception
    ],
}


# =============================================================================
# PERSISTENCE LAYER: Base & In-Memory / Postgres Stores
# =============================================================================

class BaseStrategyEffectivenessStore(ABC):
    """Abstract persistence interface for learner strategy effectiveness."""

    @abstractmethod
    def get_effectiveness(
        self,
        user_id: int,
        strategy_category: str,
        strategy_type: str,
        concept_domain: str
    ) -> Optional[StrategyEffectivenessRecord]:
        """Loads a specific strategy effectiveness record."""
        pass

    @abstractmethod
    def get_all_for_category(
        self,
        user_id: int,
        strategy_category: str,
        concept_domain: str
    ) -> Dict[str, StrategyEffectivenessRecord]:
        """Loads all strategy effectiveness records for a given category & domain."""
        pass

    @abstractmethod
    def record_outcome(
        self,
        user_id: int,
        strategy_category: str,
        strategy_type: str,
        concept_domain: str,
        led_to_mastery: bool
    ) -> StrategyEffectivenessRecord:
        """Increments times_used, and conditionally increments times_led_to_mastery."""
        pass


class InMemoryStrategyEffectivenessStore(BaseStrategyEffectivenessStore):
    """Thread-safe in-memory effectiveness store for local execution and unit tests."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # Key: (user_id, strategy_category, strategy_type, concept_domain)
        self._store: Dict[Tuple[int, str, str, str], StrategyEffectivenessRecord] = {}

    def get_effectiveness(
        self,
        user_id: int,
        strategy_category: str,
        strategy_type: str,
        concept_domain: str
    ) -> Optional[StrategyEffectivenessRecord]:
        key = (int(user_id), strategy_category.lower(), strategy_type.lower(), concept_domain.lower())
        with self._lock:
            record = self._store.get(key)
            return record.model_copy(deep=True) if record else None

    def get_all_for_category(
        self,
        user_id: int,
        strategy_category: str,
        concept_domain: str
    ) -> Dict[str, StrategyEffectivenessRecord]:
        uid = int(user_id)
        cat = strategy_category.lower()
        dom = concept_domain.lower()
        results: Dict[str, StrategyEffectivenessRecord] = {}
        with self._lock:
            for (r_uid, r_cat, r_type, r_dom), record in self._store.items():
                if r_uid == uid and r_cat == cat and r_dom == dom:
                    results[r_type] = record.model_copy(deep=True)
        return results

    def record_outcome(
        self,
        user_id: int,
        strategy_category: str,
        strategy_type: str,
        concept_domain: str,
        led_to_mastery: bool
    ) -> StrategyEffectivenessRecord:
        uid = int(user_id)
        cat = strategy_category.lower()
        stype = strategy_type.lower()
        dom = concept_domain.lower()
        key = (uid, cat, stype, dom)

        with self._lock:
            if key in self._store:
                rec = self._store[key]
                rec.times_used += 1
                if led_to_mastery:
                    rec.times_led_to_mastery += 1
                rec.updated_at = datetime.now(timezone.utc).isoformat()
            else:
                rec = StrategyEffectivenessRecord(
                    user_id=uid,
                    strategy_category=cat,
                    strategy_type=stype,
                    concept_domain=dom,
                    times_used=1,
                    times_led_to_mastery=1 if led_to_mastery else 0,
                    updated_at=datetime.now(timezone.utc).isoformat()
                )
                self._store[key] = rec

            return rec.model_copy(deep=True)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()


class PostgresStrategyEffectivenessStore(BaseStrategyEffectivenessStore):
    """
    PostgreSQL-backed store matching the schema:
    CREATE TABLE learner_strategy_effectiveness (
        id SERIAL PRIMARY KEY,
        user_id BIGINT NOT NULL,
        strategy_category VARCHAR(32),
        strategy_type VARCHAR(32),
        concept_domain VARCHAR(64),
        times_used INT DEFAULT 0,
        times_led_to_mastery INT DEFAULT 0,
        updated_at TIMESTAMPTZ DEFAULT now(),
        UNIQUE (user_id, strategy_category, strategy_type, concept_domain)
    );
    """

    UPSERT_SQL = """
    INSERT INTO learner_strategy_effectiveness (
        user_id, strategy_category, strategy_type, concept_domain,
        times_used, times_led_to_mastery, updated_at
    )
    VALUES (%(user_id)s, %(strategy_category)s, %(strategy_type)s, %(concept_domain)s, 1, %(led_to_mastery_int)s, NOW())
    ON CONFLICT (user_id, strategy_category, strategy_type, concept_domain)
    DO UPDATE SET
        times_used = learner_strategy_effectiveness.times_used + 1,
        times_led_to_mastery = learner_strategy_effectiveness.times_led_to_mastery + EXCLUDED.times_led_to_mastery,
        updated_at = NOW()
    RETURNING user_id, strategy_category, strategy_type, concept_domain, times_used, times_led_to_mastery, updated_at;
    """

    SELECT_CATEGORY_SQL = """
    SELECT user_id, strategy_category, strategy_type, concept_domain, times_used, times_led_to_mastery, updated_at
    FROM learner_strategy_effectiveness
    WHERE user_id = %s AND strategy_category = %s AND concept_domain = %s;
    """

    def __init__(self, postgres_dsn: str) -> None:
        self.postgres_dsn = postgres_dsn

    def get_effectiveness(
        self,
        user_id: int,
        strategy_category: str,
        strategy_type: str,
        concept_domain: str
    ) -> Optional[StrategyEffectivenessRecord]:
        records = self.get_all_for_category(user_id, strategy_category, concept_domain)
        return records.get(strategy_type.lower())

    def get_all_for_category(
        self,
        user_id: int,
        strategy_category: str,
        concept_domain: str
    ) -> Dict[str, StrategyEffectivenessRecord]:
        try:
            import psycopg2
            with psycopg2.connect(self.postgres_dsn) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        self.SELECT_CATEGORY_SQL,
                        (int(user_id), strategy_category.lower(), concept_domain.lower())
                    )
                    rows = cur.fetchall()
            results = {}
            for r in rows:
                rec = StrategyEffectivenessRecord(
                    user_id=r[0],
                    strategy_category=r[1],
                    strategy_type=r[2],
                    concept_domain=r[3],
                    times_used=r[4],
                    times_led_to_mastery=r[5],
                    updated_at=str(r[6])
                )
                results[rec.strategy_type] = rec
            return results
        except Exception as e:
            logger.error("[PostgresStrategyEffectivenessStore] Query error: %s", e)
            return {}

    def record_outcome(
        self,
        user_id: int,
        strategy_category: str,
        strategy_type: str,
        concept_domain: str,
        led_to_mastery: bool
    ) -> StrategyEffectivenessRecord:
        try:
            import psycopg2
            params = {
                "user_id": int(user_id),
                "strategy_category": strategy_category.lower(),
                "strategy_type": strategy_type.lower(),
                "concept_domain": concept_domain.lower(),
                "led_to_mastery_int": 1 if led_to_mastery else 0,
            }
            with psycopg2.connect(self.postgres_dsn) as conn:
                with conn.cursor() as cur:
                    cur.execute(self.UPSERT_SQL, params)
                    r = cur.fetchone()
                conn.commit()
            if r:
                return StrategyEffectivenessRecord(
                    user_id=r[0],
                    strategy_category=r[1],
                    strategy_type=r[2],
                    concept_domain=r[3],
                    times_used=r[4],
                    times_led_to_mastery=r[5],
                    updated_at=str(r[6])
                )
        except Exception as e:
            logger.error("[PostgresStrategyEffectivenessStore] Save error: %s", e)
            raise

        return StrategyEffectivenessRecord(
            user_id=int(user_id),
            strategy_category=strategy_category.lower(),
            strategy_type=strategy_type.lower(),
            concept_domain=concept_domain.lower(),
            times_used=1,
            times_led_to_mastery=1 if led_to_mastery else 0
        )


# =============================================================================
# STRATEGY ENGINE
# =============================================================================

class StrategyEngine:
    """
    Orchestrates the Plan step:
    Decides strategy_category and strategy_type per taxonomy, resolving ties
    using learner_strategy_effectiveness empirical performance ratios.
    """

    def __init__(
        self,
        store: Optional[BaseStrategyEffectivenessStore] = None,
        taxonomy: Optional[Dict[str, List[str]]] = None,
        default_hint_budget: int = 3,
        quiz_mastery_threshold: float = 0.8,
        challenge_mastery_threshold: float = 0.9,
        explain_failure_threshold: int = 2,
    ) -> None:
        self.store = store or InMemoryStrategyEffectivenessStore()
        self.taxonomy = taxonomy or STRATEGY_TAXONOMY
        self.default_hint_budget = default_hint_budget
        self.quiz_mastery_threshold = quiz_mastery_threshold
        self.challenge_mastery_threshold = challenge_mastery_threshold
        self.explain_failure_threshold = explain_failure_threshold

    @staticmethod
    def _parse_user_id(user_id: Optional[Union[int, str]]) -> int:
        """Safely parses user_id to BIGINT integer; defaults to 1 for non-numeric/anonymous."""
        if user_id is None:
            return 1
        if isinstance(user_id, int):
            return user_id
        # If string contains digits only
        if isinstance(user_id, str):
            digits = "".join([c for c in user_id if c.isdigit()])
            if digits:
                return int(digits)
            return abs(hash(user_id)) % (10**10)
        return 1

    def resolve_strategy_type_tie(
        self,
        user_id: int,
        strategy_category: str,
        concept_domain: str,
        candidates: Optional[List[str]] = None,
    ) -> str:
        """
        Picks the best strategy_type for this user + category + domain.
        Prefers the highest (times_led_to_mastery / times_used) ratio.
        Falls back to fixed default order in taxonomy if no history / equal.
        """
        valid_candidates = candidates or self.taxonomy.get(strategy_category, ["default"])
        if len(valid_candidates) <= 1:
            return valid_candidates[0]

        # Load effectiveness history for this user + category + domain
        history_map = self.store.get_all_for_category(
            user_id=user_id,
            strategy_category=strategy_category,
            concept_domain=concept_domain,
        )

        best_type = valid_candidates[0]
        best_ratio = -1.0
        best_times_used = -1

        for c_type in valid_candidates:
            rec = history_map.get(c_type.lower())
            if rec and rec.times_used > 0:
                ratio = rec.effectiveness_ratio
                # Prefer strictly higher ratio, or higher sample volume if ratio tied
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_times_used = rec.times_used
                    best_type = c_type
                elif ratio == best_ratio and rec.times_used > best_times_used and ratio > 0.0:
                    best_times_used = rec.times_used
                    best_type = c_type

        return best_type

    def plan(
        self,
        learner_state: Optional[LearnerState] = None,
        concept_graph: Optional["ConceptGraph"] = None,
        target_concept: Optional[str] = None,
        concept_domain: Optional[str] = None,
        user_id: Optional[Union[int, str]] = None,
        consecutive_failures: int = 0,
        last_answer_correct: Optional[bool] = None,
        hint_budget_remaining: Optional[int] = None,
        course_id: Optional[int] = None,
        lecture_id: Optional[int] = None,
        strategy_category_override: Optional[str] = None,
        strategy_type_override: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> TeachingStrategy:
        """
        Executes the Plan step:
        Given LearnerState + ConceptGraph + performance metrics, decides
        strategy_category and strategy_type.
        """
        parsed_user_id = self._parse_user_id(
            user_id or (learner_state.student_id if learner_state else None)
        )
        domain = concept_domain or "general"
        budget = hint_budget_remaining if hint_budget_remaining is not None else self.default_hint_budget

        # 1. Deterministic ConceptGraph computations
        curriculum_position: Optional[CurriculumPosition] = None
        if concept_graph and learner_state:
            try:
                curriculum_position = concept_graph.compute_curriculum_position(
                    learner_state=learner_state,
                    current_concept=target_concept
                )
            except Exception as e:
                logger.warning("[StrategyEngine] CurriculumPosition failed: %s", e)

        root_cause_diagnosis: Optional[RootCauseDiagnosis] = None
        if concept_graph and learner_state and target_concept and consecutive_failures >= 1:
            try:
                diagnosis = concept_graph.diagnose_root_cause(
                    struggling_concept=target_concept,
                    learner_state=learner_state
                )
                if diagnosis.likely_root_gap is not None:
                    root_cause_diagnosis = diagnosis
            except Exception as e:
                logger.warning("[StrategyEngine] RootCauseDiagnosis failed: %s", e)

        # 2. Extract current mastery & active misconceptions
        current_mastery: Optional[float] = None
        if learner_state and target_concept:
            cm = learner_state.concept_mastery.get(target_concept)
            if cm:
                current_mastery = cm.mastery

        highest_misconception: Optional[Misconception] = None
        if learner_state and learner_state.misconceptions:
            matched = [
                m for m in learner_state.misconceptions
                if target_concept and (m.concept.lower() in target_concept.lower() or target_concept.lower() in m.concept.lower())
            ]
            candidates = matched if matched else learner_state.misconceptions
            highest_misconception = max(candidates, key=lambda m: (m.confidence, m.hit_count))

        # 3. Determine Strategy Category & Recommendation Action
        if strategy_category_override:
            category = strategy_category_override.lower()
            rec_action = StrategyAction.GUIDE
        elif root_cause_diagnosis and root_cause_diagnosis.likely_root_gap and root_cause_diagnosis.confidence >= 0.3:
            category = "remediation"
            rec_action = StrategyAction.EXPLAIN
        elif highest_misconception and consecutive_failures >= 1:
            category = "remediation"
            rec_action = StrategyAction.EXPLAIN
        elif consecutive_failures >= self.explain_failure_threshold:
            category = "explanation"
            rec_action = StrategyAction.EXPLAIN

        elif last_answer_correct is False and budget > 0:
            category = "scaffolding"
            rec_action = StrategyAction.HINT
        elif current_mastery is not None and current_mastery > self.challenge_mastery_threshold:
            category = "challenge"
            rec_action = StrategyAction.CHALLENGE
        elif current_mastery is not None and current_mastery > self.quiz_mastery_threshold:
            category = "assessment"
            rec_action = StrategyAction.QUIZ
        else:
            category = "scaffolding"
            rec_action = StrategyAction.GUIDE

        # 4. Determine Strategy Type via Candidate Evaluation & Tie-Breaking
        if strategy_type_override:
            chosen_type = strategy_type_override.lower()
        elif category == "remediation":
            if root_cause_diagnosis and root_cause_diagnosis.likely_root_gap:
                chosen_type = "prerequisite_review"
            elif highest_misconception:
                chosen_type = "misconception_refutation"
            else:
                chosen_type = self.resolve_strategy_type_tie(parsed_user_id, category, domain)
        else:
            # Multi-type candidate selection with tie-breaking
            candidates = self.taxonomy.get(category, ["default"])
            chosen_type = self.resolve_strategy_type_tie(
                user_id=parsed_user_id,
                strategy_category=category,
                concept_domain=domain,
                candidates=candidates
            )

        # 5. Build Rationale String
        rationale = (
            f"Plan selected category '{category}' and type '{chosen_type}' "
            f"for concept '{target_concept or 'active topic'}' in domain '{domain}' "
            f"(mastery={current_mastery or 0.0:.2f}, failures={consecutive_failures}, budget={budget})."
        )

        return TeachingStrategy(
            recommendation=rec_action,
            target_concept=target_concept,
            target_mastery=current_mastery,
            misconception_to_address=highest_misconception,
            hint_budget_remaining=budget,
            consecutive_failures=consecutive_failures,
            rationale=rationale,
            course_id=course_id,
            lecture_id=lecture_id,
            strategy_category=category,
            strategy_type=chosen_type,
            concept_domain=domain,
            curriculum_position=curriculum_position,
            root_cause_diagnosis=root_cause_diagnosis,
            metadata=metadata or {}
        )

    def record_assess_result(
        self,
        user_id: Union[int, str],
        strategy_category: str,
        strategy_type: str,
        concept_domain: str = "general",
        assess_result: Union[str, bool] = False,
    ) -> StrategyEffectivenessRecord:
        """
        Assess Step Hook:
        Increments times_used for the tried strategy, and increments
        times_led_to_mastery if the assessment result was 'mastered' or True.
        """
        parsed_user_id = self._parse_user_id(user_id)
        is_mastered = (
            assess_result is True
            or (isinstance(assess_result, str) and assess_result.strip().lower() in ("mastered", "correct", "passed", "true"))
        )

        record = self.store.record_outcome(
            user_id=parsed_user_id,
            strategy_category=strategy_category,
            strategy_type=strategy_type,
            concept_domain=concept_domain,
            led_to_mastery=is_mastered
        )

        logger.info(
            "[StrategyEngine] Assess update: user=%d cat=%s type=%s domain=%s -> used=%d mastered=%d (ratio=%.2f)",
            parsed_user_id, strategy_category, strategy_type, concept_domain,
            record.times_used, record.times_led_to_mastery, record.effectiveness_ratio
        )
        return record
