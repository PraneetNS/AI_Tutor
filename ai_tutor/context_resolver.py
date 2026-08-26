"""
context_resolver.py
--------------------
ContextResolver: Reads LearnerState + current course/lesson context and
produces a TeachingStrategy object.

Rules:
1. recommend 'explain'   -> if 2+ wrong attempts on target_concept (Rule 5 compliance).
2. recommend 'hint'      -> if hint_budget_remaining > 0 and last answer was wrong.
3. recommend 'challenge' -> if mastery on target_concept > 0.9 (push to next concept).
4. recommend 'quiz'      -> if mastery on target_concept > 0.8 (check for real understanding).
5. Default: recommend 'guide' (Socratic inquiry / diagnostics).

Always surfaces the highest-confidence unresolved misconception as `misconception_to_address`.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from .models import (
    LearnerState,
    Misconception,
    StrategyAction,
    TeachingStrategy,
)

logger = logging.getLogger("ai_tutor.context_resolver")


class ContextResolver:
    """
    Evaluates student state, current learning goals, and recent performance
    to recommend the optimal next pedagogical strategy.
    """

    def __init__(
        self,
        default_hint_budget: int = 3,
        quiz_mastery_threshold: float = 0.8,
        challenge_mastery_threshold: float = 0.9,
        explain_failure_threshold: int = 2,
    ) -> None:
        self.default_hint_budget = default_hint_budget
        self.quiz_mastery_threshold = quiz_mastery_threshold
        self.challenge_mastery_threshold = challenge_mastery_threshold
        self.explain_failure_threshold = explain_failure_threshold

    def resolve(
        self,
        learner_state: Optional[LearnerState] = None,
        target_concept: Optional[str] = None,
        course_id: Optional[int] = None,
        lecture_id: Optional[int] = None,
        last_answer_correct: Optional[bool] = None,
        consecutive_failures: int = 0,
        hint_budget_remaining: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> TeachingStrategy:
        """
        Produce a TeachingStrategy recommendation based on learner state and context.

        Parameters
        ----------
        learner_state : Optional[LearnerState]
            Current persistent learner state for the student.
        target_concept : Optional[str]
            Concept currently being explored or evaluated.
        course_id : Optional[int]
            LMS course scope.
        lecture_id : Optional[int]
            LMS lecture / lesson scope.
        last_answer_correct : Optional[bool]
            Whether the learner's immediate prior answer attempt was correct.
        consecutive_failures : int
            Count of consecutive incorrect submissions on target_concept.
        hint_budget_remaining : Optional[int]
            Remaining hint budget allowances. Defaults to self.default_hint_budget.
        metadata : Optional[Dict[str, Any]]
            Additional context parameters.

        Returns
        -------
        TeachingStrategy
        """
        extra_meta = dict(metadata or {})

        # 1. Resolve Hint Budget
        if hint_budget_remaining is not None:
            budget = max(0, hint_budget_remaining)
        else:
            budget = self.default_hint_budget

        # 2. Extract current mastery for target concept
        current_mastery: Optional[float] = None
        if learner_state and target_concept:
            cm = learner_state.concept_mastery.get(target_concept)
            if cm:
                current_mastery = cm.mastery

        # 3. Find highest-confidence unresolved misconception
        highest_misconception = self._find_highest_confidence_misconception(
            learner_state=learner_state,
            target_concept=target_concept
        )

        # 4. Evaluate decision tree according to pedagogical rules
        recommendation: StrategyAction
        rationale: str

        # Rule 1: 2+ wrong attempts -> Explain
        if consecutive_failures >= self.explain_failure_threshold:
            recommendation = StrategyAction.EXPLAIN
            rationale = (
                f"Student has {consecutive_failures} consecutive incorrect attempts on "
                f"'{target_concept or 'the target concept'}'. Switching to direct explanation with full reasoning."
            )

        # Rule 2: Last answer was wrong & hint budget available -> Hint
        elif last_answer_correct is False and budget > 0:
            recommendation = StrategyAction.HINT
            rationale = (
                f"Student submitted an incorrect answer on '{target_concept or 'the concept'}' "
                f"and has {budget} hint(s) remaining. Providing progressive scaffolding hint."
            )

        # Rule 3: Mastery > 0.9 -> Challenge (push to next concept)
        elif current_mastery is not None and current_mastery > self.challenge_mastery_threshold:
            recommendation = StrategyAction.CHALLENGE
            rationale = (
                f"High mastery on '{target_concept}' ({current_mastery:.2f} > {self.challenge_mastery_threshold}). "
                f"Pushing learner to advanced application or next syllabus concept."
            )

        # Rule 4: Mastery > 0.8 -> Quiz (verify understanding)
        elif current_mastery is not None and current_mastery > self.quiz_mastery_threshold:
            recommendation = StrategyAction.QUIZ
            rationale = (
                f"Mastery on '{target_concept}' reached {current_mastery:.2f} (> {self.quiz_mastery_threshold}). "
                f"Administering formative quiz check to verify robust conceptual understanding."
            )

        # Default: Guide with Socratic inquiry
        else:
            recommendation = StrategyAction.GUIDE
            rationale = (
                f"Guiding student with diagnostic Socratic questioning on "
                f"'{target_concept or 'active topic'}'."
            )

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
            metadata=extra_meta,
        )

    def _find_highest_confidence_misconception(
        self,
        learner_state: Optional[LearnerState],
        target_concept: Optional[str]
    ) -> Optional[Misconception]:
        """
        Surfaces the highest-confidence misconception, prioritizing target_concept if present.
        """
        if not learner_state or not learner_state.misconceptions:
            return None

        misconceptions = learner_state.misconceptions

        # If target_concept is provided, check if there are misconceptions specifically for it
        if target_concept:
            concept_specific = [
                m for m in misconceptions
                if m.concept.lower() == target_concept.lower()
            ]
            if concept_specific:
                # Return highest confidence
                return max(concept_specific, key=lambda m: (m.confidence, m.hit_count))

        # Fallback to highest confidence across all active misconceptions
        return max(misconceptions, key=lambda m: (m.confidence, m.hit_count))
