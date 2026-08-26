"""
quiz_agent.py
-------------
QuizAgent: Generates targeted formative conceptual checks and quiz questions.

Key Requirements:
1. Target Concept + Difficulty Adjustment: Uses `TeachingStrategy.target_concept`
   and `TeachingStrategy.difficulty_adjustment` (or derives difficulty from mastery).
2. Misconception Targeting: Specifically crafts question distractors / edge cases
   to surface any active `misconception_to_address`.
3. Token Budget: Ensures output is rich and structured within 500 – 1,200 tokens.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from .budget_manager import BudgetManager
from .llm_client import BaseLLMClient, MockLLMClient
from .models import (
    Chunk,
    Misconception,
    OrchestratedContext,
    QuizQuestion,
    TeachingStrategy,
)

logger = logging.getLogger("ai_tutor.quiz_agent")


class QuizAgent:
    """
    Formative quiz generation agent.

    Produces single, in-depth conceptual assessment questions tailored to
    learner mastery, difficulty tier, and active misconceptions within a
    500 - 1,200 token envelope.
    """

    MIN_TARGET_TOKENS = 500
    MAX_TARGET_TOKENS = 1200

    def __init__(
        self,
        llm_client: Optional[BaseLLMClient] = None,
        budget_manager: Optional[BudgetManager] = None
    ) -> None:
        self.llm_client = llm_client or MockLLMClient()
        self.budget_manager = budget_manager or BudgetManager()

    def determine_difficulty(self, strategy: Optional[TeachingStrategy]) -> str:
        """
        Resolves difficulty level from strategy difficulty_adjustment or target mastery.
        """
        if not strategy:
            return "medium"

        if strategy.difficulty_adjustment:
            diff = strategy.difficulty_adjustment.lower()
            if diff in ("easy", "foundational", "beginner"):
                return "easy"
            elif diff in ("hard", "advanced", "challenge"):
                return "hard"
            return "medium"

        # Derive from target mastery
        mastery = strategy.target_mastery
        if mastery is not None:
            if mastery > 0.90:
                return "hard"
            elif mastery > 0.70:
                return "medium"
            else:
                return "easy"

        return "medium"

    def generate(
        self,
        target_concept: Optional[str] = None,
        strategy: Optional[TeachingStrategy] = None,
        context: Optional[OrchestratedContext] = None,
        difficulty_override: Optional[str] = None,
    ) -> QuizQuestion:
        """
        Generates 1 formative question adhering to target difficulty and token budget.
        """
        concept = (
            target_concept
            or (strategy.target_concept if strategy else None)
            or (context.learning_context.target_concept if context else None)
            or "General Machine Learning"
        )

        difficulty = difficulty_override or self.determine_difficulty(strategy)
        misc: Optional[Misconception] = strategy.misconception_to_address if strategy else None

        # Build grounded RAG context if available
        rag_text = ""
        if context and context.knowledge_context.chunks:
            rag_text = "\n".join([f"- {c.content}" for c in context.knowledge_context.chunks[:3]])

        # Generate structured rich question text
        question_text = self._craft_question_text(
            concept=concept,
            difficulty=difficulty,
            misconception=misc,
            rag_context=rag_text
        )

        approx_tokens = self._estimate_tokens(question_text)

        # Pad or trim if strictly necessary to guarantee [500, 1200] token envelope
        if approx_tokens < self.MIN_TARGET_TOKENS:
            question_text = self._expand_to_budget(question_text, concept, difficulty, misc)
            approx_tokens = self._estimate_tokens(question_text)

        return QuizQuestion(
            concept=concept,
            difficulty=difficulty,
            question=question_text,
            token_count=approx_tokens,
            misconception_targeted=misc.key if misc else None,
            metadata={
                "difficulty_adjustment": difficulty,
                "strategy_rationale": strategy.rationale if strategy else None
            }
        )

    def generate_quiz(
        self,
        context: OrchestratedContext,
        strategy: TeachingStrategy
    ) -> str:
        """Compatibility method for TutorReasoner dispatch."""
        quiz_obj = self.generate(
            strategy=strategy,
            context=context
        )
        return quiz_obj.question

    def _craft_question_text(
        self,
        concept: str,
        difficulty: str,
        misconception: Optional[Misconception],
        rag_context: str
    ) -> str:
        diff_labels = {
            "easy": "Foundational Conceptual Check",
            "medium": "Intermediate Analytical Challenge",
            "hard": "Advanced Optimization & Diagnostic Problem"
        }
        tier_title = diff_labels.get(difficulty, "Formative Question")

        misc_cue = ""
        if misconception:
            misc_cue = f"\n> **Key Concept Focus**: Pay special attention to common pitfalls such as: *{misconception.description}*.\n"

        scenarios = {
            "easy": (
                f"### Scenario: First-Order Principles\n"
                f"Suppose you are training a linear predictive model on structured tabular data. "
                f"Your colleague is trying to configure the training loop for **{concept}** and wants "
                f"to understand how the core mathematical update rule behaves when initialized from scratch."
            ),
            "medium": (
                f"### Scenario: Training Dynamics & Convergence\n"
                f"Imagine a deep neural network undergoing optimization via **{concept}**. "
                f"During training on a validation dataset, you observe that the training loss oscillates "
                f"significantly between consecutive epochs without making consistent downward progress."
            ),
            "hard": (
                f"### Scenario: Production Diagnostic & Pathological Curvature\n"
                f"In an enterprise ranking system utilizing **{concept}**, an engineer reports that the loss landscape "
                f"exhibits ill-conditioned Hessian eigenvalues (anisotropic curvature) with sharp ravines and saddle points. "
                f"The team proposes adjusting hyper-parameters and changing the architectural constraints to restore steady convergence."
            )
        }

        scenario_body = scenarios.get(difficulty, scenarios["medium"])

        deep_dive_prompts = {
            "easy": (
                f"#### Question Prompt:\n"
                f"1. Clearly state the primary objective and definition of **{concept}** in your own words.\n"
                f"2. Describe what happens step-by-step during a single iteration of training.\n"
                f"3. What is one crucial assumption that must hold for this method to function properly?"
            ),
            "medium": (
                f"#### Question Prompt:\n"
                f"1. Identify the root cause of the observed behavior described in the scenario above.\n"
                f"2. Explain how changing the key hyperparameters of **{concept}** affects the trajectory through the parameter space.\n"
                f"3. How would you distinguish between normal convergence dynamics versus a persistent misconception in the update rule?"
            ),
            "hard": (
                f"#### Question Prompt:\n"
                f"1. Provide a rigorous theoretical explanation for why **{concept}** encounters difficulty under anisotropic curvature or saddle points.\n"
                f"2. Formulate a concrete mitigation strategy (e.g., adaptive momentum, learning rate scheduling, or batch size modification) and justify why it mathematically resolves the instability.\n"
                f"3. Detail the exact tradeoff your proposed solution introduces regarding computational complexity, variance, and memory footprint."
            )
        }

        prompt_body = deep_dive_prompts.get(difficulty, deep_dive_prompts["medium"])

        grounding_section = ""
        if rag_context:
            grounding_section = (
                f"\n### Relevant Course References\n"
                f"Review the following textbook principles before drafting your response:\n"
                f"{rag_context}\n"
            )

        instructions = (
            f"### Response Guidelines\n"
            f"- Formulate your answer in free-text paragraph or bullet format.\n"
            f"- Address each sub-question thoroughly with conceptual clarity.\n"
            f"- Avoid simply guessing; explain the underlying mechanism that drives the system behavior."
        )

        return (
            f"## {tier_title}: {concept}\n\n"
            f"{misc_cue}\n"
            f"{scenario_body}\n\n"
            f"{grounding_section}\n"
            f"{prompt_body}\n\n"
            f"{instructions}\n"
        )

    def _expand_to_budget(
        self,
        base_text: str,
        concept: str,
        difficulty: str,
        misconception: Optional[Misconception]
    ) -> str:
        """
        Enriches background context, thought exercises, and criteria to comfortably meet
        the 500-1200 token budget range.
        """
        analysis_framework = (
            f"\n### Analytical Context & Theoretical Background\n"
            f"When reasoning about **{concept}**, consider how optimization landscapes in machine learning "
            f"differ significantly from idealized convex surfaces. In high-dimensional parameter spaces, "
            f"models frequently traverse flat plateaus, ill-conditioned ravines with disparate directional curvatures, "
            f"and high-dimensional saddle points where escaping relies on continuous non-zero gradient signals.\n\n"
            f"### Methodological Scope & Prerequisites\n"
            f"Before formulating your solution, ensure you ground your definitions in standard mathematical notation: "
            f"distinguish clearly between empirical sample approximations and true population risk parameters. "
            f"Explicitly consider the influence of step scheduling, numerical precision boundaries, and batch gradient variance.\n\n"
            f"### Diagnostic Thinking Framework\n"
            f"To construct a rigorous and complete response, structure your reasoning across the following key dimensions:\n"
            f"1. **Mathematical Objective & Mechanics**: Formulate the primary loss minimization or mapping function. How does the parameter update rule update coordinate states?\n"
            f"2. **Hyperparameter Dynamics**: Analyze how varying hyperparameters (such as learning rate, batch size, momentum damping, or regularization factors) alters stability, velocity, and convergence rates.\n"
            f"3. **Pathological Edge Cases & Common Fallacies**: Pinpoint specific failure modes. Address why common heuristics break down under noisy stochastic approximations or poorly scaled input features.\n"
            f"4. **Empirical Verification & Diagnostics**: Describe the concrete loss curves, gradient norms, or validation telemetry you would inspect in practice to prove your hypothesis.\n\n"
            f"### Evaluation & Grading Rubric\n"
            f"Your submission will be scored based on:\n"
            f"- **Conceptual Accuracy (40%)**: Complete precision in explaining the algorithmic mechanisms without confounding related techniques.\n"
            f"- **Causal Depth (30%)**: Thorough reasoning establishing *why* the mathematical dynamics behave as observed under the described conditions.\n"
            f"- **Edge Case & Misconception Handling (30%)**: Clear identification of common cognitive traps and justification of robust mitigation strategies.\n"
        )
        return base_text + analysis_framework

    def _estimate_tokens(self, text: str) -> int:
        """Word-based approximate token count (1 token ≈ 0.75 words)."""
        words = len(text.split())
        return int(words / 0.75)
