"""
tutor_reasoner.py
-----------------
TutorReasoner: The core 'decide -> teach -> assess -> adapt' loop for the AI Tutor.

Loop Stages:
1. DECIDE: Inspects the merged OrchestratedContext and TeachingStrategy to pick
   the responsible sub-agent (TutorAgent, QuizAgent, or AssessmentAgent).
2. TEACH / ASSESS: Executes the selected sub-agent to generate the tailored response.
3. ADAPT: Updates the dynamic pedagogy state (hint_level, stuck, mode, topic) for the next turn.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from .budget_manager import BudgetManager
from .llm_client import BaseLLMClient, MockLLMClient
from .models import (
    ChatMessage,
    OrchestratedContext,
    PedagogyMode,
    PedagogyState,
    ReasonerResult,
    Role,
    SourceCitation,
    StrategyAction,
    TeachingStrategy,
)
from .tutor_core import DEFAULT_SYSTEM_PROMPT

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
        # 1. Extract prompt sections from orchestrated context
        sections = context.to_prompt_sections()
        sections["system"] = DEFAULT_SYSTEM_PROMPT

        # Define default section budgets (budget_min, budget_max)
        budgets = {
            "system": (150, 400),
            "learner_state": (30, 150),
            "rag_knowledge": (100, 500),
            "teaching_strategy": (20, 100),
            "conversation_history": (50, 400),
        }

        # 2. Assemble budget-conscious prompt
        assembled_prompt, _ = self.budget_manager.assemble(sections, budgets)

        # 3. Call LLM
        messages = list(context.session_context.recent_messages) + [
            ChatMessage(role=Role.USER, content=context.student_message)
        ]

        output = self.llm_client.generate(
            system_prompt=assembled_prompt,
            messages=messages,
            current_state=context.session_context.pedagogy_state
        )

        return output.answer


from .quiz_agent import QuizAgent
from .assessment_agent import AssessmentAgent


# ---------------------------------------------------------------------------
# 2. TutorReasoner: The 'decide -> teach -> assess -> adapt' Orchestrator
# ---------------------------------------------------------------------------

class TutorReasoner:
    """
    Coordinates the full cycle:
    1. DECIDE: Selects the appropriate sub-agent based on context, intent, and teaching strategy.
    2. TEACH / ASSESS: Delegates turn generation to the selected sub-agent.
    3. ADAPT: Updates pedagogy state (hint_level, stuck, mode, topic) for the next interaction turn.
    """

    def __init__(
        self,
        tutor_agent: Optional[BaseTutorSubAgent] = None,
        quiz_agent: Optional[BaseQuizSubAgent] = None,
        assessment_agent: Optional[BaseAssessmentSubAgent] = None,
    ) -> None:
        self.tutor_agent = tutor_agent or TutorAgent()
        self.quiz_agent = quiz_agent or QuizAgent()
        self.assessment_agent = assessment_agent or AssessmentAgent()

    # ------------------------------------------------------------------
    # Step 1: DECIDE
    # ------------------------------------------------------------------

    def decide(
        self,
        context: OrchestratedContext,
        strategy: TeachingStrategy
    ) -> str:
        """
        Determines which sub-agent should handle the current interaction.

        Decision Rules:
        - AssessmentAgent: If prior turn asked a quiz/question and student is responding with an answer attempt.
        - QuizAgent: If strategy recommendation is 'quiz' or student asked for a test/quiz.
        - TutorAgent: Default for teaching, hinting, explaining, and Socratic guiding.
        """
        student_msg = context.student_message.lower()

        # 1. Did student explicitly ask for a quiz or does strategy mandate a quiz?
        if any(w in student_msg for w in ["quiz me", "test me", "give me a problem", "practice question"]):
            return "QuizAgent"

        if strategy.recommendation == StrategyAction.QUIZ or strategy.recommendation == "quiz":
            return "QuizAgent"

        # 2. Is the student submitting an answer to an active question?
        prior_messages = context.session_context.recent_messages
        if prior_messages:
            last_assistant_msg = next((m for m in reversed(prior_messages) if m.role == Role.ASSISTANT), None)
            if last_assistant_msg:
                last_content = last_assistant_msg.content.lower()
                is_prior_question = any(q in last_content for q in ["?", "try", "what happens", "in your own words", "check:"])
                if is_prior_question and not any(h in student_msg for h in ["hint", "stuck", "don't know", "tell me", "explain"]):
                    return "AssessmentAgent"

        # 3. Default to TutorAgent for teaching / hinting / explanation
        return "TutorAgent"

    # ------------------------------------------------------------------
    # Step 2: TEACH / ASSESS
    # ------------------------------------------------------------------

    def teach_or_assess(
        self,
        selected_agent: str,
        context: OrchestratedContext,
        strategy: TeachingStrategy
    ) -> Tuple[str, Optional[Dict[str, Any]]]:
        """
        Executes the chosen sub-agent.
        """
        if selected_agent == "QuizAgent":
            answer = self.quiz_agent.generate_quiz(context=context, strategy=strategy)
            return answer, None

        elif selected_agent == "AssessmentAgent":
            answer, meta = self.assessment_agent.evaluate(context=context, strategy=strategy)
            return answer, meta

        else:  # TutorAgent
            answer = self.tutor_agent.generate(context=context, strategy=strategy)
            return answer, None

    # ------------------------------------------------------------------
    # Step 3: ADAPT
    # ------------------------------------------------------------------

    def adapt(
        self,
        selected_agent: str,
        current_state: PedagogyState,
        strategy: TeachingStrategy,
        assessment_result: Optional[Dict[str, Any]],
        student_message: str,
        detected_mood: str
    ) -> PedagogyState:
        """
        Updates the pedagogy state (hint_level, stuck, pedagogy_mode, topic)
        in response to the interaction outcome.
        """
        msg_lower = student_message.lower()
        new_hint_level = current_state.hint_level
        new_stuck = current_state.stuck
        target_topic = strategy.target_concept or current_state.topic

        # Adapt hint level
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

        # Adapt pedagogy mode
        if strategy.recommendation == StrategyAction.EXPLAIN or strategy.recommendation == "explain":
            new_mode = PedagogyMode.DIRECT
        else:
            new_mode = PedagogyMode.SOCRATIC

        return PedagogyState(
            hint_level=new_hint_level,
            topic=target_topic,
            stuck=new_stuck,
            pedagogy_mode=new_mode
        )

    # ------------------------------------------------------------------
    # Full Cycle: 'decide -> teach -> assess -> adapt'
    # ------------------------------------------------------------------

    def reason_turn(
        self,
        context: OrchestratedContext,
        strategy_override: Optional[TeachingStrategy] = None
    ) -> ReasonerResult:
        """
        Executes the complete reasoning turn.
        """
        # Resolve strategy from context or override
        strategy = (
            strategy_override
            or context.learning_context.teaching_strategy
            or TeachingStrategy(
                recommendation=StrategyAction.GUIDE,
                target_concept=context.learning_context.target_concept,
                rationale="Default diagnostic guidance"
            )
        )

        current_state = context.session_context.pedagogy_state

        # 1. DECIDE: Pick responsible agent
        selected_agent = self.decide(context=context, strategy=strategy)
        logger.debug("TutorReasoner DECIDE -> %s", selected_agent)

        # 2. TEACH / ASSESS: Execute agent
        answer, assessment_meta = self.teach_or_assess(
            selected_agent=selected_agent,
            context=context,
            strategy=strategy
        )

        # 3. ADAPT: Update state for next turn
        updated_state = self.adapt(
            selected_agent=selected_agent,
            current_state=current_state,
            strategy=strategy,
            assessment_result=assessment_meta,
            student_message=context.student_message,
            detected_mood=context.session_context.detected_mood
        )

        # Build citations list
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
