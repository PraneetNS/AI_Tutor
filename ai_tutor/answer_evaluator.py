"""
answer_evaluator.py
-------------------
AnswerEvaluator: Fast, testable, non-LLM concept identification and answer evaluation service.

Capabilities:
1. `identify_concepts(student_response, question_context) -> list[concept_id]`:
   - Fast lexical and keyword matching against the ConceptGraph nodes & aliases (cheap & deterministic).
2. `evaluate(response, expected_concepts, question_context, hints_used) -> EvaluationResult`:
   - Computes correctness, concepts touched, partial credit (0.0 to 1.0), and formative feedback.
3. `evaluate_and_update(...) -> EvaluationResult`:
   - Feeds evaluation results directly into `BKTUpdater.update()` and `StrategyEngine`'s
     `learner_strategy_effectiveness` (`times_led_to_mastery` and `times_used`).
"""

from __future__ import annotations

import re
import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Union

from .concept_graph import ConceptGraph, create_ml_concept_graph
from .learner_model import BKTUpdater, MisconceptionEngine
from .models import (
    ConceptMastery,
    EvaluationResult,
    LearnerState,
    StrategyEffectivenessRecord,
)

if TYPE_CHECKING:
    from .strategy_engine import StrategyEngine

logger = logging.getLogger("ai_tutor.answer_evaluator")

# Keyword and alias dictionary for rapid retrieval against concept nodes
CONCEPT_KEYWORD_ALIASES: Dict[str, List[str]] = {
    "variables": ["variable", "variables", "assign", "data type", "primitive", "float", "int", "string"],
    "expressions": ["expression", "operator", "operand", "arithmetic", "boolean logic", "evaluate"],
    "functions": ["function", "functions", "parameter", "argument", "return value", "scope", "def "],
    "linear_algebra": ["matrix", "matrices", "vector", "vectors", "dot product", "eigenvalue", "linear algebra", "rank", "dimension"],
    "calculus_basics": ["calculus", "derivative", "derivatives", "rate of change", "limit", "limits", "slope", "tangent", "d/dx", "integral"],
    "chain_rule": ["chain rule", "composite function", "outer derivative", "inner derivative", "nested function"],
    "partial_derivatives": ["partial derivative", "partial derivatives", "gradient vector", "del", "wrt", "with respect to"],
    "probability": ["probability", "distribution", "expected value", "variance", "bayes", "prior", "posterior", "likelihood"],
    "supervised_learning": ["supervised", "labeled data", "ground truth", "target label", "features", "training set", "regression", "classification"],
    "loss_functions": ["loss function", "loss", "cost function", "mse", "mean squared error", "cross-entropy", "cross entropy", "objective function"],
    "gradient_descent": ["gradient descent", "step size", "learning rate", "descent", "steepest descent", "update weights", "optimization", "optimizer"],
    "backpropagation": ["backpropagation", "backprop", "backward pass", "reverse-mode", "chain rule in networks", "error propagation"],
    "regularization": ["regularization", "l1", "l2", "weight decay", "dropout", "overfitting", "penalty"],
    "neural_networks": ["neural network", "neural networks", "mlp", "perceptron", "hidden layer", "activation function", "weights and biases"],
    "attention_mechanisms": ["attention", "self-attention", "query key value", "qkv", "scaled dot-product"],
    "transformers": ["transformer", "transformers", "multi-head attention", "positional encoding", "bert", "gpt"],
    "gradient_descent_variants": ["adam", "rmsprop", "momentum", "adagrad", "sgd", "stochastic gradient descent"],
}


class AnswerEvaluator:
    """
    Evaluates student free-text responses against expected concepts, computes partial credit,
    and directly drives BKT mastery updates and strategy effectiveness increments.
    """

    def __init__(
        self,
        concept_graph: Optional[ConceptGraph] = None,
        bkt_updater: Optional[BKTUpdater] = None,
        misconception_engine: Optional[MisconceptionEngine] = None,
        keyword_aliases: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        self.concept_graph = concept_graph or create_ml_concept_graph()
        self.bkt_updater = bkt_updater or BKTUpdater()
        self.misconception_engine = misconception_engine or MisconceptionEngine()
        self.aliases = keyword_aliases or CONCEPT_KEYWORD_ALIASES

    def _normalize_text(self, text: str) -> str:
        """Lowercases and cleans punctuation for lexical matching."""
        return re.sub(r"[^\w\s]", " ", (text or "").lower())

    def identify_concepts(
        self,
        student_response: str,
        question_context: Optional[str] = None
    ) -> List[str]:
        """
        Extracts concept IDs touched in the student response using cheap,
        deterministic keyword & graph node retrieval (zero LLM overhead).
        """
        resp_norm = self._normalize_text(student_response)
        q_norm = self._normalize_text(question_context or "")
        combined = f"{q_norm} {resp_norm}"

        matched_concepts: Set[str] = set()

        # 1. Match registered concepts in ConceptGraph
        all_concepts = self.concept_graph.store.all_concepts()
        for node in all_concepts:
            cid = node.concept_id.lower()
            name = self._normalize_text(node.name)

            # Exact ID or name match in response
            if cid in resp_norm or (name and name in resp_norm):
                matched_concepts.add(node.concept_id)

            # Match keyword aliases
            aliases = self.aliases.get(node.concept_id, [])
            for alias in aliases:
                alias_norm = self._normalize_text(alias)
                # Word boundary check for short aliases, substring for multi-word
                if len(alias_norm.split()) > 1:
                    if alias_norm in resp_norm:
                        matched_concepts.add(node.concept_id)
                        break
                else:
                    pattern = rf"\b{re.escape(alias_norm)}\b"
                    if re.search(pattern, resp_norm):
                        matched_concepts.add(node.concept_id)
                        break

        return sorted(list(matched_concepts))

    def evaluate(
        self,
        response: str,
        expected_concepts: Union[str, List[str]],
        question_context: Optional[str] = None,
        hints_used: int = 0
    ) -> EvaluationResult:
        """
        Evaluates the student response against expected concept(s).
        Returns structured EvaluationResult with correctness, partial credit, and concepts touched.
        """
        if isinstance(expected_concepts, str):
            expected_list = [expected_concepts]
        else:
            expected_list = list(expected_concepts)

        expected_normalized = [self._normalize_text(c) for c in expected_list]
        expected_ids = set()
        for c in expected_list:
            # Map name or ID to standard concept_id
            c_norm = self._normalize_text(c)
            node = self.concept_graph.get_concept(c)
            if node:
                expected_ids.add(node.concept_id)
            else:
                expected_ids.add(c)

        # 1. Identify concepts touched in response
        concepts_touched = self.identify_concepts(response, question_context)

        # 2. Check for active misconceptions
        misconceptions_detected: List[str] = []
        for exp in expected_list:
            matched_mc = self.misconception_engine.match(concept=exp, text=response)
            if matched_mc:
                misconceptions_detected.append(matched_mc[0])

        # 3. Calculate concept overlap and partial credit
        matched_expected_count = 0
        resp_norm = self._normalize_text(response)

        for exp_id in expected_ids:
            exp_clean = self._normalize_text(exp_id)
            if exp_id in concepts_touched or exp_clean in resp_norm:
                matched_expected_count += 1
            else:
                # Check aliases
                aliases = self.aliases.get(exp_id, [])
                if any(self._normalize_text(a) in resp_norm for a in aliases):
                    matched_expected_count += 1

        total_expected = max(1, len(expected_ids))
        base_score = matched_expected_count / total_expected

        # Content length & keyword density signal
        words = resp_norm.split()
        if len(words) < 3 and base_score > 0:
            base_score = max(0.2, base_score * 0.5)

        # Penalize for misconceptions
        if misconceptions_detected:
            base_score = max(0.0, base_score - 0.4)

        # Penalize slightly for hints used
        if hints_used > 0:
            score = max(0.0, round(base_score * max(0.4, 1.0 - (hints_used * 0.15)), 2))
        else:
            score = round(base_score, 2)

        correct = (score >= 0.65) and (len(misconceptions_detected) == 0)
        is_mastered = (score >= 0.85) and (hints_used == 0) and (len(misconceptions_detected) == 0)

        # 4. Formulate Feedback
        if correct:
            feedback = f"Accurate explanation covering {', '.join(concepts_touched or expected_list)}."
        elif misconceptions_detected:
            feedback = f"Identified misconception ({', '.join(misconceptions_detected)}). Revisit fundamental definition."
        elif base_score > 0:
            feedback = f"Partially correct. Missing key elements of {', '.join(expected_list)}."
        else:
            feedback = f"Response did not address the expected concept(s): {', '.join(expected_list)}."

        return EvaluationResult(
            correct=correct,
            concepts_touched=concepts_touched,
            partial_credit=score,
            feedback=feedback,
            is_mastered=is_mastered,
            misconceptions_detected=misconceptions_detected,
            updated_p_known=None
        )

    def evaluate_and_update(
        self,
        user_id: Union[int, str],
        response: str,
        expected_concepts: Union[str, List[str]],
        question_context: Optional[str] = None,
        learner_state: Optional[LearnerState] = None,
        strategy_engine: Optional["StrategyEngine"] = None,
        strategy_category: Optional[str] = None,
        strategy_type: Optional[str] = None,
        concept_domain: str = "general",
        hints_used: int = 0,
        prior_mastery_override: Optional[float] = None,
    ) -> EvaluationResult:
        """
        End-to-End Evaluation Hook:
        1. Evaluates response.
        2. Updates BKT Bayesian Knowledge Tracing posterior P(L).
        3. Updates learner_strategy_effectiveness times_used and times_led_to_mastery.
        """
        eval_result = self.evaluate(
            response=response,
            expected_concepts=expected_concepts,
            question_context=question_context,
            hints_used=hints_used
        )

        primary_concept = (
            expected_concepts if isinstance(expected_concepts, str) else expected_concepts[0]
        )

        # 1. BKT Mastery Update
        prior_mastery = 0.30
        if prior_mastery_override is not None:
            prior_mastery = prior_mastery_override
        elif learner_state and primary_concept in learner_state.concept_mastery:
            prior_mastery = learner_state.concept_mastery[primary_concept].mastery

        new_p_known = self.bkt_updater.update(
            prior_mastery=prior_mastery,
            correct=eval_result.correct,
            hints_used=hints_used
        )
        eval_result.updated_p_known = round(new_p_known, 4)

        if learner_state:
            if primary_concept not in learner_state.concept_mastery:
                learner_state.concept_mastery[primary_concept] = ConceptMastery(
                    concept=primary_concept,
                    mastery=new_p_known,
                    attempts=1,
                    correct=1 if eval_result.correct else 0
                )
            else:
                cm = learner_state.concept_mastery[primary_concept]
                cm.mastery = new_p_known
                cm.attempts += 1
                if eval_result.correct:
                    cm.correct += 1

        # 2. Strategy Effectiveness Update
        if strategy_engine and strategy_category and strategy_type:
            strategy_engine.record_assess_result(
                user_id=user_id,
                strategy_category=strategy_category,
                strategy_type=strategy_type,
                concept_domain=concept_domain,
                assess_result="mastered" if eval_result.is_mastered else "in_progress"
            )

        logger.info(
            "[AnswerEvaluator] Evaluated user=%s concept=%s -> correct=%s score=%.2f p_known=%.4f mastered=%s",
            str(user_id), primary_concept, eval_result.correct, eval_result.partial_credit,
            new_p_known, eval_result.is_mastered
        )

        return eval_result
