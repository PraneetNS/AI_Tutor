"""
tests/test_truthfulness_checker.py
-----------------------------------
Unit tests for the TruthfulnessChecker stage of the output guardrail:
- Math & derivation symbolic/numeric verification (derivatives, integrals, arithmetic, algebra)
- Factual verification against RAG ground truth sources
- Affirmation and praise detection
- Rejection of draft responses validating incorrect student claims
- Explicit step-naming regeneration instructions (no generic encouragement on errors)
- Integration with GuardrailPipeline and ResponseGuardrail
"""

import pytest
from ai_tutor import (
    TruthfulnessChecker,
    TruthfulnessResult,
    GuardrailPipeline,
    ResponseGuardrail,
    PedagogyState,
    AIChatRequest,
    SourceCitation,
)


# ---------------------------------------------------------------------------
# 1. Math Verification Tests
# ---------------------------------------------------------------------------

class TestMathVerification:
    def setup_method(self):
        self.checker = TruthfulnessChecker(use_sympy=True)

    def test_derivative_incorrect_claim_flagged(self):
        # Student claims derivative of x^3 is 2x^2 (actual is 3x^2)
        res = self.checker.extract_and_verify_math("I think the derivative of x^3 is 2x^2.")
        assert res is not None
        assert res.is_valid is False
        assert "3*x^2" in res.expected_result or "3x^2" in res.expected_result
        assert "2x^2" in res.actual_claimed
        assert "is 3*x^2" in res.specific_error or "is 3x^2" in res.specific_error

    def test_derivative_correct_claim_passes(self):
        res = self.checker.extract_and_verify_math("So d/dx(x^3) = 3x^2, right?")
        assert res is not None
        assert res.is_valid is True

    def test_integral_incorrect_claim_flagged(self):
        # Student claims integral of 2x is x^3 (actual is x^2)
        res = self.checker.extract_and_verify_math("The integral of 2x is x^3 + C")
        assert res is not None
        assert res.is_valid is False
        assert "x^2" in res.expected_result
        assert "x^3" in res.actual_claimed

    def test_integral_correct_claim_passes(self):
        res = self.checker.extract_and_verify_math("The integral of 2x is x^2")
        assert res is not None
        assert res.is_valid is True

    def test_arithmetic_incorrect_equality_flagged(self):
        res = self.checker.extract_and_verify_math("So 2 + 2 = 5")
        assert res is not None
        assert res.is_valid is False
        assert res.expected_result in ["4", "4.0"]
        assert res.actual_claimed == "5"
        assert "yields 4" in res.specific_error or "evaluates to 4" in res.specific_error

    def test_arithmetic_multiplication_error_flagged(self):
        res = self.checker.extract_and_verify_math("I calculated 5 * 8 = 45")
        assert res is not None
        assert res.is_valid is False
        assert res.expected_result in ["40", "40.0"]

    def test_algebraic_incorrect_expansion_flagged(self):
        # Student claims (x+1)^2 = x^2 + 1 (missing 2x)
        res = self.checker.extract_and_verify_math("Therefore (x+1)^2 = x^2 + 1")
        assert res is not None
        assert res.is_valid is False
        assert "2*x" in res.expected_result or "2x" in res.expected_result

    def test_algebraic_correct_expansion_passes(self):
        res = self.checker.extract_and_verify_math("(x+1)^2 = x^2 + 2x + 1")
        assert res is not None
        assert res.is_valid is True


# ---------------------------------------------------------------------------
# 2. Factual / RAG Verification Tests
# ---------------------------------------------------------------------------

class TestFactualVerification:
    def setup_method(self):
        self.checker = TruthfulnessChecker()
        self.sources = [
            SourceCitation(
                lecture_id=1,
                title="Supervised Learning Basics",
                snippet="Supervised learning algorithms are trained on labeled data pairs (x, y)."
            ),
            SourceCitation(
                lecture_id=2,
                title="Regularization",
                snippet="L1 regularization produces sparse feature weights, while L2 shrinks weights without exact zeros."
            ),
        ]

    def test_factual_unlabeled_data_contradiction_flagged(self):
        student_msg = "Supervised learning uses unlabeled data to discover hidden patterns."
        res = self.checker.extract_and_verify_factual(student_msg, sources=self.sources)
        assert res is not None
        assert res.is_valid is False
        assert "labeled" in res.specific_error

    def test_factual_l2_sparsity_contradiction_flagged(self):
        student_msg = "L2 regularization produces sparse weights for feature selection."
        res = self.checker.extract_and_verify_factual(student_msg, sources=self.sources)
        assert res is not None
        assert res.is_valid is False
        assert "L1" in res.specific_error

    def test_factual_correct_claim_passes(self):
        student_msg = "L1 regularization drives weights to zero producing sparsity."
        res = self.checker.extract_and_verify_factual(student_msg, sources=self.sources)
        assert res is None or res.is_valid is True


# ---------------------------------------------------------------------------
# 3. Affirmation Detection Tests
# ---------------------------------------------------------------------------

class TestAffirmationDetection:
    def setup_method(self):
        self.checker = TruthfulnessChecker()

    def test_detects_praise_and_affirmation_phrases(self):
        assert self.checker.detects_affirmation("Great job! That's completely correct.") is True
        assert self.checker.detects_affirmation("Spot on! You got it right.") is True
        assert self.checker.detects_affirmation("Excellent work! That is the right answer.") is True
        assert self.checker.detects_affirmation("Correct! Your derivation is sound.") is True
        assert self.checker.detects_affirmation("Yes, exactly! Well done.") is True

    def test_critique_or_neutral_not_affirmation(self):
        assert self.checker.detects_affirmation("Let's look at that step again. What is the derivative of x^3?") is False
        assert self.checker.detects_affirmation("Not quite, let's recalculate the power rule.") is False
        assert self.checker.detects_affirmation("Can you explain why you multiplied by 2?") is False


# ---------------------------------------------------------------------------
# 4. Truthfulness Check Rejection & Explicit Instruction Tests
# ---------------------------------------------------------------------------

class TestTruthfulnessCheckerLogic:
    def setup_method(self):
        self.checker = TruthfulnessChecker(use_sympy=True)

    def test_rejects_praise_on_wrong_math_derivation(self):
        student_msg = "The derivative of x^3 is 2x^2."
        draft_response = "Great job! That's correct, you applied the power rule nicely."

        truth_res = self.checker.check_truthfulness(
            student_message=student_msg,
            draft_response=draft_response,
        )

        assert truth_res.is_truthful is False
        assert truth_res.rejected is True
        assert truth_res.affirmed_incorrect_claim is True
        # Explicit error named
        assert "derivative of x^3" in truth_res.specific_error
        # Regeneration instruction contains explicit guidance
        assert "REGENERATE INSTRUCTION" in truth_res.regeneration_instruction
        assert "Never provide generic praise" in truth_res.regeneration_instruction

    def test_allows_corrective_response_on_wrong_math_derivation(self):
        student_msg = "The derivative of x^3 is 2x^2."
        # Assistant correctly questions or fixes it -> No false affirmation -> Allowed!
        draft_response = "Let's check the power rule: when you bring down the power 3, what power remains?"

        truth_res = self.checker.check_truthfulness(
            student_message=student_msg,
            draft_response=draft_response,
        )

        assert truth_res.is_truthful is True
        assert truth_res.rejected is False

    def test_allows_praise_on_correct_math_derivation(self):
        student_msg = "The derivative of x^3 is 3x^2."
        draft_response = "Great job! That's completely correct."

        truth_res = self.checker.check_truthfulness(
            student_message=student_msg,
            draft_response=draft_response,
        )

        assert truth_res.is_truthful is True
        assert truth_res.rejected is False

    def test_rejects_praise_on_factual_misconception(self):
        sources = [
            SourceCitation(
                lecture_id=1,
                title="ML 101",
                snippet="Supervised learning uses labeled training examples."
            )
        ]
        student_msg = "Supervised learning uses unlabeled data."
        draft_response = "Spot on! That is the right approach."

        truth_res = self.checker.check_truthfulness(
            student_message=student_msg,
            draft_response=draft_response,
            sources=sources,
        )

        assert truth_res.is_truthful is False
        assert truth_res.rejected is True
        assert "labeled" in truth_res.specific_error
        assert "REGENERATE INSTRUCTION" in truth_res.regeneration_instruction


# ---------------------------------------------------------------------------
# 5. GuardrailPipeline & ResponseGuardrail Integration
# ---------------------------------------------------------------------------

class TestGuardrailPipelineTruthfulnessIntegration:
    def setup_method(self):
        self.pipeline = GuardrailPipeline()

    def test_guardrail_pipeline_blocks_false_affirmation(self):
        student_msg = "I calculated 2 + 2 = 5."
        draft_resp = "Great job! You nailed it."

        out_res = self.pipeline.check_output(
            response=draft_resp,
            student_message=student_msg,
        )

        assert out_res.blocked is True
        assert out_res.is_truthful is False
        assert "Truthfulness check rejected draft" in out_res.reason
        assert out_res.regeneration_instruction is not None
        assert "2 + 2" in out_res.regeneration_instruction

    def test_guardrail_pipeline_passes_truthful_affirmation(self):
        student_msg = "I calculated 2 + 2 = 4."
        draft_resp = "Great job! You nailed it."

        out_res = self.pipeline.check_output(
            response=draft_resp,
            student_message=student_msg,
        )

        assert out_res.blocked is False
        assert out_res.is_truthful is True

    def test_response_guardrail_sanitizes_false_praise_with_correction(self):
        guardrail = ResponseGuardrail()
        req = AIChatRequest(message="The derivative of x^3 is 2x^2.")
        false_praise = "Great job! That's correct!"

        res = guardrail.validate_and_sanitize(
            raw_answer=false_praise,
            pedagogy_state=PedagogyState(),
            request=req,
        )

        # False praise is stripped, replaced by specific correction, flag is set
        assert "UNTRUTHFUL_AFFIRMATION_REJECTED" in res.flags
        assert "Great job" not in res.sanitized_answer
        assert "derivative of x^3" in res.sanitized_answer or "3*x^2" in res.sanitized_answer
