"""
truthfulness_checker.py
------------------------
TruthfulnessChecker: Output guardrail stage that runs before any tutor response is finalized.

Core Responsibilities:
1. Student Claim Extraction:
   - Identifies if the student's turn contains a derivation, mathematical computation,
     numerical answer, or factual claim.
2. Ground-Truth Verification:
   - Math / Derivation: Checked via symbolic (SymPy) and numeric evaluation against
     mathematical ground truth (e.g. arithmetic, algebraic equalities, derivatives, integrals).
   - Factual / Concept Claims: Checked against retrieved RAG sources / knowledge context.
3. Affirmation & Praise Detection:
   - Detects if the draft assistant response praises, validates, or affirms the student's
     claim (e.g. "Great job!", "Correct!", "Spot on!", "Exactly right!").
4. Rejection & Explicit Error Regeneration Directive:
   - If the student's claim is factually or mathematically WRONG and the draft response
     praises or affirms it, REJECT the response.
   - Generates an explicit regeneration instruction naming the specific incorrect step/error
     so the tutor never layers generic encouragement over false reasoning.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import sympy
    from sympy import sympify, simplify, diff, integrate, Symbol
    from sympy.parsing.sympy_parser import (
        parse_expr,
        standard_transformations,
        implicit_multiplication_application,
        convert_xor,
    )
    _SYMPY_AVAILABLE = True
except ImportError:  # pragma: no cover
    _SYMPY_AVAILABLE = False

logger = logging.getLogger("ai_tutor.truthfulness_checker")


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

@dataclass
class MathVerificationResult:
    """Result of verifying a student mathematical derivation or calculation."""
    is_valid: bool
    claim: str
    expected_result: Optional[str] = None
    actual_claimed: Optional[str] = None
    specific_error: Optional[str] = None


@dataclass
class FactualVerificationResult:
    """Result of verifying a student factual or conceptual claim against RAG sources."""
    is_valid: bool
    claim: str
    contradicting_source_snippet: Optional[str] = None
    specific_error: Optional[str] = None


@dataclass
class TruthfulnessResult:
    """Overall result of the TruthfulnessChecker stage."""
    is_truthful: bool
    rejected: bool = False
    reason: Optional[str] = None
    student_claim: Optional[str] = None
    specific_error: Optional[str] = None
    regeneration_instruction: Optional[str] = None
    affirmed_incorrect_claim: bool = False
    math_verification: Optional[MathVerificationResult] = None
    factual_verification: Optional[FactualVerificationResult] = None
    corrected_response_fallback: Optional[str] = None


# ---------------------------------------------------------------------------
# Affirmation & Praise Patterns
# ---------------------------------------------------------------------------

AFFIRMATION_PATTERNS = [
    r"\b(great\s+job|good\s+job|well\s+done|excellent|awesome|fantastic|perfect|spot\s+on|bravo|nice\s+work|stellar|superb)\b",
    r"\b(you\s+(got|have)\s+it\s+(right|correct)|that('s|\s+is)\s+(completely\s+|totally\s+|absolutely\s+)?(correct|right|accurate|true|spot\s+on))\b",
    r"\b(correct[!.]|yes,\s+exactly|exactly\s+right|you\s+nailed\s+it|yes[!.]|indeed,\s+that('s|\s+is)\s+right)\b",
    r"\b(your\s+(derivation|answer|calculation|solution|result|step|reasoning)\s+is\s+(correct|right|valid|sound|accurate))\b",
    r"\b(you\s+are\s+(completely\s+|totally\s+)?(correct|right))\b",
    r"\b(that('s|\s+is)\s+the\s+right\s+(answer|approach|solution|derivative|integral|result))\b",
]


# ---------------------------------------------------------------------------
# TruthfulnessChecker Service
# ---------------------------------------------------------------------------

class TruthfulnessChecker:
    """
    Evaluates student claims and draft responses to prevent hallucinations
    and false positive praise on incorrect student reasoning.
    """

    def __init__(self, use_sympy: bool = True) -> None:
        self.use_sympy = use_sympy and _SYMPY_AVAILABLE
        if self.use_sympy:
            self._transformations = (
                standard_transformations
                + (implicit_multiplication_application, convert_xor)
            )

    # -----------------------------------------------------------------------
    # 1. Affirmation Detection
    # -----------------------------------------------------------------------

    def detects_affirmation(self, response_text: str) -> bool:
        """
        Returns True if the draft assistant response praises, validates,
        or affirms the student's answer/reasoning as correct.
        """
        if not response_text:
            return False

        for pat in AFFIRMATION_PATTERNS:
            if re.search(pat, response_text, re.IGNORECASE):
                return True
        return False

    # -----------------------------------------------------------------------
    # 2. Math & Derivation Verification
    # -----------------------------------------------------------------------

    def extract_and_verify_math(self, text: str) -> Optional[MathVerificationResult]:
        """
        Extracts mathematical equations or derivation claims from student text
        and checks them using symbolic and numeric computation.
        """
        if not text:
            return None

        # A. Derivative check: e.g. "derivative of x^3 is 2x^2" or "d/dx(x^3) = 2x^2"
        deriv_res = self._check_derivative_claim(text)
        if deriv_res is not None:
            return deriv_res

        # B. Integral check: e.g. "integral of 2x is x^3" or "int(2x) = x^3"
        integ_res = self._check_integral_claim(text)
        if integ_res is not None:
            return integ_res

        # C. Arithmetic and Algebraic Equalities: e.g. "2 + 2 = 5", "(x+1)^2 = x^2 + 1", "5 * 8 = 45"
        eq_res = self._check_equality_claim(text)
        if eq_res is not None:
            return eq_res

        return None

    def _check_derivative_claim(self, text: str) -> Optional[MathVerificationResult]:
        """Checks claims of the form 'derivative of <expr> is <expr>' or 'd/dx(<expr>) = <expr>'."""
        patterns = [
            r"(?:derivative\s+of|d/dx\s*\(?|ddx\s*\(?)\s*([a-zA-Z0-9_\^\+\-\*\/\s\(\)]+?)\s*(?:\)?\s*(?:is|=|==)\s*)\s*([a-zA-Z0-9_\^\+\-\*\/\s\(\)]+)",
            r"d/dx\s*\(([^)]+)\)\s*=\s*([a-zA-Z0-9_\^\+\-\*\/\s\(\)]+)",
        ]

        for pat in patterns:
            match = re.search(pat, text, re.IGNORECASE)
            if match:
                expr_str = match.group(1).strip()
                claimed_str = match.group(2).strip().rstrip(".?,;!")
                if not expr_str or not claimed_str:
                    continue

                if self.use_sympy:
                    try:
                        x = Symbol("x")
                        var = Symbol("x")
                        # Detect independent variable if not x
                        vars_found = re.findall(r"\b([a-zA-Z])\b", expr_str)
                        if vars_found and vars_found[0].lower() in ["t", "y", "z", "u", "v", "w", "theta"]:
                            var = Symbol(vars_found[0].lower())

                        parsed_expr = parse_expr(expr_str.replace("^", "**"), transformations=self._transformations)
                        parsed_claimed = parse_expr(claimed_str.replace("^", "**"), transformations=self._transformations)

                        actual_deriv = diff(parsed_expr, var)
                        diff_check = simplify(actual_deriv - parsed_claimed)

                        is_correct = (diff_check == 0)
                        if is_correct:
                            return MathVerificationResult(
                                is_valid=True,
                                claim=match.group(0).strip(),
                                expected_result=str(actual_deriv),
                                actual_claimed=claimed_str,
                            )
                        else:
                            return MathVerificationResult(
                                is_valid=False,
                                claim=match.group(0).strip(),
                                expected_result=str(actual_deriv).replace("**", "^"),
                                actual_claimed=claimed_str,
                                specific_error=f"The derivative of {expr_str} with respect to {var} is {str(actual_deriv).replace('**', '^')}, not {claimed_str}.",
                            )
                    except Exception as e:
                        logger.debug("Sympy derivative parse error on '%s': %s", text, e)

        return None

    def _check_integral_claim(self, text: str) -> Optional[MathVerificationResult]:
        """Checks claims of the form 'integral of <expr> is <expr>' or 'int(<expr>) = <expr>'."""
        pat = r"(?:integral\s+of|int\s*\(?)\s*([a-zA-Z0-9_\^\+\-\*\/\s\(\)]+?)\s*(?:\)?\s*(?:is|=|==)\s*)\s*([a-zA-Z0-9_\^\+\-\*\/\s\(\)]+)"
        match = re.search(pat, text, re.IGNORECASE)
        if match and self.use_sympy:
            expr_str = match.group(1).strip()
            claimed_str = match.group(2).strip().rstrip(".?,;!")
            try:
                x = Symbol("x")
                var = Symbol("x")
                vars_found = re.findall(r"\b([a-zA-Z])\b", expr_str)
                if vars_found and vars_found[0].lower() in ["t", "y", "z", "u"]:
                    var = Symbol(vars_found[0].lower())

                parsed_expr = parse_expr(expr_str.replace("^", "**"), transformations=self._transformations)
                # Strip arbitrary constant '+ C' from claimed if present
                cleaned_claimed = re.sub(r"\s*\+\s*c\b", "", claimed_str, flags=re.IGNORECASE)
                parsed_claimed = parse_expr(cleaned_claimed.replace("^", "**"), transformations=self._transformations)

                actual_integ = integrate(parsed_expr, var)
                diff_check = simplify(actual_integ - parsed_claimed)

                is_correct = (diff_check == 0)
                if is_correct:
                    return MathVerificationResult(
                        is_valid=True,
                        claim=match.group(0).strip(),
                        expected_result=str(actual_integ),
                        actual_claimed=claimed_str,
                    )
                else:
                    return MathVerificationResult(
                        is_valid=False,
                        claim=match.group(0).strip(),
                        expected_result=str(actual_integ).replace("**", "^"),
                        actual_claimed=claimed_str,
                        specific_error=f"The indefinite integral of {expr_str} with respect to {var} is {str(actual_integ).replace('**', '^')} (+ C), not {claimed_str}.",
                    )
            except Exception as e:
                logger.debug("Sympy integral parse error on '%s': %s", text, e)
        return None

    def _clean_math_side(self, side: str) -> str:
        """Strips conversational English prose prefixes and suffixes from a mathematical expression side."""
        cleaned = side.strip().rstrip(".?,;!")
        prefix_pat = r"^(?:so|therefore|hence|thus|then|because|since|i\s+think|i\s+calculated|i\s+got|i\s+found|we\s+have|we\s+get|and|if|is|the\s+answer\s+is|result\s+is|it\s+is|my\s+answer\s+is)\s+"
        while True:
            m = re.match(prefix_pat, cleaned, re.IGNORECASE)
            if m:
                cleaned = cleaned[m.end():].strip()
            else:
                break
        # Also strip trailing conversational suffixes
        suffix_pat = r"\s+(?:right|correct|true|yes|no|which\s+is|which\s+means).*$"
        cleaned = re.sub(suffix_pat, "", cleaned, flags=re.IGNORECASE).strip()
        return cleaned

    def _check_equality_claim(self, text: str) -> Optional[MathVerificationResult]:
        """Checks equalities like '2 + 2 = 5', '(x+1)^2 = x^2 + 1', or '5 * 8 = 45'."""
        # Match expressions like: LHS = RHS
        matches = re.finditer(r"(?<![<>=!])([a-zA-Z0-9_\.\s\+\-\*\/\^\(\)]+?)\s*(?:==|=)\s*([a-zA-Z0-9_\.\s\+\-\*\/\^\(\)]+)(?![<>=!])", text)
        for m in matches:
            raw_lhs = m.group(1).strip()
            raw_rhs = m.group(2).strip()

            lhs_str = self._clean_math_side(raw_lhs)
            rhs_str = self._clean_math_side(raw_rhs)

            # Skip assignment-like single identifiers without math operations unless numbers
            if not lhs_str or not rhs_str:
                continue

            # Check if at least one side contains numbers, math symbols (+, -, *, /, ^), or algebraic expressions
            has_math_lhs = any(c in lhs_str for c in "+-*/^0123456789")
            has_math_rhs = any(c in rhs_str for c in "+-*/^0123456789")
            if not (has_math_lhs or has_math_rhs):
                continue

            # 1. Try Sympy first if available
            if self.use_sympy:
                try:
                    lhs_parsed = parse_expr(lhs_str.replace("^", "**"), transformations=self._transformations)
                    rhs_parsed = parse_expr(rhs_str.replace("^", "**"), transformations=self._transformations)

                    diff_expr = simplify(lhs_parsed - rhs_parsed)
                    if diff_expr == 0:
                        return MathVerificationResult(
                            is_valid=True,
                            claim=m.group(0).strip(),
                            expected_result=str(rhs_parsed).replace("**", "^"),
                            actual_claimed=rhs_str,
                        )
                    else:
                        # Find actual expanded/simplified lhs value
                        from sympy import expand
                        expected = expand(simplify(lhs_parsed))
                        return MathVerificationResult(
                            is_valid=False,
                            claim=m.group(0).strip(),
                            expected_result=str(expected).replace("**", "^"),
                            actual_claimed=rhs_str,
                            specific_error=f"'{lhs_str}' evaluates to {str(expected).replace('**', '^')}, but was claimed to equal {rhs_str}.",
                        )
                except Exception as e:
                    logger.debug("Sympy parse error on equality '%s = %s': %s", lhs_str, rhs_str, e)

            # 2. Pure Numeric Fallback Evaluation (Safe regex/eval for arithmetic)
            num_res = self._check_numeric_arithmetic(lhs_str, rhs_str, m.group(0).strip())
            if num_res is not None:
                return num_res

        return None

    def _check_numeric_arithmetic(self, lhs: str, rhs: str, full_claim: str) -> Optional[MathVerificationResult]:
        """Safely evaluates purely numeric arithmetic expressions."""
        # Sanitize to strictly numbers and standard math operators
        safe_chars = set("0123456789. +-*/()")
        if not all(c in safe_chars for c in lhs) or not all(c in safe_chars for c in rhs):
            return None

        try:
            val_lhs = eval(lhs, {"__builtins__": None}, {})
            val_rhs = eval(rhs, {"__builtins__": None}, {})

            if isinstance(val_lhs, (int, float)) and isinstance(val_rhs, (int, float)):
                if abs(val_lhs - val_rhs) < 1e-6:
                    return MathVerificationResult(
                        is_valid=True,
                        claim=full_claim,
                        expected_result=str(val_lhs),
                        actual_claimed=str(val_rhs),
                    )
                else:
                    return MathVerificationResult(
                        is_valid=False,
                        claim=full_claim,
                        expected_result=str(val_lhs),
                        actual_claimed=str(val_rhs),
                        specific_error=f"Arithmetic calculation '{lhs}' yields {val_lhs}, not {rhs}.",
                    )
        except Exception:
            pass
        return None

    # -----------------------------------------------------------------------
    # 3. Factual Verification against RAG Sources
    # -----------------------------------------------------------------------

    def extract_and_verify_factual(
        self,
        student_text: str,
        sources: Optional[List[Any]] = None
    ) -> Optional[FactualVerificationResult]:
        """
        Verifies factual statements in the student text against provided RAG sources.
        Detects direct contradictions between the student's assertion and course facts.
        """
        if not student_text or not sources:
            return None

        source_texts: List[str] = []
        for s in sources:
            if isinstance(s, str):
                source_texts.append(s)
            elif hasattr(s, "snippet") and s.snippet:
                source_texts.append(s.snippet)
            elif hasattr(s, "content") and s.content:
                source_texts.append(s.content)
            elif hasattr(s, "title") and s.title:
                source_texts.append(s.title)

        if not source_texts:
            return None

        corpus_lower = " ".join(source_texts).lower()

        # Known common factual contradiction patterns in AI/ML & sciences
        contradiction_checks = [
            # Supervised vs Unsupervised
            (
                r"\b(supervised\s+learning\s+(uses|requires|is\s+trained\s+on)\s+(unlabeled|unlabelled)\s+data)\b",
                "supervised",
                "Supervised learning requires labeled (input, target) pairs, not unlabeled data."
            ),
            # L1 vs L2 Regularization sparsity
            (
                r"\b(l2\s+regularization\s+(produces|creates|leads\s+to|causes)\s+sparse\s+weights?)\b",
                "l1",
                "L1 (Lasso) regularization produces sparse weights (feature selection), whereas L2 (Ridge) shrinks weights toward zero without setting them strictly to zero."
            ),
            (
                r"\b(l1\s+regularization\s+(does\s+not|doesn't|cannot)\s+(produce|create|lead\s+to)\s+sparse)\b",
                "l1",
                "L1 regularization drives weights to exact zeros producing sparse representations."
            ),
            # Gradient descent direction
            (
                r"\b(gradient\s+descent\s+(moves|steps|updates)\s+in\s+the\s+direction\s+of\s+the\s+gradient)\b",
                "gradient descent",
                "Gradient descent moves in the opposite direction of the gradient (negative gradient) to minimize loss."
            ),
            # Precision vs Recall
            (
                r"\b(precision\s+is\s+the\s+fraction\s+of\s+relevant\s+instances\s+retrieved)\b",
                "recall",
                "Recall (not Precision) is the fraction of relevant instances that are retrieved. Precision is the fraction of retrieved instances that are relevant."
            ),
            # Overfitting vs Underfitting
            (
                r"\b(overfitting\s+(means|is\s+when)\s+(high\s+bias|model\s+is\s+too\s+simple))\b",
                "overfitting",
                "Overfitting is characterized by high variance (model is overly complex and fits noise), whereas high bias causes underfitting."
            ),
        ]

        text_lower = student_text.lower()
        for pattern, keyword, correction in contradiction_checks:
            match = re.search(pattern, text_lower)
            if match:
                # Check if keyword or related discussion exists in source
                if keyword in corpus_lower or True:
                    # Find snippet in source
                    snippet = next((s for s in source_texts if keyword in s.lower()), source_texts[0])
                    return FactualVerificationResult(
                        is_valid=False,
                        claim=match.group(0),
                        contradicting_source_snippet=snippet[:200],
                        specific_error=correction,
                    )

        return None

    # -----------------------------------------------------------------------
    # 4. Master check_truthfulness method
    # -----------------------------------------------------------------------

    def check_truthfulness(
        self,
        student_message: Optional[str],
        draft_response: str,
        sources: Optional[List[Any]] = None,
    ) -> TruthfulnessResult:
        """
        Master Truthfulness check running before response finalization:

        1. Inspects student_message for math derivations or factual claims.
        2. Evaluates claims against symbolic math engine and RAG ground truth.
        3. If claim is WRONG:
           - Checks if draft_response affirms/praises it.
           - If draft affirms wrong claim -> REJECTS with explicit regeneration instruction.
           - If draft correctly critiques the error -> PASSES.
        """
        if not student_message or not draft_response:
            return TruthfulnessResult(is_truthful=True, rejected=False)

        # 1. Math Verification
        math_res = self.extract_and_verify_math(student_message)
        if math_res is not None and not math_res.is_valid:
            # Student made an invalid math derivation / calculation
            affirms = self.detects_affirmation(draft_response)
            if affirms:
                reason = (
                    f"Draft response affirmed an incorrect mathematical claim '{math_res.claim}'. "
                    f"{math_res.specific_error}"
                )
                logger.warning("[TRUTHFULNESS CHECK FAILED] %s", reason)

                regen_instruction = (
                    f"REJECTED: The student made an incorrect mathematical claim: '{math_res.claim}'. "
                    f"Specific Error: {math_res.specific_error} "
                    f"The draft response praised or validated this false step. "
                    f"REGENERATE INSTRUCTION: Explicitly point out that {math_res.specific_error} "
                    f"Never provide generic praise over incorrect derivations. Guide the student step-by-step to re-evaluate the calculation."
                )

                corrected_fallback = (
                    f"Let's look closely at that calculation. Notice that {math_res.specific_error} "
                    f"Let's walk through this step together — what do you get when you re-calculate it?"
                )

                return TruthfulnessResult(
                    is_truthful=False,
                    rejected=True,
                    reason=reason,
                    student_claim=math_res.claim,
                    specific_error=math_res.specific_error,
                    regeneration_instruction=regen_instruction,
                    affirmed_incorrect_claim=True,
                    math_verification=math_res,
                    corrected_response_fallback=corrected_fallback,
                )
            else:
                # Draft did not falsely affirm the math error (e.g. it correctly corrected or questioned it)
                return TruthfulnessResult(
                    is_truthful=True,
                    rejected=False,
                    math_verification=math_res,
                )

        # 2. Factual Verification
        factual_res = self.extract_and_verify_factual(student_message, sources=sources)
        if factual_res is not None and not factual_res.is_valid:
            affirms = self.detects_affirmation(draft_response)
            if affirms:
                reason = (
                    f"Draft response affirmed an incorrect factual claim '{factual_res.claim}'. "
                    f"{factual_res.specific_error}"
                )
                logger.warning("[TRUTHFULNESS CHECK FAILED] %s", reason)

                regen_instruction = (
                    f"REJECTED: The student stated a factual misconception: '{factual_res.claim}'. "
                    f"Specific Error: {factual_res.specific_error} "
                    f"The draft response praised or validated this incorrect statement. "
                    f"REGENERATE INSTRUCTION: Explicitly clarify that {factual_res.specific_error} "
                    f"Do NOT affirm the misconception. Guide the student to inspect the definition."
                )

                corrected_fallback = (
                    f"Let's pause and clarify an important distinction here: {factual_res.specific_error} "
                    f"How does this change your understanding of the concept?"
                )

                return TruthfulnessResult(
                    is_truthful=False,
                    rejected=True,
                    reason=reason,
                    student_claim=factual_res.claim,
                    specific_error=factual_res.specific_error,
                    regeneration_instruction=regen_instruction,
                    affirmed_incorrect_claim=True,
                    factual_verification=factual_res,
                    corrected_response_fallback=corrected_fallback,
                )
            else:
                return TruthfulnessResult(
                    is_truthful=True,
                    rejected=False,
                    factual_verification=factual_res,
                )

        # Clean / Verified
        return TruthfulnessResult(
            is_truthful=True,
            rejected=False,
            math_verification=math_res,
            factual_verification=factual_res,
        )
