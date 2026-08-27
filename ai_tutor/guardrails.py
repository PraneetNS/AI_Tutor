import re
import json
import logging
from typing import List, Dict, Any, Optional, Tuple, Type, Union
from pydantic import BaseModel

from .models import (
    PedagogyState,
    PedagogyMode,
    AIChatRequest,
    Chunk,
    SourceCitation
)
from .truthfulness_checker import (
    TruthfulnessChecker,
    TruthfulnessResult,
    MathVerificationResult,
    FactualVerificationResult,
)

logger = logging.getLogger("ai_tutor.guardrails")


# ---------------------------------------------------------------------------
# Guardrail Stage Result Data Structures
# ---------------------------------------------------------------------------

class InputGuardrailResult(dict):
    """
    Result returned by GuardrailPipeline.check_input(message).
    Inherits from dict so that result['blocked'], result['reason'], result['redirect_mode']
    work seamlessly while supporting attribute access as well.
    """
    def __init__(
        self,
        blocked: bool = False,
        reason: Optional[str] = None,
        redirect_mode: Optional[str] = None,
        supportive_response: Optional[str] = None,
        short_circuited: bool = False,
    ):
        super().__init__(
            blocked=blocked,
            reason=reason,
            redirect_mode=redirect_mode,
            supportive_response=supportive_response,
            short_circuited=short_circuited,
        )
        self.blocked = blocked
        self.reason = reason
        self.redirect_mode = redirect_mode
        self.supportive_response = supportive_response
        self.short_circuited = short_circuited


class OutputGuardrailResult(dict):
    """
    Result returned by GuardrailPipeline.check_output(response, sources, expected_schema, student_message).
    Inherits from dict so that result['blocked'], result['reason'], result['sanitized_response']
    work seamlessly while supporting attribute access as well.
    """
    def __init__(
        self,
        blocked: bool = False,
        reason: Optional[str] = None,
        sanitized_response: str = "",
        is_grounded: bool = True,
        grounding_score: float = 1.0,
        schema_valid: bool = True,
        repair_attempted: bool = False,
        is_truthful: bool = True,
        regeneration_instruction: Optional[str] = None,
        truth_violation_reason: Optional[str] = None,
    ):
        super().__init__(
            blocked=blocked,
            reason=reason,
            sanitized_response=sanitized_response,
            is_grounded=is_grounded,
            grounding_score=grounding_score,
            schema_valid=schema_valid,
            repair_attempted=repair_attempted,
            is_truthful=is_truthful,
            regeneration_instruction=regeneration_instruction,
            truth_violation_reason=truth_violation_reason,
        )
        self.blocked = blocked
        self.reason = reason
        self.sanitized_response = sanitized_response
        self.is_grounded = is_grounded
        self.grounding_score = grounding_score
        self.schema_valid = schema_valid
        self.repair_attempted = repair_attempted
        self.is_truthful = is_truthful
        self.regeneration_instruction = regeneration_instruction
        self.truth_violation_reason = truth_violation_reason


class GuardrailResult:
    """Legacy response container maintained for backwards compatibility."""
    def __init__(
        self,
        sanitized_answer: str,
        is_safe: bool = True,
        is_grounded: bool = True,
        grounding_score: float = 1.0,
        flags: Optional[List[str]] = None
    ):
        self.sanitized_answer = sanitized_answer
        self.is_safe = is_safe
        self.is_grounded = is_grounded
        self.grounding_score = grounding_score
        self.flags = flags or []


# ---------------------------------------------------------------------------
# Two-Stage Provider-Agnostic Guardrail Pipeline
# ---------------------------------------------------------------------------

class GuardrailPipeline:
    """
    Production Two-Stage Provider-Agnostic Guardrail Pipeline:

    Stage 1: check_input(message) -> {blocked: bool, reason: str, redirect_mode: str}
      - Crisis-signal detector (short-circuits straight to fixed supportive response bypassing reasoner)
      - Prompt-injection pattern & classifier check
      - Toxicity & dangerous content filter
      - Academic-integrity classifier (returns redirect_mode='socratic' rather than blocking outright)

    Stage 2: check_output(response, sources) -> {blocked: bool, reason: str, sanitized_response: str}
      - System-prompt-leak regex / classifier check & scrubber
      - JSON schema validation with one deterministic repair retry on failure
      - Grounding check against provided course sources / chunks
    """

    CRISIS_PATTERNS = [
        r"\b(suicid(e|al)|kill\s+(my|one)?self|end\s+my\s+life|want\s+to\s+die|hang\s+myself)\b",
        r"\b(slit\s+(my\s+wrists?|myself)|cut(ting)?\s+(my\s+wrists?|myself)|overdose\s+on\s+pills?|take\s+my\s+own\s+life)\b",
        r"\b(don'?t\s+want\s+to\s+live\s+anymore|no\s+reason\s+to\s+live|can'?t\s+go\s+on\s+living)\b",
        r"\b(self[\s-]harm|hurt(ing)?\s+myself)\b",
    ]

    FIXED_CRISIS_RESPONSE = (
        "It sounds like you are going through a very difficult time, and your safety and well-being come first. "
        "Please know that you are not alone and support is available right now:\n\n"
        "- **Suicide & Crisis Lifeline**: Call or text **988** (Available 24/7, free, and confidential in the US & Canada).\n"
        "- **Crisis Text Line**: Text **HOME to 741741** to connect with a crisis counselor.\n"
        "- **International Support**: If you are outside North America, please reach out to your local emergency services or visit https://findahelpline.com.\n\n"
        "Please speak with a counselor, trusted adult, or professional immediately."
    )

    PROMPT_INJECTION_PATTERNS = [
        r"\b(ignore|disregard|forget|override|bypass)\s+(all\s+)?(previous|prior|system|above)\s+(instructions|prompts|rules|commands)\b",
        r"\b(you\s+are\s+now\s+(in\s+)?|act\s+as\s+(an?\s+)?|pretend\s+to\s+be\s+(an?\s+)?)(DAN|unrestricted|jailbroken|developer\s+mode|evil\s+ai|jailbreak)\b",
        r"\b(DAN\s+mode|jailbreak|jailbroken|system\s*override|disable\s+safety|developer\s+mode)\b",
        r"\b(reveal|show|print|output|display)\s+(your\s+)?(system\s+prompt|initial\s+instructions|system\s+directives|hidden\s+prompt)\b",
        r"\[\s*system\s*\]|\{\s*system\s*\}|<\|im_start\|>system|<system>",
    ]

    ACADEMIC_INTEGRITY_PATTERNS = [
        r"\b(write|do|solve|complete)\s+(my\s+)?(entire|whole|all\s+of\s+my\s+)?.*?\b(homework|assignment|exam|quiz|test|take[\s-]home|essay|paper)\b",
        r"\b(give|tell)\s+me\s+(the\s+)?(direct\s+)?(answers?|solutions?)\s+(to|for)\s+(my\s+)?(quiz|exam|test|assignment|homework)\b",
        r"\b(write\s+(my|an?)\s+(full\s+)?essay\s+for\s+me|do\s+this\s+test\s+for\s+me)\b",
        r"\b(just\s+give\s+me\s+the\s+answers?|solve\s+this\s+for\s+me\s+so\s+i\s+can\s+submit)\b",
    ]

    TOXICITY_PATTERNS = [
        r"\b(make|build|create|manufacture)\s+(a\s+)?(bomb|explosive|weapon|poison|toxin)\b",
        r"\b(how\s+to\s+)?(hack|exploit|ddos|sql\s+inject|bypass\s+security\s+of)\s+([a-zA-Z0-9_\-\.]+)\b",
        r"\b(stolen\s+credit\s+card|ssn\s+dump|passwords?:\s*\S+)\b",
        r"\b(kill|murder|attack|stab|shoot)\s+(someone|people|him|her|them)\b",
    ]

    PROMPT_LEAK_PATTERNS = [
        r"SYSTEM\s+PROMPT:",
        r"PEDAGOGICAL\s+INSTRUCTION",
        r"CURRENT\s+PEDAGOGY\s+STATE:",
        r"OUTPUT\s+FORMAT:\s*\{",
        r"<system>",
        r"\[INSTRUCTION\]"
    ]

    GROUNDING_DISCLAIMER = (
        "Note: Please verify this against your course syllabus, as some details may exceed the retrieved lecture notes."
    )

    def __init__(
        self,
        grounding_threshold: float = 0.25,
        enforce_strict_grounding: bool = False,
        truthfulness_checker: Optional[TruthfulnessChecker] = None,
        enforce_truthfulness: bool = True,
    ):
        self.grounding_threshold = grounding_threshold
        self.enforce_strict_grounding = enforce_strict_grounding
        self.truthfulness_checker = truthfulness_checker or TruthfulnessChecker()
        self.enforce_truthfulness = enforce_truthfulness

    # -----------------------------------------------------------------------
    # STAGE 1: check_input(message)
    # -----------------------------------------------------------------------
    def check_input(self, message: str) -> InputGuardrailResult:
        """
        Stage 1: Evaluates user input against safety, security, and pedagogy policies.

        Returns:
            InputGuardrailResult with {blocked: bool, reason: Optional[str], redirect_mode: Optional[str]}
        """
        if not message:
            return InputGuardrailResult(blocked=False, reason=None, redirect_mode=None)

        msg_clean = message.strip()

        # 1. Crisis-Signal Detector (Short-circuits straight to fixed supportive response)
        for pattern in self.CRISIS_PATTERNS:
            if re.search(pattern, msg_clean, re.IGNORECASE):
                logger.warning("[GUARDRAIL INPUT] Crisis signal detected. Short-circuiting to supportive response.")
                return InputGuardrailResult(
                    blocked=True,
                    reason="Crisis signal detected: immediate supportive intervention required.",
                    redirect_mode=None,
                    supportive_response=self.FIXED_CRISIS_RESPONSE,
                    short_circuited=True,
                )

        # 2. Prompt-Injection Pattern & Classifier Check
        for pattern in self.PROMPT_INJECTION_PATTERNS:
            if re.search(pattern, msg_clean, re.IGNORECASE):
                logger.warning(f"[GUARDRAIL INPUT] Prompt injection pattern matched: {pattern}")
                return InputGuardrailResult(
                    blocked=True,
                    reason="Prompt injection or system instruction override attempt detected.",
                    redirect_mode=None,
                    supportive_response=None,
                    short_circuited=False,
                )

        # 3. Toxicity & Harm Filter
        for pattern in self.TOXICITY_PATTERNS:
            if re.search(pattern, msg_clean, re.IGNORECASE):
                logger.warning(f"[GUARDRAIL INPUT] Toxic or dangerous content matched: {pattern}")
                return InputGuardrailResult(
                    blocked=True,
                    reason="Harmful, toxic, or dangerous content detected.",
                    redirect_mode=None,
                    supportive_response=None,
                    short_circuited=False,
                )

        # 4. Academic-Integrity Classifier (returns redirect_mode='socratic' rather than blocking outright)
        for pattern in self.ACADEMIC_INTEGRITY_PATTERNS:
            if re.search(pattern, msg_clean, re.IGNORECASE):
                logger.info(f"[GUARDRAIL INPUT] Academic integrity flagged: {pattern}. Redirecting to Socratic mode.")
                return InputGuardrailResult(
                    blocked=False,
                    reason="Academic integrity: direct assignment/exam completion requested. Redirecting to Socratic scaffolding.",
                    redirect_mode="socratic",
                    supportive_response=None,
                    short_circuited=False,
                )

        # Clean input
        return InputGuardrailResult(
            blocked=False,
            reason=None,
            redirect_mode=None,
            supportive_response=None,
            short_circuited=False,
        )

    # -----------------------------------------------------------------------
    # STAGE 2: check_output(response, sources, expected_schema, student_message)
    # -----------------------------------------------------------------------
    def check_output(
        self,
        response: str,
        sources: Optional[List[Any]] = None,
        expected_schema: Optional[Any] = None,
        student_message: Optional[str] = None,
    ) -> OutputGuardrailResult:
        """
        Stage 2: Evaluates model output for prompt leaks, schema conformance,
        truthfulness (math derivations and factual claims), and grounding.

        Returns:
            OutputGuardrailResult with {blocked: bool, reason: Optional[str], sanitized_response: str, is_truthful: bool}
        """
        if not response:
            return OutputGuardrailResult(
                blocked=False,
                reason=None,
                sanitized_response="",
                is_grounded=True,
                schema_valid=True,
                is_truthful=True,
            )

        sanitized = response

        # 1. System-Prompt-Leak Regex & Classifier Check
        leak_detected = False
        for pat in self.PROMPT_LEAK_PATTERNS:
            if re.search(pat, sanitized, re.IGNORECASE):
                leak_detected = True
                sanitized = re.split(pat, sanitized, flags=re.IGNORECASE)[0].strip()

        # 2. JSON Schema Validation with One Repair Retry on Failure
        schema_valid = True
        repair_attempted = False

        if expected_schema is not None:
            valid, repaired_text, err, was_repaired = self._validate_and_repair_json(sanitized, expected_schema)
            if not valid:
                logger.warning(f"[GUARDRAIL OUTPUT] JSON schema validation failed: {err}")
                return OutputGuardrailResult(
                    blocked=True,
                    reason=f"JSON schema validation failed after 1 repair retry: {err}",
                    sanitized_response=sanitized,
                    is_grounded=True,
                    schema_valid=False,
                    repair_attempted=was_repaired,
                    is_truthful=True,
                )
            sanitized = repaired_text
            repair_attempted = was_repaired

        # 3. Truthfulness Checker Stage (Math derivations & Factual claims verification)
        is_truthful = True
        truth_reason = None
        regeneration_instruction = None

        if self.enforce_truthfulness and student_message:
            truth_res = self.truthfulness_checker.check_truthfulness(
                student_message=student_message,
                draft_response=sanitized,
                sources=sources,
            )
            if truth_res.rejected:
                is_truthful = False
                truth_reason = truth_res.reason
                regeneration_instruction = truth_res.regeneration_instruction
                logger.warning(f"[GUARDRAIL OUTPUT] Truthfulness check rejected draft response: {truth_res.reason}")

                # Reject response and supply specific error instruction
                return OutputGuardrailResult(
                    blocked=True,
                    reason=f"Truthfulness check rejected draft: {truth_res.reason}",
                    sanitized_response=truth_res.corrected_response_fallback or sanitized,
                    is_grounded=True,
                    schema_valid=schema_valid,
                    repair_attempted=repair_attempted,
                    is_truthful=False,
                    regeneration_instruction=truth_res.regeneration_instruction,
                    truth_violation_reason=truth_res.reason,
                )

        # 4. Grounding Check against Provided Sources
        is_grounded = True
        grounding_score = 1.0
        grounding_reason = None

        if sources:
            is_grounded, grounding_score = self._check_grounding(sanitized, sources)
            if not is_grounded:
                grounding_reason = f"Low grounding score ({grounding_score:.2f} < {self.grounding_threshold})"
                logger.warning(f"[GUARDRAIL OUTPUT] {grounding_reason}")
                if self.enforce_strict_grounding:
                    sanitized = f"{sanitized}\n\n*({self.GROUNDING_DISCLAIMER})*"

        # Compile final output reason
        reasons = []
        if leak_detected:
            reasons.append("System prompt leak scrubbed")
        if repair_attempted:
            reasons.append("JSON schema repaired on retry")
        if truth_reason:
            reasons.append(truth_reason)
        if grounding_reason:
            reasons.append(grounding_reason)

        return OutputGuardrailResult(
            blocked=False,
            reason="; ".join(reasons) if reasons else None,
            sanitized_response=sanitized.strip(),
            is_grounded=is_grounded,
            grounding_score=grounding_score,
            schema_valid=schema_valid,
            repair_attempted=repair_attempted,
            is_truthful=is_truthful,
            regeneration_instruction=regeneration_instruction,
            truth_violation_reason=truth_reason,
        )

    # -----------------------------------------------------------------------
    # Helper Methods
    # -----------------------------------------------------------------------
    def _validate_and_repair_json(
        self,
        text: str,
        expected_schema: Any
    ) -> Tuple[bool, str, Optional[str], bool]:
        """
        Validates JSON text against schema with exactly one repair retry.
        Returns (is_valid, repaired_or_original_text, error_message, was_repaired).
        """
        err1_msg = ""
        # Attempt 1: Validate original text directly
        try:
            self._execute_schema_check(text.strip(), expected_schema)
            return True, text.strip(), None, False
        except Exception as e1:
            err1_msg = str(e1)
            logger.debug(f"[GUARDRAIL] Attempt 1 JSON validation failed: {e1}. Attempting 1 repair retry...")

        # Attempt 2 (One repair retry)
        try:
            repaired = self._repair_json(text)
            self._execute_schema_check(repaired, expected_schema)
            return True, repaired, None, True
        except Exception as e2:
            return False, text, f"Initial parse: {err1_msg}; Repair retry failed: {e2}", True

    def _execute_schema_check(self, text: str, schema: Any) -> Any:
        """Helper to validate text against Pydantic model, JSON schema, or JSON validity."""
        data = json.loads(text)
        if isinstance(schema, type) and issubclass(schema, BaseModel):
            return schema.model_validate(data)
        elif callable(schema):
            return schema(data)
        elif isinstance(schema, dict):
            # Check required keys if provided in schema dict
            required_keys = schema.get("required", [])
            for k in required_keys:
                if k not in data:
                    raise ValueError(f"Missing required key: '{k}'")
        return data

    def _repair_json(self, text: str) -> str:
        """Deterministic heuristic repair for common LLM JSON formatting issues."""
        cleaned = text.strip()

        # 1. Extract markdown code fence if wrapped
        fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned)
        if fence_match:
            cleaned = fence_match.group(1).strip()
        else:
            # Extract from first { or [ to last } or ]
            start_brace = cleaned.find("{")
            start_bracket = cleaned.find("[")
            if start_brace != -1 and (start_bracket == -1 or start_brace < start_bracket):
                end_brace = cleaned.rfind("}")
                if end_brace != -1:
                    cleaned = cleaned[start_brace:end_brace + 1]
                else:
                    cleaned = cleaned[start_brace:] + "}"
            elif start_bracket != -1:
                end_bracket = cleaned.rfind("]")
                if end_bracket != -1:
                    cleaned = cleaned[start_bracket:end_bracket + 1]
                else:
                    cleaned = cleaned[start_bracket:] + "]"

        # 2. Strip trailing commas before closing braces/brackets
        cleaned = re.sub(r",\s*([\}\]])", r"\1", cleaned)

        # 3. Replace single quotes around keys/values with double quotes
        cleaned = re.sub(r"(?<=[{\s,])'([^']+)'\s*:", r'"\1":', cleaned)
        cleaned = re.sub(r":\s*'([^']*)'(?=[,\s}\]])", r': "\1"', cleaned)

        # 4. Balance missing closing braces/brackets
        open_braces = cleaned.count("{") - cleaned.count("}")
        if open_braces > 0:
            cleaned += "}" * open_braces
        open_brackets = cleaned.count("[") - cleaned.count("]")
        if open_brackets > 0:
            cleaned += "]" * open_brackets

        return cleaned

    def _check_grounding(
        self,
        answer: str,
        sources: List[Any],
    ) -> Tuple[bool, float]:
        """Calculates token overlap between answer and sources."""
        source_texts = []
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
            return True, 1.0

        corpus = " ".join(source_texts).lower()
        source_tokens = set(re.findall(r"\b[a-zA-Z]{3,}\b", corpus))

        answer_tokens = set(re.findall(r"\b[a-zA-Z]{3,}\b", answer.lower()))
        stopwords = {
            "the", "and", "that", "this", "with", "from", "for", "are", "was",
            "were", "will", "what", "how", "why", "can", "you", "your", "think",
            "about", "let", "step", "explain", "here", "look", "which", "into"
        }
        substantive_tokens = answer_tokens - stopwords

        if not substantive_tokens:
            return True, 1.0

        overlap = len(substantive_tokens.intersection(source_tokens))
        score = overlap / len(substantive_tokens)
        is_grounded = score >= self.grounding_threshold

        return is_grounded, round(score, 3)


# ---------------------------------------------------------------------------
# Legacy ResponseGuardrail Adapter (Maintains Full Backwards Compatibility)
# ---------------------------------------------------------------------------

class ResponseGuardrail:
    """
    Production Guardrail Layer for AI Tutor:
    1. Unsafe & Toxic Content Check
    2. Prompt Leak & System Instruction Scrubbing
    3. Hallucination & Grounding Check against RAG sources[]
    4. Resilient Fallback for Model Failures or Timeouts
    """

    FALLBACK_MESSAGES = {
        "TIMEOUT_OR_ERROR": (
            "I encountered a temporary connection issue while formulating your response. "
            "Please try asking your question again in a moment!"
        ),
        "UNSAFE_CONTENT": (
            "I am designed to assist only with educational and academic inquiries in a safe environment. "
            "Let's redirect our focus back to the course topics."
        ),
        "HALLUCINATION_WARNING": (
            "Note: Please verify this against your course syllabus, as some details may exceed the retrieved lecture notes."
        )
    }

    def __init__(
        self,
        grounding_consistency_threshold: float = 0.25,
        enforce_strict_grounding: bool = False
    ):
        self.pipeline = GuardrailPipeline(
            grounding_threshold=grounding_consistency_threshold,
            enforce_strict_grounding=enforce_strict_grounding
        )
        self.grounding_threshold = grounding_consistency_threshold
        self.enforce_strict_grounding = enforce_strict_grounding

    def check_safety(self, text: str) -> Tuple[bool, Optional[str]]:
        input_res = self.pipeline.check_input(text)
        if input_res.blocked:
            return False, input_res.reason
        return True, None

    def scrub_prompt_leaks(self, text: str) -> str:
        out_res = self.pipeline.check_output(text)
        return out_res.sanitized_response

    def check_rag_grounding(
        self,
        answer: str,
        sources: List[SourceCitation],
        chunks: Optional[List[Chunk]] = None
    ) -> Tuple[bool, float]:
        combined_sources = list(sources or []) + list(chunks or [])
        return self.pipeline._check_grounding(answer, combined_sources)

    def validate_and_sanitize(
        self,
        raw_answer: str,
        pedagogy_state: PedagogyState,
        request: AIChatRequest,
        sources: Optional[List[SourceCitation]] = None,
        chunks: Optional[List[Chunk]] = None
    ) -> GuardrailResult:
        flags = []

        # 1. Safety check on raw answer
        is_safe, reason = self.check_safety(raw_answer)
        if not is_safe:
            logger.warning(f"[GUARDRAIL TRIGGERED] Safety check failed: {reason}")
            return GuardrailResult(
                sanitized_answer=self.FALLBACK_MESSAGES["UNSAFE_CONTENT"],
                is_safe=False,
                flags=[f"UNSAFE_CONTENT: {reason}"]
            )

        # 2. Check output via pipeline (includes TruthfulnessChecker against student request)
        combined_sources = list(sources or []) + list(chunks or [])
        student_msg = request.message if request else None
        out_res = self.pipeline.check_output(
            raw_answer,
            sources=combined_sources,
            student_message=student_msg
        )
        sanitized = out_res.sanitized_response

        if out_res.reason and "leak" in out_res.reason.lower():
            flags.append("PROMPT_LEAK_SCRUBBED")

        if not out_res.is_grounded:
            flags.append(f"LOW_GROUNDING_SCORE: {out_res.grounding_score}")

        if not out_res.is_truthful:
            flags.append("UNTRUTHFUL_AFFIRMATION_REJECTED")

        if pedagogy_state.pedagogy_mode == PedagogyMode.OFF_TOPIC and len(sanitized.strip()) < 5:
            sanitized = "Let's focus on our course learning goals! What topic in the lecture would you like to explore?"

        if not sanitized or len(sanitized.strip()) < 2:
            sanitized = "Let's think through this step by step. What part of the concept feels most challenging?"
            flags.append("EMPTY_RESPONSE_FALLBACK")

        return GuardrailResult(
            sanitized_answer=sanitized.strip(),
            is_safe=True,
            is_grounded=out_res.is_grounded,
            grounding_score=out_res.grounding_score,
            flags=flags
        )

    def get_fallback_response(
        self,
        error: Optional[Exception] = None,
        request: Optional[AIChatRequest] = None
    ) -> str:
        logger.error(f"[MODEL CALL FAILED] Returning fallback message. Error: {error}")
        return self.FALLBACK_MESSAGES["TIMEOUT_OR_ERROR"]
