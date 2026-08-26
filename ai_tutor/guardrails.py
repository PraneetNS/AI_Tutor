import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from .models import (
    PedagogyState,
    PedagogyMode,
    AIChatRequest,
    Chunk,
    SourceCitation
)

logger = logging.getLogger("ai_tutor.guardrails")


class GuardrailResult:
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


class ResponseGuardrail:
    """
    Production Guardrail Layer for AI Tutor:
    1. Unsafe & Toxic Content Check
    2. Prompt Leak & System Instruction Scrubbing
    3. Hallucination & Grounding Check against RAG sources[]
    4. Resilient Fallback for Model Failures or Timeouts
    """

    UNSAFE_PATTERNS = [
        r"\b(harm\s+yourself|suicide|kill\s+yourself)\b",
        r"\b(make\s+a\s+bomb|build\s+a\s+weapon|hack|exploit|malware|bypass\s+security)\b",
        r"\b(credit\s*card\s*number|social\s*security\s*number|passwords?:\s*\S+)\b"
    ]

    PROMPT_LEAK_PATTERNS = [
        r"SYSTEM\s+PROMPT:",
        r"PEDAGOGICAL\s+INSTRUCTION",
        r"CURRENT\s+PEDAGOGY\s+STATE:",
        r"OUTPUT\s+FORMAT:\s*\{"
    ]

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
        self.grounding_threshold = grounding_consistency_threshold
        self.enforce_strict_grounding = enforce_strict_grounding

    def check_safety(self, text: str) -> Tuple[bool, Optional[str]]:
        """Detect unsafe, toxic, or privacy-violating content."""
        clean = text.lower()
        for pat in self.UNSAFE_PATTERNS:
            if re.search(pat, clean):
                return False, f"Unsafe pattern match: {pat}"
        return True, None

    def scrub_prompt_leaks(self, text: str) -> str:
        """Strip internal prompt markers or system instructions."""
        sanitized = text
        for pat in self.PROMPT_LEAK_PATTERNS:
            if re.search(pat, sanitized, re.IGNORECASE):
                sanitized = re.split(pat, sanitized, flags=re.IGNORECASE)[0].strip()
        return sanitized

    def check_rag_grounding(
        self,
        answer: str,
        sources: List[SourceCitation],
        chunks: Optional[List[Chunk]] = None
    ) -> Tuple[bool, float]:
        """
        Verify that the answer is grounded in the retrieved sources/chunks when RAG is used.
        Extracts key informative tokens from the answer and computes overlap with source text.
        """
        if not sources and not chunks:
            # RAG was not used, grounding check is not applicable
            return True, 1.0

        # Combine all source texts
        source_texts = []
        if sources:
            source_texts.extend([s.snippet or s.title for s in sources])
        if chunks:
            source_texts.extend([c.content for c in chunks])

        corpus = " ".join(source_texts).lower()
        source_tokens = set(re.findall(r"\b[a-zA-Z]{3,}\b", corpus))

        # Extract answer substantive tokens (ignoring short stopwords)
        answer_tokens = set(re.findall(r"\b[a-zA-Z]{3,}\b", answer.lower()))
        stopwords = {
            "the", "and", "that", "this", "with", "from", "for", "are", "was",
            "were", "will", "what", "how", "why", "can", "you", "your", "think",
            "about", "let", "step", "explain", "here", "look"
        }
        substantive_tokens = answer_tokens - stopwords

        if not substantive_tokens:
            return True, 1.0

        overlap = len(substantive_tokens.intersection(source_tokens))
        score = overlap / len(substantive_tokens)

        is_grounded = score >= self.grounding_threshold
        return is_grounded, round(score, 3)

    def validate_and_sanitize(
        self,
        raw_answer: str,
        pedagogy_state: PedagogyState,
        request: AIChatRequest,
        sources: Optional[List[SourceCitation]] = None,
        chunks: Optional[List[Chunk]] = None
    ) -> GuardrailResult:
        flags = []

        # 1. Check Safety
        is_safe, reason = self.check_safety(raw_answer)
        if not is_safe:
            logger.warning(f"[GUARDRAIL TRIGGERED] Safety check failed: {reason}")
            return GuardrailResult(
                sanitized_answer=self.FALLBACK_MESSAGES["UNSAFE_CONTENT"],
                is_safe=False,
                flags=[f"UNSAFE_CONTENT: {reason}"]
            )

        # 2. Scrub Prompt Leaks
        sanitized = self.scrub_prompt_leaks(raw_answer)
        if sanitized != raw_answer:
            flags.append("PROMPT_LEAK_SCRUBBED")

        # 3. Hallucination / Grounding Check when RAG is active
        sources = sources or []
        is_grounded, grounding_score = self.check_rag_grounding(sanitized, sources, chunks)

        if not is_grounded:
            logger.warning(
                f"[GUARDRAIL TRIGGERED] Low Grounding Score ({grounding_score:.2f} < {self.grounding_threshold})"
            )
            flags.append(f"LOW_GROUNDING_SCORE: {grounding_score}")

            if self.enforce_strict_grounding:
                # Add grounding disclaimer to the student
                sanitized = (
                    f"{sanitized}\n\n*({self.FALLBACK_MESSAGES['HALLUCINATION_WARNING']})*"
                )

        # 4. Off-Topic Mode Sanity Check
        if pedagogy_state.pedagogy_mode == PedagogyMode.OFF_TOPIC:
            if len(sanitized.strip()) < 5:
                sanitized = "Let's focus on our course learning goals! What topic in the lecture would you like to explore?"

        # 5. Empty / Truncated fallback
        if not sanitized or len(sanitized.strip()) < 2:
            sanitized = "Let's think through this step by step. What part of the concept feels most challenging?"
            flags.append("EMPTY_RESPONSE_FALLBACK")

        return GuardrailResult(
            sanitized_answer=sanitized.strip(),
            is_safe=True,
            is_grounded=is_grounded,
            grounding_score=grounding_score,
            flags=flags
        )

    def get_fallback_response(
        self,
        error: Optional[Exception] = None,
        request: Optional[AIChatRequest] = None
    ) -> str:
        """Fallback message when model times out, crashes, or fails to respond."""
        logger.error(f"[MODEL CALL FAILED] Returning fallback message. Error: {error}")
        return self.FALLBACK_MESSAGES["TIMEOUT_OR_ERROR"]
