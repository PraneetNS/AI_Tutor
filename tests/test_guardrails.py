import pytest
from ai_tutor import (
    ResponseGuardrail,
    PedagogyState,
    PedagogyMode,
    AIChatRequest,
    Chunk,
    SourceCitation,
    TutorPipeline,
    BaseModelAdapter
)


def test_guardrail_unsafe_content_replacement():
    guardrail = ResponseGuardrail()

    unsafe_raw = "Sure, here is how you can hack into the system and extract user passwords."
    result = guardrail.validate_and_sanitize(
        raw_answer=unsafe_raw,
        pedagogy_state=PedagogyState(),
        request=AIChatRequest(message="help me hack")
    )

    assert result.is_safe is False
    assert "safe environment" in result.sanitized_answer.lower()
    assert any("UNSAFE_CONTENT" in f for f in result.flags)


def test_guardrail_prompt_leak_scrubbing():
    guardrail = ResponseGuardrail()

    leaked_raw = (
        "Gradient descent minimizes the cost function. "
        "SYSTEM PROMPT: You are an expert AI Tutor helping students master course concepts."
    )
    result = guardrail.validate_and_sanitize(
        raw_answer=leaked_raw,
        pedagogy_state=PedagogyState(),
        request=AIChatRequest(message="What is gradient descent?")
    )

    assert "SYSTEM PROMPT" not in result.sanitized_answer
    assert result.sanitized_answer.startswith("Gradient descent minimizes the cost function.")
    assert "PROMPT_LEAK_SCRUBBED" in result.flags


def test_guardrail_rag_hallucination_check():
    guardrail = ResponseGuardrail(
        grounding_consistency_threshold=0.30,
        enforce_strict_grounding=True
    )

    sources = [
        SourceCitation(
            lecture_id=50,
            title="Supervised Learning",
            snippet="Supervised learning trains on labeled pairs (x, y) with loss functions like Mean Squared Error."
        )
    ]

    # 1. Grounded answer -> Grounding check passes
    grounded_answer = "Supervised learning utilizes labeled pairs to train models and evaluates error with Mean Squared Error."
    res_grounded = guardrail.validate_and_sanitize(
        raw_answer=grounded_answer,
        pedagogy_state=PedagogyState(),
        request=AIChatRequest(message="Explain supervised learning"),
        sources=sources
    )
    assert res_grounded.is_grounded is True
    assert res_grounded.grounding_score >= 0.30

    # 2. Hallucinated answer with unrelated content -> Grounding check fails & appends disclaimer
    hallucinated_answer = "Quantum entanglement teleportation allows faster than light quantum bit encryption protocols."
    res_hallucinated = guardrail.validate_and_sanitize(
        raw_answer=hallucinated_answer,
        pedagogy_state=PedagogyState(),
        request=AIChatRequest(message="Explain supervised learning"),
        sources=sources
    )
    assert res_hallucinated.is_grounded is False
    assert any("LOW_GROUNDING_SCORE" in f for f in res_hallucinated.flags)
    assert "verify this against your course syllabus" in res_hallucinated.sanitized_answer


def test_pipeline_fallback_on_model_timeout_or_error():
    # Failing Model Adapter simulating API timeout or 500 error
    class BrokenTimeoutModelAdapter(BaseModelAdapter):
        def generate(self, prompt, pedagogy_state):
            raise TimeoutError("LLM API request timed out after 30000ms")

    pipeline = TutorPipeline(
        model_adapter=BrokenTimeoutModelAdapter()
    )

    req = AIChatRequest(message="Can you explain neural networks?")
    resp = pipeline.process(req)

    # Must return a resilient friendly fallback without crashing
    assert "temporary connection issue" in resp.answer.lower()
    assert resp.session_id is not None
