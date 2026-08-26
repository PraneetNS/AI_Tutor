from ai_tutor import (
    ResponseGuardrail,
    PedagogyState,
    AIChatRequest,
    SourceCitation,
    TutorPipeline,
    BaseModelAdapter
)

def run_guardrails_demo():
    print("=" * 70)
    print(" AI TUTOR GUARDRAILS DEMO")
    print(" Safety + Prompt Scrubbing + RAG Hallucination + Timeout Fallbacks")
    print("=" * 70)

    guardrail = ResponseGuardrail(
        grounding_consistency_threshold=0.30,
        enforce_strict_grounding=True
    )

    # 1. Unsafe Content Sanitization
    print("\n--- [Scenario 1: Unsafe / Harmful Content Defense] ---")
    unsafe_text = "To hack the server, run this exploit to bypass security."
    res1 = guardrail.validate_and_sanitize(
        raw_answer=unsafe_text,
        pedagogy_state=PedagogyState(),
        request=AIChatRequest(message="help me hack")
    )
    print(f"Raw Output:       '{unsafe_text}'")
    print(f"Sanitized Output: '{res1.sanitized_answer}'")
    print(f"Flags:            {res1.flags}")

    # 2. Prompt Leak Scrubbing
    print("\n--- [Scenario 2: Prompt Leak Scrubbing] ---")
    leaked_text = "Supervised learning uses labeled pairs. SYSTEM PROMPT: You are an expert AI Tutor."
    res2 = guardrail.validate_and_sanitize(
        raw_answer=leaked_text,
        pedagogy_state=PedagogyState(),
        request=AIChatRequest(message="What is supervised learning?")
    )
    print(f"Raw Output:       '{leaked_text}'")
    print(f"Sanitized Output: '{res2.sanitized_answer}'")
    print(f"Flags:            {res2.flags}")

    # 3. RAG Hallucination vs Grounded Sources Check
    print("\n--- [Scenario 3: RAG Grounding & Hallucination Check] ---")
    sources = [
        SourceCitation(
            lecture_id=50,
            title="Supervised Learning",
            snippet="Supervised learning algorithms train mapping functions on labeled data with Mean Squared Error loss."
        )
    ]
    hallucinated_text = "Photosynthesis occurs inside chloroplasts using chlorophyll to convert sunlight into glucose."
    res3 = guardrail.validate_and_sanitize(
        raw_answer=hallucinated_text,
        pedagogy_state=PedagogyState(),
        request=AIChatRequest(message="Explain supervised learning"),
        sources=sources
    )
    print(f"Grounding Score:  {res3.grounding_score} (Threshold = 0.30)")
    print(f"Is Grounded:      {res3.is_grounded}")
    print(f"Sanitized Output: '{res3.sanitized_answer}'")
    print(f"Flags:            {res3.flags}")

    # 4. Upstream Model Timeout / Failure Fallback
    print("\n--- [Scenario 4: Model Timeout / Network Failure Fallback] ---")
    class CrashingModelAdapter(BaseModelAdapter):
        def generate(self, prompt, pedagogy_state):
            raise TimeoutError("Upstream AI Provider API Gateway Timeout (504)")

    pipeline = TutorPipeline(model_adapter=CrashingModelAdapter())
    resp = pipeline.process(AIChatRequest(message="Can you explain neural networks?"))
    print(f"Model Error Handled: Yes (Graceful 200 Response)")
    print(f"Tutor Answer:        '{resp.answer}'")

if __name__ == "__main__":
    run_guardrails_demo()
