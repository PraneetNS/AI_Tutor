import pytest
from starlette.testclient import TestClient

from ai_tutor import (
    create_app,
    TutorPipeline,
    AIChatRequest,
    AIChatResponse,
    BaseRouter,
    BasePedagogyEngine,
    BasePromptOrchestrator,
    BaseModelAdapter,
    BaseGuardrails,
    DefaultGuardrails,
    GuardrailResult,
    DefaultPromptOrchestrator,
    DefaultPedagogyEngine,
    DefaultRouter,
    MockKnowledgeSource,
    MockLLMClient,
    DefaultModelAdapter,
    ClassificationResult,
    IntentLabel,
    PedagogyMode,
    PedagogyState,
    ChatMessage,
    Role
)


def test_pipeline_concept_with_course_id_triggers_knowledge_source():
    ks = MockKnowledgeSource()
    model_adapter = DefaultModelAdapter(llm_client=MockLLMClient())
    pipeline = TutorPipeline(
        knowledge_source=ks,
        model_adapter=model_adapter
    )

    req = AIChatRequest(
        message="Why does backpropagation calculate gradients backwards?",
        course_id=101,
        lecture_id=50
    )

    resp = pipeline.process(req)
    assert isinstance(resp, AIChatResponse)
    assert resp.pedagogy_mode == PedagogyMode.SOCRATIC
    assert resp.knowledge_source_used == "MockKnowledgeSource"
    assert len(resp.sources) > 0
    assert resp.sources[0].lecture_id == 50


def test_pipeline_off_topic_skips_knowledge_source():
    ks = MockKnowledgeSource()
    model_adapter = DefaultModelAdapter(llm_client=MockLLMClient())
    pipeline = TutorPipeline(
        knowledge_source=ks,
        model_adapter=model_adapter
    )

    req = AIChatRequest(
        message="What is the weather in Tokyo?",
        course_id=101
    )

    resp = pipeline.process(req)
    assert resp.pedagogy_mode == PedagogyMode.OFF_TOPIC
    # Off-topic should NOT invoke knowledge retrieval
    assert resp.knowledge_source_used is None
    assert len(resp.sources) == 0


def test_stage_mockability_and_swappability():
    # Custom Mock Router that always returns FACTUAL
    class AlwaysFactualRouter(BaseRouter):
        def route(self, request, history):
            return ClassificationResult(label=IntentLabel.FACTUAL, confidence=0.99)

    # Custom Mock Guardrails that appends a signature
    class WatermarkGuardrails(BaseGuardrails):
        def validate_and_sanitize(self, raw_answer, pedagogy_state, request, sources=None, chunks=None):
            return GuardrailResult(sanitized_answer=f"{raw_answer} [VERIFIED_SAFE]")

        def get_fallback_response(self, error=None, request=None):
            return "Fallback [VERIFIED_SAFE]"

    pipeline = TutorPipeline(
        router=AlwaysFactualRouter(),
        guardrails=WatermarkGuardrails(),
        model_adapter=DefaultModelAdapter(llm_client=MockLLMClient())
    )

    req = AIChatRequest(message="Tell me anything")
    resp = pipeline.process(req)

    assert resp.pedagogy_mode == PedagogyMode.DIRECT
    assert resp.answer.endswith("[VERIFIED_SAFE]")


def test_fastapi_chat_endpoint():
    pipeline = TutorPipeline(
        knowledge_source=MockKnowledgeSource(),
        model_adapter=DefaultModelAdapter(llm_client=MockLLMClient())
    )
    app = create_app(pipeline=pipeline)
    client = TestClient(app)

    # 1. External contract compliant request
    payload = {
        "message": "Explain supervised learning",
        "course_id": 101,
        "lecture_id": 50
    }

    response = client.post("/api/ai/chat", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert "answer" in data
    assert "session_id" in data
    assert "sources" in data
    assert len(data["sources"]) > 0
    assert data["sources"][0]["lecture_id"] == 50
