import pytest
import logging
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from ai_tutor.models import (
    AIChatRequest,
    AIChatResponse,
    ChatMessage,
    Role,
    PedagogyMode,
)
from ai_tutor.pipeline import (
    TutorPipeline,
    SAFE_IN_CHARACTER_FALLBACK,
    BaseRouter,
    BasePedagogyEngine,
    BasePromptOrchestrator,
    BaseModelAdapter,
    BaseGuardrails,
    BaseSessionStore,
)
from ai_tutor.knowledge_source import KnowledgeSource
from ai_tutor.api import create_app
from ai_tutor.config import Settings, reload_settings
from ai_tutor.model_gateway import (
    GPTAdapter,
    ClaudeAdapter,
    GeminiAdapter,
    QwenProvider,
)
from ai_tutor.learner_store import PostgresRedisLearnerStateStore
from ai_tutor.event_bus import PostgresEventBus
from ai_tutor.strategy_engine import PostgresStrategyEffectivenessStore


# ---------------------------------------------------------------------------
# Pipeline Stage Failure Tests
# ---------------------------------------------------------------------------

def test_pipeline_router_failure_returns_fallback(caplog):
    """If the router raises an unexpected exception, return in-character fallback."""
    failing_router = MagicMock(spec=BaseRouter)
    failing_router.route.side_effect = RuntimeError("Intent classifier crash!")

    pipeline = TutorPipeline(router=failing_router)
    req = AIChatRequest(
        message="Explain neural networks",
        student_id="student_123",
        session_id="sess_router_fail"
    )

    with caplog.at_level(logging.ERROR):
        res = pipeline.process(req, request_id="req_test_001")

    assert isinstance(res, AIChatResponse)
    assert res.answer == SAFE_IN_CHARACTER_FALLBACK
    assert res.session_id == "sess_router_fail"

    # Verify structured error logging
    log_text = caplog.text
    assert "[PIPELINE UNHANDLED EXCEPTION]" in log_text
    assert "req_test_001" in log_text
    assert "student_123" in log_text
    assert "sess_router_fail" in log_text
    assert "failed_step=router" in log_text


def test_pipeline_pedagogy_engine_failure_returns_fallback(caplog):
    """If the pedagogy engine raises an unexpected exception, return in-character fallback."""
    failing_pedagogy = MagicMock(spec=BasePedagogyEngine)
    failing_pedagogy.evaluate.side_effect = ValueError("Corrupt pedagogy state matrix")

    pipeline = TutorPipeline(pedagogy_engine=failing_pedagogy)
    req = AIChatRequest(
        message="What is gradient descent?",
        student_id="student_456",
        session_id="sess_ped_fail"
    )

    with caplog.at_level(logging.ERROR):
        res = pipeline.process(req, request_id="req_test_002")

    assert res.answer == SAFE_IN_CHARACTER_FALLBACK
    assert "failed_step=pedagogy_engine" in caplog.text
    assert "req_test_002" in caplog.text


def test_pipeline_prompt_orchestrator_failure_returns_fallback(caplog):
    """If prompt builder fails, return safe fallback."""
    failing_orchestrator = MagicMock(spec=BasePromptOrchestrator)
    failing_orchestrator.build_prompt.side_effect = IndexError("Prompt template out of bounds")

    pipeline = TutorPipeline(prompt_orchestrator=failing_orchestrator)
    req = AIChatRequest(
        message="Help me with backprop",
        student_id="student_789",
        session_id="sess_prompt_fail"
    )

    with caplog.at_level(logging.ERROR):
        res = pipeline.process(req, request_id="req_test_003")

    assert res.answer == SAFE_IN_CHARACTER_FALLBACK
    assert "failed_step=prompt_orchestrator" in caplog.text


def test_pipeline_guardrail_failure_returns_fallback(caplog):
    """If guardrails fail with unhandled exception, return safe fallback."""
    failing_guardrails = MagicMock(spec=BaseGuardrails)
    failing_guardrails.validate_and_sanitize.side_effect = RuntimeError("Regex engine crash in safety check")

    pipeline = TutorPipeline(guardrails=failing_guardrails)
    req = AIChatRequest(
        message="How does attention work?",
        student_id="student_999",
        session_id="sess_guardrail_fail"
    )

    with caplog.at_level(logging.ERROR):
        res = pipeline.process(req, request_id="req_test_004")

    assert res.answer == SAFE_IN_CHARACTER_FALLBACK
    assert "failed_step=guardrails" in caplog.text


# ---------------------------------------------------------------------------
# API Gateway End-to-End Tests
# ---------------------------------------------------------------------------

def test_api_chat_endpoint_graceful_recovery_on_internal_error():
    """Chat endpoint should always return 200 with in-character fallback on pipeline failure."""
    failing_pipeline = MagicMock(spec=TutorPipeline)
    failing_pipeline.process.side_effect = Exception("Critical pipeline deadlock")

    settings = Settings(openai_api_key="mock-key")
    app = create_app(pipeline=failing_pipeline, settings=settings)
    client = TestClient(app)

    response = client.post(
        "/api/ai/chat",
        headers={"X-Request-ID": "req_custom_123"},
        json={
            "message": "Explain eigenvalues",
            "student_id": "student_abc",
            "session_id": "sess_abc"
        }
    )

    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == SAFE_IN_CHARACTER_FALLBACK
    assert data["session_id"] == "sess_abc"
    assert response.headers.get("X-Request-ID") == "req_custom_123"


def test_api_global_exception_handler_sanitized_json(caplog):
    """Global exception handler returns clean JSON with request_id for non-chat routes."""
    settings = Settings(openai_api_key="mock-key")
    app = create_app(settings=settings)

    # Simulate an endpoint raising an unhandled exception
    @app.get("/api/test-crash")
    def crash_route():
        raise RuntimeError("Database pool connection exhaustion")

    client = TestClient(app, raise_server_exceptions=False)

    with caplog.at_level(logging.ERROR):
        response = client.get("/api/test-crash", headers={"X-Request-ID": "req_crash_456"})

    assert response.status_code == 500
    data = response.json()
    assert data["status"] == "error"
    assert data["message"] == SAFE_IN_CHARACTER_FALLBACK
    assert data["request_id"] == "req_crash_456"
    assert "Traceback" not in response.text
    assert "Database pool connection exhaustion" not in data["message"]


# ---------------------------------------------------------------------------
# Timeout Settings & Client Configuration Tests
# ---------------------------------------------------------------------------

def test_settings_timeout_defaults():
    """Settings should have default timeouts configured."""
    settings = Settings(openai_api_key="key")
    assert settings.model_request_timeout == 15.0
    assert settings.db_timeout == 5.0
    assert settings.redis_timeout == 3.0


def test_provider_adapters_accept_timeouts():
    """Provider adapters configure timeouts properly."""
    settings = Settings(
        openai_api_key="sk-openai-12345678901234567890",
        anthropic_api_key="sk-ant-api03-12345678901234567890",
        gemini_api_key="AIzaSy12345678901234567890",
        dashscope_api_key="sk-dash-12345678901234567890",
        model_request_timeout=25.0,
    )

    gpt = GPTAdapter(settings=settings)
    assert gpt.timeout == 25.0

    claude = ClaudeAdapter(settings=settings)
    assert claude.timeout == 25.0

    gemini = GeminiAdapter(settings=settings)
    assert gemini.timeout == 25.0

    qwen = QwenProvider(settings=settings)
    assert qwen.timeout == 25.0

    # Custom override
    custom_gpt = GPTAdapter(timeout=8.5, settings=settings)
    assert custom_gpt.timeout == 8.5


def test_learner_store_timeouts():
    """PostgresRedisLearnerStateStore stores configured DB & Redis timeouts."""
    store = PostgresRedisLearnerStateStore(
        postgres_dsn="postgresql://user:pass@localhost:5432/db",
        redis_url="redis://localhost:6379/0",
        db_timeout=6.0,
        redis_timeout=2.5,
        auto_migrate=False
    )
    assert store.db_timeout == 6.0
    assert store.redis_timeout == 2.5


def test_strategy_store_and_event_bus_timeouts():
    """Strategy store and event bus accept and configure db_timeout."""
    strat_store = PostgresStrategyEffectivenessStore(
        postgres_dsn="postgresql://user:pass@localhost:5432/db",
        db_timeout=7.5
    )
    assert strat_store.db_timeout == 7.5

    bus = PostgresEventBus(
        dsn="postgresql://user:pass@localhost:5432/db",
        db_timeout=4.0,
        auto_migrate=False
    )
    assert bus._db_timeout == 4.0
