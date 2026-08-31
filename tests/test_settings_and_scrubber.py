"""
tests/test_settings_and_scrubber.py
-----------------------------------
Unit tests for:
1. Typed Settings module (pydantic-settings) & zero-secret-default enforcement.
2. Fast-fail startup validation with clear missing-variable reporting.
3. ModelGateway and provider adapter Settings integration.
4. SecretScrubberFilter regex-based and dynamic API key redaction in logs and stack traces.
"""

import io
import logging
import os
import pytest
from unittest.mock import patch, MagicMock

from ai_tutor.config import (
    Settings,
    MissingApiKeyError,
    get_settings,
    reload_settings,
)
from ai_tutor.log_scrubber import (
    SecretScrubberFilter,
    install_log_scrubber,
    scrub_text,
    API_KEY_PATTERNS,
)
from ai_tutor.model_gateway import (
    ModelGateway,
    GPTAdapter,
    ClaudeAdapter,
    GeminiAdapter,
    QwenProvider,
    MockAdapter,
    _build_adapter,
)


@pytest.fixture(autouse=True)
def reset_settings_cache():
    """Ensure cached Settings is reset between tests."""
    reload_settings()
    yield
    reload_settings()


# ===========================================================================
# 1. Typed Settings Module Tests
# ===========================================================================

class TestSettingsModule:
    def test_settings_default_has_no_real_secrets(self):
        """Ensure no secrets contain hardcoded real values by default."""
        empty_env = {}
        with patch.dict(os.environ, empty_env, clear=True):
            settings = Settings()
            assert settings.openai_api_key is None
            assert settings.anthropic_api_key is None
            assert settings.gemini_api_key is None
            assert settings.google_api_key is None
            assert settings.dashscope_api_key is None
            assert settings.database_url is None
            assert settings.redis_url is None

    def test_settings_loads_from_environment_variables(self):
        env = {
            "OPENAI_API_KEY": "sk-test-openai-12345678901234567890",
            "ANTHROPIC_API_KEY": "sk-ant-test-anthropic-123456789012345",
            "GEMINI_API_KEY": "AIzaSyTestGeminiKey1234567890123456",
            "DASHSCOPE_API_KEY": "dashscope-secret-key-12345",
            "PRIMARY_PROVIDER": "gpt",
            "FALLBACK_PROVIDER": "claude",
            "PORT": "9000",
            "CORS_ORIGINS": "http://localhost:3000, http://example.com",
            "DEFAULT_HINT_BUDGET": "5",
        }
        with patch.dict(os.environ, env, clear=True):
            settings = Settings()
            assert settings.openai_api_key == "sk-test-openai-12345678901234567890"
            assert settings.anthropic_api_key == "sk-ant-test-anthropic-123456789012345"
            assert settings.gemini_api_key == "AIzaSyTestGeminiKey1234567890123456"
            assert settings.dashscope_api_key == "dashscope-secret-key-12345"
            assert settings.primary_provider == "gpt"
            assert settings.fallback_provider == "claude"
            assert settings.port == 9000
            assert settings.cors_origins == ["http://localhost:3000", "http://example.com"]
            assert settings.default_hint_budget == 5

    def test_get_api_key_for_mock_provider_returns_empty(self):
        settings = Settings()
        assert settings.get_api_key_for_provider("mock") == ""
        assert settings.get_api_key_for_provider("offline") == ""

    def test_get_api_key_for_known_providers(self):
        settings = Settings(
            openai_api_key="sk-openai-key",
            anthropic_api_key="sk-ant-claude-key",
            gemini_api_key="AIzaSyGeminiKey",
            dashscope_api_key="dashscope-key",
        )
        assert settings.get_api_key_for_provider("openai") == "sk-openai-key"
        assert settings.get_api_key_for_provider("gpt") == "sk-openai-key"
        assert settings.get_api_key_for_provider("claude") == "sk-ant-claude-key"
        assert settings.get_api_key_for_provider("anthropic") == "sk-ant-claude-key"
        assert settings.get_api_key_for_provider("gemini") == "AIzaSyGeminiKey"
        assert settings.get_api_key_for_provider("google") == "AIzaSyGeminiKey"
        assert settings.get_api_key_for_provider("qwen") == "dashscope-key"

    def test_get_api_key_missing_raises_missing_api_key_error(self):
        settings = Settings()
        with pytest.raises(MissingApiKeyError) as exc_info:
            settings.get_api_key_for_provider("openai")
        assert "OPENAI_API_KEY" in str(exc_info.value)
        assert "openai" in str(exc_info.value)

    def test_get_api_key_unknown_provider_raises_value_error(self):
        settings = Settings()
        with pytest.raises(ValueError, match="Unknown provider"):
            settings.get_api_key_for_provider("quantum_ai")

    def test_reload_settings(self):
        try:
            with patch.dict(os.environ, {"PRIMARY_PROVIDER": "mock"}):
                s1 = reload_settings()
                assert s1.primary_provider == "mock"
            with patch.dict(os.environ, {"PRIMARY_PROVIDER": "gpt"}):
                s2 = reload_settings()
                assert s2.primary_provider == "gpt"
        finally:
            with patch.dict(os.environ, {}, clear=True):
                reload_settings()


# ===========================================================================
# 2. Fast-Fail Startup Validation Tests
# ===========================================================================

class TestStartupValidation:
    def test_mock_provider_passes_startup_validation_without_keys(self):
        settings = Settings(primary_provider="mock", fallback_provider=None)
        # Should not raise
        settings.validate_startup()

    def test_startup_validation_fails_when_primary_missing_key(self):
        settings = Settings(primary_provider="openai", openai_api_key=None)
        with pytest.raises(ValueError) as exc_info:
            settings.validate_startup()
        msg = str(exc_info.value)
        assert "Application startup failed" in msg
        assert "OPENAI_API_KEY" in msg
        assert "openai" in msg

    def test_startup_validation_fails_when_fallback_missing_key(self):
        settings = Settings(
            primary_provider="mock",
            fallback_provider="anthropic",
            anthropic_api_key=None,
        )
        with pytest.raises(ValueError) as exc_info:
            settings.validate_startup()
        msg = str(exc_info.value)
        assert "ANTHROPIC_API_KEY" in msg
        assert "anthropic" in msg

    def test_startup_validation_require_all(self):
        settings = Settings(openai_api_key="sk-123")
        with pytest.raises(ValueError) as exc_info:
            settings.validate_startup(require_all=True)
        msg = str(exc_info.value)
        assert "ANTHROPIC_API_KEY" in msg
        assert "GEMINI_API_KEY" in msg
        assert "DASHSCOPE_API_KEY" in msg


# ===========================================================================
# 3. ModelGateway & Adapter Integration Tests
# ===========================================================================

class TestModelGatewayAdapterIntegration:
    def test_gpt_adapter_fails_fast_with_missing_key(self):
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(MissingApiKeyError) as exc_info:
                GPTAdapter()
            assert "OPENAI_API_KEY" in str(exc_info.value)

    def test_claude_adapter_fails_fast_with_missing_key(self):
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(MissingApiKeyError) as exc_info:
                ClaudeAdapter()
            assert "ANTHROPIC_API_KEY" in str(exc_info.value)

    def test_gemini_adapter_fails_fast_with_missing_key(self):
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(MissingApiKeyError) as exc_info:
                GeminiAdapter()
            assert "GEMINI_API_KEY" in str(exc_info.value)

    def test_qwen_adapter_fails_fast_with_missing_key(self):
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(MissingApiKeyError) as exc_info:
                QwenProvider()
            assert "DASHSCOPE_API_KEY" in str(exc_info.value)

    def test_adapters_accept_settings_instance(self):
        settings = Settings(
            openai_api_key="sk-test-settings-key-1234567890",
            openai_model="gpt-4o-custom",
        )
        adapter = GPTAdapter(settings=settings)
        assert adapter.api_key == "sk-test-settings-key-1234567890"
        assert adapter.model == "gpt-4o-custom"

    def test_model_gateway_with_settings(self):
        settings = Settings(primary_provider="mock")
        gw = ModelGateway(settings=settings)
        assert gw.provider_name == "MockAdapter"
        resp = gw.generate("Hello", response_type="hint")
        assert resp.content.startswith("[MockProvider]")

    def test_model_gateway_build_adapter_with_missing_key_fails_fast(self):
        settings = Settings(primary_provider="openai", openai_api_key=None)
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(MissingApiKeyError):
                ModelGateway(primary_provider="openai", settings=settings)


# ===========================================================================
# 4. SecretScrubberFilter & Log Redaction Tests
# ===========================================================================

class TestSecretScrubberFilter:
    def setup_method(self):
        self.filter = SecretScrubberFilter()

    def test_scrub_openai_legacy_key(self):
        text = "Error connecting with key sk-1234567890abcdef1234567890 to OpenAI"
        scrubbed = self.filter.scrub_text(text)
        assert "sk-1234567890abcdef1234567890" not in scrubbed
        assert "[REDACTED_OPENAI_KEY]" in scrubbed

    def test_scrub_openai_project_key(self):
        text = "Using project key sk-proj-abc123xyz45678901234567890_test for completion"
        scrubbed = self.filter.scrub_text(text)
        assert "sk-proj-abc123xyz45678901234567890_test" not in scrubbed
        assert "[REDACTED_OPENAI_KEY]" in scrubbed

    def test_scrub_anthropic_key(self):
        text = "Anthropic client initialized with sk-ant-api03-abcdef12345678901234567890"
        scrubbed = self.filter.scrub_text(text)
        assert "sk-ant-api03-abcdef12345678901234567890" not in scrubbed
        assert "[REDACTED_ANTHROPIC_KEY]" in scrubbed

    def test_scrub_gemini_key(self):
        text = "Gemini key: AIzaSyD3x9L8p2Qw1R4t7U6v5Y8z0A1B2C3D4E5"
        scrubbed = self.filter.scrub_text(text)
        assert "AIzaSyD3x9L8p2Qw1R4t7U6v5Y8z0A1B2C3D4E5" not in scrubbed
        assert "[REDACTED_GEMINI_KEY]" in scrubbed

    def test_scrub_aws_key(self):
        text = "Failed AWS S3 auth: AKIAIOSFODNN7EXAMPLE"
        scrubbed = self.filter.scrub_text(text)
        assert "AKIAIOSFODNN7EXAMPLE" not in scrubbed
        assert "[REDACTED_AWS_KEY]" in scrubbed

    def test_scrub_bearer_token(self):
        text = "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.e30"
        scrubbed = self.filter.scrub_text(text)
        assert "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9" not in scrubbed
        assert "Bearer [REDACTED_BEARER_TOKEN]" in scrubbed

    def test_scrub_generic_key_assignment(self):
        text = 'Payload dump: {"api_key": "my_super_secret_token_12345", "status": "ok"}'
        scrubbed = self.filter.scrub_text(text)
        assert "my_super_secret_token_12345" not in scrubbed
        assert "[REDACTED_SECRET]" in scrubbed

    def test_scrub_exact_custom_secret(self):
        filter_with_custom = SecretScrubberFilter(extra_secrets=["custom_db_password_XYZ987"])
        text = "Failed to connect to postgresql://user:custom_db_password_XYZ987@localhost:5432/db"
        scrubbed = filter_with_custom.scrub_text(text)
        assert "custom_db_password_XYZ987" not in scrubbed
        assert "[REDACTED_SECRET]" in scrubbed

    def test_filter_scrubs_log_record_message_and_args(self):
        logger = logging.getLogger("test_scrub_logger")
        logger.setLevel(logging.INFO)
        log_stream = io.StringIO()
        handler = logging.StreamHandler(log_stream)
        scrubber = SecretScrubberFilter()
        logger.addFilter(scrubber)
        handler.addFilter(scrubber)
        logger.addHandler(handler)

        logger.info(
            "Connection failed with key %s for user %s",
            "sk-proj-supersecretkey123456789012345",
            "student_01"
        )
        handler.flush()
        output = log_stream.getvalue()

        assert "sk-proj-supersecretkey123456789012345" not in output
        assert "[REDACTED_OPENAI_KEY]" in output

    def test_filter_scrubs_stack_trace_exc_text(self):
        logger = logging.getLogger("test_exc_scrub_logger")
        logger.setLevel(logging.ERROR)
        log_stream = io.StringIO()
        handler = logging.StreamHandler(log_stream)
        scrubber = SecretScrubberFilter()
        logger.addFilter(scrubber)
        handler.addFilter(scrubber)
        logger.addHandler(handler)

        try:
            raise ValueError("Authentication error with sk-ant-api03-secretkey1234567890")
        except Exception:
            logger.exception("Caught unhandled exception")

        handler.flush()
        output = log_stream.getvalue()

        assert "sk-ant-api03-secretkey1234567890" not in output
        assert "[REDACTED_ANTHROPIC_KEY]" in output

    def test_scrub_text_utility_function(self):
        result = scrub_text("API token sk-test12345678901234567890 leaked")
        assert "sk-test12345678901234567890" not in result
        assert "[REDACTED_OPENAI_KEY]" in result
