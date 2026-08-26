"""
tests/test_model_gateway.py
---------------------------
Unit tests for ModelGateway — provider abstraction, budget table,
response_type -> max_tokens resolution, and provider hot-swapping.
All tests run fully offline using MockProvider (no API keys required).
"""

import os
import pytest
from unittest.mock import patch, MagicMock

from ai_tutor.model_gateway import (
    ModelGateway,
    BaseModelProvider,
    MockProvider,
    OpenAIProvider,
    AnthropicProvider,
    QwenProvider,
    OUTPUT_BUDGET,
    resolve_max_tokens,
    _build_provider,
    _ALIAS_MAP,
)


# ---------------------------------------------------------------------------
# 1. Output Budget Table & Token Resolution
# ---------------------------------------------------------------------------

class TestOutputBudgetTable:
    def test_all_required_response_types_present(self):
        required = {
            "hint", "explain", "quiz_question", "assessment_feedback",
            "challenge", "guide", "summary", "general",
        }
        assert required.issubset(OUTPUT_BUDGET.keys())

    def test_token_ceilings_are_positive_integers(self):
        for response_type, ceiling in OUTPUT_BUDGET.items():
            assert isinstance(ceiling, int), f"{response_type} ceiling is not int"
            assert ceiling > 0, f"{response_type} ceiling is not positive"

    def test_hint_ceiling_is_300(self):
        assert OUTPUT_BUDGET["hint"] == 300

    def test_quiz_question_ceiling_is_1200(self):
        assert OUTPUT_BUDGET["quiz_question"] == 1200

    def test_explain_ceiling_is_600(self):
        assert OUTPUT_BUDGET["explain"] == 600

    def test_assessment_feedback_ceiling_is_400(self):
        assert OUTPUT_BUDGET["assessment_feedback"] == 400


class TestResolveMaxTokens:
    def test_resolves_exact_key(self):
        assert resolve_max_tokens("hint") == 300
        assert resolve_max_tokens("explain") == 600
        assert resolve_max_tokens("quiz_question") == 1200

    def test_resolves_alias_quiz(self):
        assert resolve_max_tokens("quiz") == OUTPUT_BUDGET["quiz_question"]

    def test_resolves_alias_assess(self):
        assert resolve_max_tokens("assess") == OUTPUT_BUDGET["assessment_feedback"]

    def test_resolves_alias_socratic_to_guide(self):
        assert resolve_max_tokens("socratic") == OUTPUT_BUDGET["guide"]

    def test_resolves_alias_direct_to_explain(self):
        assert resolve_max_tokens("direct") == OUTPUT_BUDGET["explain"]

    def test_unknown_type_falls_back_to_general(self):
        assert resolve_max_tokens("nonexistent_type") == OUTPUT_BUDGET["general"]

    def test_case_insensitive_resolution(self):
        assert resolve_max_tokens("HINT") == 300
        assert resolve_max_tokens("Explain") == 600


# ---------------------------------------------------------------------------
# 2. MockProvider
# ---------------------------------------------------------------------------

class TestMockProvider:
    def test_returns_string(self):
        provider = MockProvider()
        result = provider._call("Hello world", max_tokens=300)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_includes_max_tokens_in_output(self):
        provider = MockProvider()
        result = provider._call("Test prompt", max_tokens=600)
        assert "600" in result

    def test_generate_respects_response_type_budget(self):
        provider = MockProvider()
        result = provider.generate("Some prompt", response_type="hint")
        assert "300" in result  # MockProvider echoes max_tokens=300 for hint


# ---------------------------------------------------------------------------
# 3. ModelGateway Initialisation
# ---------------------------------------------------------------------------

class TestModelGatewayInit:
    def test_defaults_to_mock_provider_in_test_env(self):
        """GATEWAY_PROVIDER not set -> falls back to 'mock'."""
        env = {k: v for k, v in os.environ.items() if k != "GATEWAY_PROVIDER"}
        with patch.dict(os.environ, env, clear=True):
            gw = ModelGateway()
        assert gw.provider_name == "MockProvider"

    def test_explicit_mock_provider(self):
        gw = ModelGateway(provider="mock")
        assert gw.provider_name == "MockProvider"

    def test_provider_injection_bypasses_registry(self):
        custom = MockProvider()
        gw = ModelGateway(_provider_instance=custom)
        assert gw.provider_name == "MockProvider"

    def test_unknown_provider_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown provider"):
            ModelGateway(provider="superai")

    def test_env_var_selects_provider(self):
        with patch.dict(os.environ, {"GATEWAY_PROVIDER": "mock"}):
            gw = ModelGateway()
        assert gw.provider_name == "MockProvider"


# ---------------------------------------------------------------------------
# 4. ModelGateway.generate — core contract
# ---------------------------------------------------------------------------

class TestModelGatewayGenerate:
    def setup_method(self):
        self.gw = ModelGateway(provider="mock")

    def test_generate_returns_string(self):
        result = self.gw.generate("Explain gradient descent.", "explain")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_generate_hint_uses_300_token_ceiling(self):
        result = self.gw.generate("Give me a hint.", "hint")
        assert "300" in result

    def test_generate_quiz_question_uses_1200_token_ceiling(self):
        result = self.gw.generate("Generate a quiz.", "quiz_question")
        assert "1200" in result

    def test_generate_assessment_feedback_uses_400_token_ceiling(self):
        result = self.gw.generate("Grade this answer.", "assessment_feedback")
        assert "400" in result

    def test_generate_empty_prompt_returns_empty_string(self):
        result = self.gw.generate("", "hint")
        assert result == ""

    def test_generate_whitespace_prompt_returns_empty_string(self):
        result = self.gw.generate("   \n  ", "explain")
        assert result == ""

    def test_generate_via_alias_quiz(self):
        result = self.gw.generate("Quiz me.", "quiz")
        assert "1200" in result  # alias resolves to quiz_question -> 1200

    def test_generate_unknown_response_type_falls_back_to_general(self):
        result = self.gw.generate("Anything.", "unknown_type")
        assert "512" in result  # general fallback ceiling


# ---------------------------------------------------------------------------
# 5. Provider hot-swap
# ---------------------------------------------------------------------------

class TestModelGatewaySwap:
    def test_swap_provider_changes_active_provider(self):
        gw = ModelGateway(provider="mock")
        assert gw.provider_name == "MockProvider"
        gw.swap_provider("mock")  # swap to another mock instance
        assert gw.provider_name == "MockProvider"

    def test_swap_to_unknown_raises_value_error(self):
        gw = ModelGateway(provider="mock")
        with pytest.raises(ValueError, match="Unknown provider"):
            gw.swap_provider("magic")

    def test_generate_still_works_after_swap(self):
        gw = ModelGateway(provider="mock")
        gw.swap_provider("mock")
        result = gw.generate("Still works?", "guide")
        assert "250" in result  # guide ceiling


# ---------------------------------------------------------------------------
# 6. Introspection helpers
# ---------------------------------------------------------------------------

class TestModelGatewayIntrospection:
    def setup_method(self):
        self.gw = ModelGateway(provider="mock")

    def test_output_budget_returns_full_table(self):
        budget = self.gw.output_budget()
        assert "hint" in budget
        assert "quiz_question" in budget
        assert budget["hint"] == 300

    def test_output_budget_returns_copy(self):
        budget = self.gw.output_budget()
        budget["hint"] = 9999
        assert OUTPUT_BUDGET["hint"] == 300  # original unmodified

    def test_max_tokens_for_hint(self):
        assert self.gw.max_tokens_for("hint") == 300

    def test_max_tokens_for_explain(self):
        assert self.gw.max_tokens_for("explain") == 600

    def test_provider_name_is_string(self):
        assert isinstance(self.gw.provider_name, str)


# ---------------------------------------------------------------------------
# 7. Provider registry
# ---------------------------------------------------------------------------

class TestProviderRegistry:
    def test_build_openai_provider(self):
        provider = _build_provider("openai")
        assert isinstance(provider, OpenAIProvider)

    def test_build_anthropic_provider(self):
        provider = _build_provider("anthropic")
        assert isinstance(provider, AnthropicProvider)

    def test_build_claude_alias(self):
        provider = _build_provider("claude")
        assert isinstance(provider, AnthropicProvider)

    def test_build_qwen_provider(self):
        provider = _build_provider("qwen")
        assert isinstance(provider, QwenProvider)

    def test_build_mock_provider(self):
        provider = _build_provider("mock")
        assert isinstance(provider, MockProvider)

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError):
            _build_provider("unknown_xyz")


# ---------------------------------------------------------------------------
# 8. Provider env-var wiring (no live calls)
# ---------------------------------------------------------------------------

class TestProviderEnvVarWiring:
    def test_openai_reads_api_key_from_env(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key-123", "OPENAI_MODEL": "gpt-4o"}):
            provider = OpenAIProvider()
        assert provider.api_key == "test-key-123"
        assert provider.model == "gpt-4o"

    def test_anthropic_reads_api_key_from_env(self):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "anthro-key", "ANTHROPIC_MODEL": "claude-3-opus-20240229"}):
            provider = AnthropicProvider()
        assert provider.api_key == "anthro-key"
        assert provider.model == "claude-3-opus-20240229"

    def test_qwen_reads_api_key_from_env(self):
        with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "qwen-key", "QWEN_MODEL": "qwen-turbo"}):
            provider = QwenProvider()
        assert provider.api_key == "qwen-key"
        assert provider.model == "qwen-turbo"

    def test_qwen_uses_dashscope_base_url(self):
        provider = QwenProvider()
        assert "dashscope" in provider.base_url
