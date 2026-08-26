"""
model_gateway.py
----------------
ModelGateway: Single-interface abstraction over multiple LLM providers.

Design Contract:
- generate(prompt, response_type) -> str
  • `prompt`        – fully assembled prompt string (from BudgetManager or TutorReasoner)
  • `response_type` – maps to a max_tokens ceiling via the OUTPUT BUDGET TABLE below

Output Budget Table
-------------------
| response_type          | max_tokens | Use-case                                      |
|------------------------|-----------|-----------------------------------------------|
| hint                   |   300     | Single progressive scaffold hint              |
| explain                |   600     | Targeted concept explanation                  |
| quiz_question          |  1200     | Rich formative question (500–1200 budget)     |
| assessment_feedback    |   400     | Answer grading + formative feedback           |
| challenge              |   800     | Advanced multi-step challenge prompt          |
| guide                  |   250     | Socratic diagnostic / opening question        |
| summary                |   200     | Conversation summary compression             |
| general                |   512     | Default fallback for unclassified responses  |

Provider Selection:
- Configured via `GATEWAY_PROVIDER` env-var: "openai" | "anthropic" | "qwen"
- Model and API key read from env-vars per provider (see PROVIDER_ENV_MAP below).
- Swapping providers only requires changing GATEWAY_PROVIDER — this file is the
  single touch-point, callers never import provider SDKs directly.
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

logger = logging.getLogger("ai_tutor.model_gateway")


# ---------------------------------------------------------------------------
# Output Budget Table  (response_type -> max_tokens)
# ---------------------------------------------------------------------------

OUTPUT_BUDGET: Dict[str, int] = {
    "hint":                300,
    "explain":             600,
    "quiz_question":      1200,
    "assessment_feedback": 400,
    "challenge":           800,
    "guide":               250,
    "summary":             200,
    "general":             512,
}

# Alias normalisation so callers can also pass StrategyAction values directly
_ALIAS_MAP: Dict[str, str] = {
    "quiz":      "quiz_question",
    "assess":    "assessment_feedback",
    "feedback":  "assessment_feedback",
    "socratic":  "guide",
    "direct":    "explain",
}

DEFAULT_RESPONSE_TYPE = "general"


def resolve_max_tokens(response_type: str) -> int:
    """Return the max_tokens ceiling for the given response_type."""
    key = _ALIAS_MAP.get(response_type.lower(), response_type.lower())
    tokens = OUTPUT_BUDGET.get(key, OUTPUT_BUDGET[DEFAULT_RESPONSE_TYPE])
    logger.debug("resolve_max_tokens: response_type=%r -> max_tokens=%d", response_type, tokens)
    return tokens


# ---------------------------------------------------------------------------
# Provider base interface
# ---------------------------------------------------------------------------

class BaseModelProvider(ABC):
    """Low-level provider wrapper — implements _call(prompt, max_tokens) -> str."""

    @abstractmethod
    def _call(self, prompt: str, max_tokens: int, **kwargs: Any) -> str:
        pass

    def generate(self, prompt: str, response_type: str = DEFAULT_RESPONSE_TYPE, **kwargs: Any) -> str:
        max_tokens = resolve_max_tokens(response_type)
        logger.info(
            "%s.generate | response_type=%r max_tokens=%d prompt_chars=%d",
            self.__class__.__name__, response_type, max_tokens, len(prompt)
        )
        return self._call(prompt, max_tokens, **kwargs)


# ---------------------------------------------------------------------------
# Concrete Provider Implementations
# ---------------------------------------------------------------------------

class OpenAIProvider(BaseModelProvider):
    """
    OpenAI Chat Completions API provider.
    Also compatible with Azure OpenAI, Groq, and any OpenAI-compatible endpoint.

    Env vars:
        OPENAI_API_KEY      – API key (required)
        OPENAI_MODEL        – Model name (default: gpt-4o-mini)
        OPENAI_BASE_URL     – Override base URL for Azure / Groq / Ollama
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> None:
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")

    def _call(self, prompt: str, max_tokens: int, **kwargs: Any) -> str:
        from openai import OpenAI  # lazy import — not installed in test env
        client = OpenAI(api_key=self.api_key, base_url=self.base_url or None)
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=kwargs.get("temperature", 0.4),
        )
        return (response.choices[0].message.content or "").strip()


class AnthropicProvider(BaseModelProvider):
    """
    Anthropic Claude API provider.

    Env vars:
        ANTHROPIC_API_KEY   – API key (required)
        ANTHROPIC_MODEL     – Model name (default: claude-3-5-haiku-20241022)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None:
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY", "")
        self.model = model or os.getenv("ANTHROPIC_MODEL", "claude-3-5-haiku-20241022")

    def _call(self, prompt: str, max_tokens: int, **kwargs: Any) -> str:
        import anthropic  # lazy import
        client = anthropic.Anthropic(api_key=self.api_key)
        message = client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        # content is a list of ContentBlock objects
        return "".join(
            block.text for block in message.content
            if hasattr(block, "text")
        ).strip()


class QwenProvider(BaseModelProvider):
    """
    Alibaba Qwen provider via DashScope OpenAI-compatible endpoint.

    Env vars:
        DASHSCOPE_API_KEY   – API key (required)
        QWEN_MODEL          – Model name (default: qwen-plus)
        DASHSCOPE_BASE_URL  – API base URL (default: https://dashscope.aliyuncs.com/compatible-mode/v1)
    """

    _DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None:
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY", "")
        self.model = model or os.getenv("QWEN_MODEL", "qwen-plus")
        self.base_url = os.getenv("DASHSCOPE_BASE_URL", self._DEFAULT_BASE_URL)

    def _call(self, prompt: str, max_tokens: int, **kwargs: Any) -> str:
        from openai import OpenAI  # DashScope exposes an OpenAI-compatible API
        client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=kwargs.get("temperature", 0.4),
        )
        return (response.choices[0].message.content or "").strip()


class MockProvider(BaseModelProvider):
    """
    Deterministic mock provider for offline testing.
    Returns a predictable string — no network calls made.
    """

    def _call(self, prompt: str, max_tokens: int, **kwargs: Any) -> str:
        excerpt = prompt[:60].replace("\n", " ")
        return f"[MockProvider] prompt='{excerpt}...' max_tokens={max_tokens}"


# ---------------------------------------------------------------------------
# Provider Registry & Factory
# ---------------------------------------------------------------------------

PROVIDER_REGISTRY: Dict[str, type] = {
    "openai":    OpenAIProvider,
    "anthropic": AnthropicProvider,
    "claude":    AnthropicProvider,   # alias
    "qwen":      QwenProvider,
    "mock":      MockProvider,
}


def _build_provider(provider_name: str) -> BaseModelProvider:
    """Instantiate the provider class registered under *provider_name*."""
    cls = PROVIDER_REGISTRY.get(provider_name.lower())
    if cls is None:
        raise ValueError(
            f"Unknown provider '{provider_name}'. "
            f"Valid options: {sorted(PROVIDER_REGISTRY)}"
        )
    return cls()


# ---------------------------------------------------------------------------
# ModelGateway — the single public interface
# ---------------------------------------------------------------------------

class ModelGateway:
    """
    Single-interface LLM gateway for the AI Tutor.

    Usage:
        gw = ModelGateway()                           # reads GATEWAY_PROVIDER from env
        text = gw.generate(prompt, "hint")            # max_tokens=300
        text = gw.generate(prompt, "quiz_question")   # max_tokens=1200

    Config (environment variables):
        GATEWAY_PROVIDER    – "openai" | "anthropic" | "claude" | "qwen" | "mock"
                              Default: "mock" (safe for local dev / testing)

    Provider-specific keys are documented on each provider class above.
    """

    #: Default provider when GATEWAY_PROVIDER env-var is unset
    DEFAULT_PROVIDER = "mock"

    def __init__(
        self,
        provider: Optional[str] = None,
        _provider_instance: Optional[BaseModelProvider] = None,
    ) -> None:
        """
        Parameters
        ----------
        provider:
            Override the provider name. Falls back to ``GATEWAY_PROVIDER`` env-var,
            then ``DEFAULT_PROVIDER``.
        _provider_instance:
            Inject a pre-built ``BaseModelProvider`` directly (useful for testing).
        """
        if _provider_instance is not None:
            self._provider = _provider_instance
        else:
            provider_name = (
                provider
                or os.getenv("GATEWAY_PROVIDER", self.DEFAULT_PROVIDER)
            )
            self._provider = _build_provider(provider_name)

        logger.info(
            "ModelGateway initialised with provider: %s",
            self._provider.__class__.__name__
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        response_type: str = DEFAULT_RESPONSE_TYPE,
        **kwargs: Any,
    ) -> str:
        """
        Generate a model response.

        Parameters
        ----------
        prompt:
            Fully assembled prompt string (from BudgetManager / TutorReasoner).
        response_type:
            Maps to a max_tokens ceiling via OUTPUT_BUDGET table.
            Valid values: "hint", "explain", "quiz_question", "assessment_feedback",
            "challenge", "guide", "summary", "general" (and aliases).
        **kwargs:
            Passed through to the provider (e.g. temperature, stop sequences).

        Returns
        -------
        str
            Raw text response from the model.
        """
        if not prompt or not prompt.strip():
            logger.warning("ModelGateway.generate called with empty prompt")
            return ""
        return self._provider.generate(prompt=prompt, response_type=response_type, **kwargs)

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------

    @property
    def provider_name(self) -> str:
        """Human-readable name of the active provider."""
        return self._provider.__class__.__name__

    @staticmethod
    def output_budget() -> Dict[str, int]:
        """Return a copy of the full OUTPUT_BUDGET table."""
        return dict(OUTPUT_BUDGET)

    def max_tokens_for(self, response_type: str) -> int:
        """Return the max_tokens ceiling for *response_type*."""
        return resolve_max_tokens(response_type)

    def swap_provider(self, provider: str) -> None:
        """
        Hot-swap the active provider at runtime without recreating the gateway.

        Parameters
        ----------
        provider:
            Any registered provider name ("openai", "anthropic", "qwen", "mock").
        """
        old = self.provider_name
        self._provider = _build_provider(provider)
        logger.info("ModelGateway: swapped provider %s -> %s", old, self.provider_name)
