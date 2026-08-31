"""
ai_tutor/config.py
------------------
Typed application configuration module powered by pydantic-settings.
Enforces strict environment variable loading for all secrets and API keys:
- No defaults containing real secrets or keys.
- Fast-fail validation on application startup or provider adapter creation.
- Descriptive error messages naming the exact missing environment variable.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, Dict, List, Optional, Union
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class MissingApiKeyError(ValueError):
    """Raised when a required API key or secret environment variable is missing."""
    def __init__(self, provider: str, env_var: str):
        self.provider = provider
        self.env_var = env_var
        super().__init__(
            f"Missing required environment variable '{env_var}' for provider '{provider}'. "
            f"Please set '{env_var}' in your environment or .env file. "
            f"Never hardcode or inline real secrets."
        )


class Settings(BaseSettings):
    """
    Application Settings schema.
    Loads every API key and secret strictly from environment variables or .env file.
    No secrets contain real default values.
    """
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # -----------------------------------------------------------------------
    # API Keys & Secrets (Zero default real values)
    # -----------------------------------------------------------------------
    openai_api_key: Optional[str] = Field(
        default=None,
        description="OpenAI API Key (loaded from OPENAI_API_KEY)",
    )
    anthropic_api_key: Optional[str] = Field(
        default=None,
        description="Anthropic API Key (loaded from ANTHROPIC_API_KEY)",
    )
    gemini_api_key: Optional[str] = Field(
        default=None,
        description="Google Gemini API Key (loaded from GEMINI_API_KEY)",
    )
    google_api_key: Optional[str] = Field(
        default=None,
        description="Fallback Google API Key (loaded from GOOGLE_API_KEY)",
    )
    dashscope_api_key: Optional[str] = Field(
        default=None,
        description="Alibaba DashScope API Key for Qwen (loaded from DASHSCOPE_API_KEY)",
    )
    database_url: Optional[str] = Field(
        default=None,
        description="PostgreSQL Database connection URL (loaded from DATABASE_URL)",
    )
    redis_url: Optional[str] = Field(
        default=None,
        description="Redis connection URL (loaded from REDIS_URL)",
    )

    # -----------------------------------------------------------------------
    # LLM Gateway & Provider Configuration
    # -----------------------------------------------------------------------
    primary_provider: str = Field(
        default="mock",
        description="Primary LLM provider (mock, gpt/openai, claude/anthropic, gemini/google, qwen)",
    )
    fallback_provider: Optional[str] = Field(
        default=None,
        description="Optional fallback LLM provider to invoke on primary error/timeout",
    )
    openai_model: str = Field(
        default="gpt-4o-mini",
        description="Default OpenAI model name",
    )
    openai_base_url: Optional[str] = Field(
        default=None,
        description="Custom OpenAI-compatible base URL (Azure, Groq, Ollama, LiteLLM)",
    )
    anthropic_model: str = Field(
        default="claude-3-5-haiku-20241022",
        description="Default Anthropic Claude model name",
    )
    gemini_model: str = Field(
        default="gemini-2.0-flash",
        description="Default Google Gemini model name",
    )
    qwen_model: str = Field(
        default="qwen-plus",
        description="Default Qwen model name",
    )
    dashscope_base_url: Optional[str] = Field(
        default=None,
        description="Custom DashScope compatible base URL",
    )
    model_request_timeout: float = Field(
        default=15.0,
        description="Timeout in seconds for LLM provider API requests",
    )
    db_timeout: float = Field(
        default=5.0,
        description="Timeout in seconds for database operations",
    )
    redis_timeout: float = Field(
        default=3.0,
        description="Timeout in seconds for Redis operations",
    )

    # -----------------------------------------------------------------------
    # Server & Service Configuration
    # -----------------------------------------------------------------------
    host: str = Field(default="0.0.0.0", description="Server host interface")
    port: int = Field(default=8000, description="Server port")
    cors_origins: Union[List[str], str] = Field(
        default=["http://localhost:5173", "http://localhost:3000"],
        description="Allowed CORS origin domains",
    )
    ai_tutor_log_dir: str = Field(
        default="logs",
        description="Directory for audit and review logs",
    )
    environment: str = Field(
        default="development",
        description="Runtime environment (development, staging, production, test)",
    )

    # -----------------------------------------------------------------------
    # Pedagogical Defaults & Thresholds
    # -----------------------------------------------------------------------
    default_hint_budget: int = Field(default=3, description="Default hint budget per turn")
    quiz_mastery_threshold: float = Field(default=0.80, description="Mastery threshold for quiz promotion")
    challenge_mastery_threshold: float = Field(default=0.90, description="Mastery threshold for challenge mode")
    explain_failure_threshold: int = Field(default=2, description="Consecutive failure count to trigger direct explanation")

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _parse_cors_origins(cls, value: Any) -> List[str]:
        if isinstance(value, str):
            return [origin.strip() for origin in value.split(",") if origin.strip()]
        return value

    # -----------------------------------------------------------------------
    # Provider Key Resolution & Fast-Fail Startup Checks
    # -----------------------------------------------------------------------

    _PROVIDER_KEY_MAP: Dict[str, tuple[str, str]] = {
        "gpt":       ("openai_api_key", "OPENAI_API_KEY"),
        "openai":    ("openai_api_key", "OPENAI_API_KEY"),
        "claude":    ("anthropic_api_key", "ANTHROPIC_API_KEY"),
        "anthropic": ("anthropic_api_key", "ANTHROPIC_API_KEY"),
        "gemini":    ("gemini_api_key", "GEMINI_API_KEY"),
        "google":    ("gemini_api_key", "GEMINI_API_KEY"),
        "qwen":      ("dashscope_api_key", "DASHSCOPE_API_KEY"),
    }

    def get_api_key_for_provider(self, provider_name: str) -> str:
        """
        Retrieves the API key for a specified provider.
        Fails fast with MissingApiKeyError if the key is not set or empty.
        Mock provider requires no key and returns an empty string.
        """
        norm_name = provider_name.strip().lower()
        if norm_name in ("mock", "offline"):
            return ""

        mapping = self._PROVIDER_KEY_MAP.get(norm_name)
        if mapping is None:
            raise ValueError(
                f"Unknown provider '{provider_name}'. "
                f"Supported providers: {sorted(list(self._PROVIDER_KEY_MAP.keys()) + ['mock'])}"
            )

        attr_name, env_var = mapping
        key_val = getattr(self, attr_name, None)

        # Fallback for gemini -> google_api_key
        if not key_val and norm_name in ("gemini", "google"):
            key_val = self.google_api_key

        if not key_val or not str(key_val).strip():
            raise MissingApiKeyError(provider=provider_name, env_var=env_var)

        return str(key_val).strip()

    def validate_startup(
        self,
        required_providers: Optional[List[str]] = None,
        require_all: bool = False,
    ) -> None:
        """
        Startup health check to validate that necessary API keys and secrets exist.
        Fails fast at application initialization before serving any request.
        """
        missing: List[str] = []

        if require_all:
            providers_to_check = ["openai", "anthropic", "gemini", "qwen"]
        elif required_providers is not None:
            providers_to_check = required_providers
        else:
            providers_to_check = [self.primary_provider]
            if self.fallback_provider:
                providers_to_check.append(self.fallback_provider)

        for prov in providers_to_check:
            norm = prov.strip().lower()
            if norm in ("mock", "offline"):
                continue
            mapping = self._PROVIDER_KEY_MAP.get(norm)
            if not mapping:
                missing.append(f"Unknown provider configured: '{prov}'")
                continue
            attr_name, env_var = mapping
            val = getattr(self, attr_name, None)
            if not val and norm in ("gemini", "google"):
                val = self.google_api_key
            if not val or not str(val).strip():
                missing.append(f"{env_var} (required by provider '{prov}')")

        if missing:
            raise ValueError(
                f"Application startup failed due to missing required environment variable(s):\n"
                + "\n".join(f"  - {m}" for m in missing)
                + "\nPlease set these variables in your environment or .env file."
            )

    def get_all_secrets(self) -> List[str]:
        """Return non-empty secret values loaded into settings (for log redaction)."""
        secrets = [
            self.openai_api_key,
            self.anthropic_api_key,
            self.gemini_api_key,
            self.google_api_key,
            self.dashscope_api_key,
            self.database_url,
            self.redis_url,
        ]
        return [s.strip() for s in secrets if s and isinstance(s, str) and len(s.strip()) > 3]


@lru_cache()
def get_settings() -> Settings:
    """
    Retrieve cached application Settings instance.
    Uses lru_cache to avoid re-reading disk/environment on each call.
    """
    primary = (
        os.getenv("PRIMARY_PROVIDER")
        or os.getenv("GATEWAY_PRIMARY_PROVIDER")
        or os.getenv("GATEWAY_PROVIDER")
    )
    fallback = (
        os.getenv("FALLBACK_PROVIDER")
        or os.getenv("GATEWAY_FALLBACK_PROVIDER")
    )
    kwargs: Dict[str, Any] = {}
    if primary:
        kwargs["primary_provider"] = primary
    if fallback:
        kwargs["fallback_provider"] = fallback

    return Settings(**kwargs)


def reload_settings(**overrides: Any) -> Settings:
    """
    Clears the cached Settings instance and creates a fresh instance.
    Useful in unit tests and dynamic reconfiguration.
    """
    get_settings.cache_clear()
    if overrides:
        return Settings(**overrides)
    return get_settings()
