"""
ai_tutor/log_scrubber.py
------------------------
Logging filter and security utility that intercepts and redacts sensitive credentials:
- LLM API keys (Anthropic, OpenAI, Google Gemini, DashScope / Alibaba).
- Cloud access keys & bearer tokens (AWS, GitHub, Slack, OAuth Bearer tokens).
- Dynamic scrubbing of active configured secrets.
- Redacts log messages, structured args, exception text, and stack traces.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple, Union


# Common API Key & Token Regex Patterns (Order matters: specific patterns first)
API_KEY_PATTERNS: List[Tuple[re.Pattern, str]] = [
    # 1. Anthropic API Keys (sk-ant-...) - Matched BEFORE generic sk-
    (
        re.compile(r"\bsk-ant-(?:api[0-9]{2}-)?[A-Za-z0-9_\-]{20,}\b"),
        "[REDACTED_ANTHROPIC_KEY]"
    ),
    # 2. OpenAI API Keys (legacy sk-..., project sk-proj-..., svcacct sk-svcacct-..., admin sk-admin-...)
    (
        re.compile(r"\bsk-(?:proj-|live-|test-|svcacct-|admin-)?[A-Za-z0-9_\-]{20,}\b"),
        "[REDACTED_OPENAI_KEY]"
    ),
    # 3. Google / Gemini API Keys (AIzaSy...)
    (
        re.compile(r"\bAIza[0-9A-Za-z\-_]{35}\b"),
        "[REDACTED_GEMINI_KEY]"
    ),
    # 4. AWS Access Key IDs (AKIA...)
    (
        re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
        "[REDACTED_AWS_KEY]"
    ),
    # 5. GitHub Personal Access Tokens (ghp_..., gho_..., ghu_..., ghs_..., ghr_...)
    (
        re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{36,}\b"),
        "[REDACTED_GITHUB_TOKEN]"
    ),
    # 6. Slack API Tokens (xoxb-..., xoxp-..., xoxa-..., xoxr-...)
    (
        re.compile(r"\bxox[baprs]-[0-9A-Za-z\-]{10,}\b"),
        "[REDACTED_SLACK_TOKEN]"
    ),
    # 7. Authorization Bearer Tokens
    (
        re.compile(r"(?i)\bBearer\s+([A-Za-z0-9_\-\.]{20,})\b"),
        "Bearer [REDACTED_BEARER_TOKEN]"
    ),
    # 8. Key-Value pairs in JSON, query params, or log lines (e.g. api_key="xyz", "password": "xyz")
    (
        re.compile(
            r"""(?i)(["']?(?:api[_-]?key|secret|password|auth[_-]?token|access[_-]?token|client[_-]?secret)["']?\s*[:=]\s*["']?)([^"'\s,;}{\]]{8,})(["']?)"""
        ),
        r"\1[REDACTED_SECRET]\3"
    ),
]


class SecretScrubberFilter(logging.Filter):
    """
    Logging filter that sanitizes log records to prevent sensitive credentials,
    API keys, and secret values from leaking into logs or stack traces.
    """

    def __init__(self, name: str = "", extra_secrets: Optional[List[str]] = None) -> None:
        super().__init__(name)
        self._custom_secrets: Set[str] = set()
        if extra_secrets:
            for s in extra_secrets:
                self.add_secret(s)

    def add_secret(self, secret: str) -> None:
        """Register an exact secret string to be scrubbed."""
        if secret and isinstance(secret, str) and len(secret.strip()) > 3:
            self._custom_secrets.add(secret.strip())

    def scrub_text(self, text: str) -> str:
        """Apply all redaction rules and registered exact secrets to a string."""
        if not text or not isinstance(text, str):
            return text

        scrubbed = text

        # 1. Redact known regex patterns
        for pattern, replacement in API_KEY_PATTERNS:
            scrubbed = pattern.sub(replacement, scrubbed)

        # 2. Redact exact loaded secrets if configured
        for secret in self._custom_secrets:
            if secret in scrubbed:
                scrubbed = scrubbed.replace(secret, "[REDACTED_SECRET]")

        return scrubbed

    def _scrub_object(self, obj: Any) -> Any:
        """Recursively scrub strings, dicts, lists, and tuples."""
        if isinstance(obj, str):
            return self.scrub_text(obj)
        elif isinstance(obj, dict):
            return {k: self._scrub_object(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._scrub_object(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self._scrub_object(item) for item in obj)
        return obj

    def filter(self, record: logging.LogRecord) -> bool:
        """
        Intercepts LogRecord and sanitizes msg, args, exc_text, and exc_info in place.
        Always returns True so the sanitized record proceeds to handlers.
        """
        # Scrub message
        if isinstance(record.msg, str):
            record.msg = self.scrub_text(record.msg)
        elif hasattr(record, "msg"):
            record.msg = self._scrub_object(record.msg)

        # Scrub args
        if record.args:
            if isinstance(record.args, dict):
                record.args = self._scrub_object(record.args)
            elif isinstance(record.args, (tuple, list)):
                record.args = tuple(self._scrub_object(arg) for arg in record.args)

        # Pre-format and scrub exception traceback if present
        if record.exc_info:
            if not record.exc_text:
                record.exc_text = logging.Formatter().formatException(record.exc_info)
            record.exc_info = None  # Ensure downstream handlers use scrubbed exc_text

        # Scrub exception traceback text
        if record.exc_text:
            record.exc_text = self.scrub_text(record.exc_text)

        # Scrub formatted stack info
        if hasattr(record, "stack_info") and record.stack_info:
            record.stack_info = self.scrub_text(record.stack_info)

        return True


def scrub_text(text: str, extra_secrets: Optional[List[str]] = None) -> str:
    """Helper function to sanitize text outside of logging records."""
    filter_inst = SecretScrubberFilter(extra_secrets=extra_secrets)
    return filter_inst.scrub_text(text)


def install_log_scrubber(
    target_logger: Optional[logging.Logger] = None,
    extra_secrets: Optional[List[str]] = None,
) -> SecretScrubberFilter:
    """
    Installs SecretScrubberFilter on the root logger, all attached handlers,
    and the 'ai_tutor' logger hierarchy.
    """
    # Attempt to load known secrets from Settings if available
    secrets: List[str] = list(extra_secrets or [])
    try:
        from .config import get_settings
        settings_secrets = get_settings().get_all_secrets()
        secrets.extend(settings_secrets)
    except Exception:
        pass

    scrubber = SecretScrubberFilter(extra_secrets=secrets)

    # Attach to root logger
    root_logger = logging.getLogger()
    if scrubber not in root_logger.filters:
        root_logger.addFilter(scrubber)

    # Attach to all root handlers
    for handler in root_logger.handlers:
        if scrubber not in handler.filters:
            handler.addFilter(scrubber)

    # Attach to target logger or ai_tutor package logger
    pkg_logger = target_logger or logging.getLogger("ai_tutor")
    if scrubber not in pkg_logger.filters:
        pkg_logger.addFilter(scrubber)
    for handler in pkg_logger.handlers:
        if scrubber not in handler.filters:
            handler.addFilter(scrubber)

    return scrubber
