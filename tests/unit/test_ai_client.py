"""Unit tests for providers/ai_client.py — OpenAI-Compatible AI Client Factory."""

from unittest.mock import patch

import pytest
from openai import OpenAI

from providers.ai_client import get_client


@pytest.fixture
def mock_config() -> dict:
    """Return a minimal config dict sufficient for get_client."""
    return {
        "llm": {
            "base_url": "http://localhost:11434/v1",
            "model": "llama3",
            "api_key": "ollama",
        }
    }


# ── test 1: get_client returns a valid OpenAI instance ──────────────────
def test_get_client_returns_openai_instance(mock_config: dict) -> None:
    client = get_client(mock_config)
    assert isinstance(client, OpenAI), (
        f"Expected openai.OpenAI instance, got {type(client)}"
    )


# ── test 2: client.base_url matches config ──────────────────────────────
def test_client_base_url_matches_config(mock_config: dict) -> None:
    client = get_client(mock_config)
    # OpenAI SDK stores base_url as httpx.URL — stringify for comparison
    assert str(client.base_url).rstrip("/") == "http://localhost:11434/v1"


# ── test 3: OpenAI constructor called with correct kwargs ───────────────
def test_openai_constructor_receives_kwargs(mock_config: dict) -> None:
    with patch("providers.ai_client.OpenAI", wraps=OpenAI) as mock_cls:
        get_client(mock_config)
        mock_cls.assert_called_once()
        call_kwargs = mock_cls.call_args.kwargs
        assert "base_url" in call_kwargs or "api_key" in call_kwargs, (
            "OpenAI() must be called with base_url and/or api_key from config"
        )


# ── test 4: no provider-specific branching in source ───────────────────
def test_no_provider_specific_code() -> None:
    """Ensure ai_client.py contains no if/else on provider name anywhere.

    ADR-003: the client factory must be a pure passthrough — no
    provider-name checks.
    """
    import inspect
    import providers.ai_client as mod

    source = inspect.getsource(mod)
    # Simple heuristic: no 'if.*provider' or 'elif' branching
    lowered = source.lower()
    assert "elif" not in lowered, "ai_client.py must not branch on provider"
    assert "if 'openai'" not in lowered, "Provider-specific branching detected"
    assert "if \"openai\"" not in lowered, "Provider-specific branching detected"


# ── test 5: api_key forwarded when provided ─────────────────────────────
def test_api_key_forwarded(mock_config: dict) -> None:
    with patch("providers.ai_client.OpenAI", wraps=OpenAI) as mock_cls:
        get_client(mock_config)
        call_kwargs = mock_cls.call_args.kwargs
        assert call_kwargs.get("api_key") == "ollama"


# ── test 6: default api_key for local servers ───────────────────────────
def test_default_api_key_when_missing() -> None:
    config: dict = {
        "llm": {
            "base_url": "http://localhost:11434/v1",
            "model": "llama3",
            # no api_key — should default to "ollama"
        }
    }
    with patch("providers.ai_client.OpenAI", wraps=OpenAI) as mock_cls:
        get_client(config)
        call_kwargs = mock_cls.call_args.kwargs
        assert call_kwargs.get("api_key") == "ollama", (
            "Default api_key must be 'ollama' for local servers"
        )
