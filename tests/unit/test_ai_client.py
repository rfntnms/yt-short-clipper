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


def test_get_client_returns_openai_instance(mock_config: dict) -> None:
    client = get_client(mock_config)
    assert isinstance(client, OpenAI), f"Expected openai.OpenAI instance, got {type(client)}"


def test_client_base_url_matches_config(mock_config: dict) -> None:
    client = get_client(mock_config)
    assert str(client.base_url).rstrip("/") == "http://localhost:11434/v1"


def test_openai_constructor_receives_kwargs(mock_config: dict) -> None:
    with patch("providers.ai_client.OpenAI", wraps=OpenAI) as mock_cls:
        get_client(mock_config)
        mock_cls.assert_called_once()
        call_kwargs = mock_cls.call_args.kwargs
        assert call_kwargs["base_url"] == "http://localhost:11434/v1"
        assert call_kwargs["api_key"] == "ollama"


def test_no_provider_specific_code() -> None:
    """Ensure ai_client.py contains no provider-specific branching."""
    import inspect
    import providers.ai_client as mod

    source = inspect.getsource(mod).lower()
    assert "elif" not in source, "ai_client.py must not branch on provider"
    assert "if 'openai'" not in source, "Provider-specific branching detected"
    assert 'if "openai"' not in source, "Provider-specific branching detected"


def test_api_key_forwarded(mock_config: dict) -> None:
    with patch("providers.ai_client.OpenAI", wraps=OpenAI) as mock_cls:
        get_client(mock_config)
        call_kwargs = mock_cls.call_args.kwargs
        assert call_kwargs.get("api_key") == "ollama"


def test_default_api_key_when_missing() -> None:
    config = {"llm": {"base_url": "http://localhost:11434/v1", "model": "llama3"}}
    with patch("providers.ai_client.OpenAI", wraps=OpenAI) as mock_cls:
        get_client(config)
        call_kwargs = mock_cls.call_args.kwargs
        assert call_kwargs.get("api_key") == "ollama"


def test_default_base_url_when_missing() -> None:
    client = get_client({"llm": {"api_key": "sk-test"}})
    assert str(client.base_url).rstrip("/") == "https://api.openai.com/v1"


def test_empty_base_url_uses_default() -> None:
    client = get_client({"llm": {"base_url": "", "api_key": "sk-test"}})
    assert str(client.base_url).rstrip("/") == "https://api.openai.com/v1"


def test_integration_with_mock_server() -> None:
    """Verify a real OpenAI SDK client points at a local OpenAI-compatible endpoint."""
    import threading
    from wsgiref.simple_server import make_server

    def mock_app(environ, start_response):
        start_response("200 OK", [("Content-Type", "application/json")])
        return [b'{"object":"list","data":[{"id":"mock-model","object":"model"}]}']

    server = make_server("127.0.0.1", 0, mock_app)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        client = get_client({"llm": {"base_url": f"http://127.0.0.1:{port}/v1", "api_key": "test"}})
        models = client.models.list()
        assert isinstance(client, OpenAI)
        assert str(client.base_url).rstrip("/") == f"http://127.0.0.1:{port}/v1"
        assert models.data[0].id == "mock-model"
    finally:
        server.shutdown()
        thread.join(timeout=2)
