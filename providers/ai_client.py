"""OpenAI-Compatible AI Client Factory.

Single point of instantiation for the OpenAI client SDK.
Follows ADR-003: Generic endpoint passthrough with NO provider branching.
"""
from openai import OpenAI


def get_client(config: dict) -> OpenAI:
    """Return an OpenAI client configured via the given config dict.

    Uses `llm.base_url` and `llm.api_key`.
    Defaults: base_url="https://api.openai.com/v1", api_key="ollama".

    Args:
        config: A dict containing at minimum `llm.base_url`.

    Returns:
        A configured openai.OpenAI instance.
    """
    llm_conf = config.get("llm", {})

    base_url: str = llm_conf.get("base_url") or "https://api.openai.com/v1"
    # API key is required by OpenAI constructor even if backend ignores it.
    api_key: str = llm_conf.get("api_key") or "ollama"

    return OpenAI(base_url=base_url, api_key=api_key)
