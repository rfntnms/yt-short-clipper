"""OpenAI-Compatible AI Client Factory.

Single point of instantiation for the OpenAI client SDK.
Follows ADR-003: Generic endpoint passthrough with NO provider branching.
"""
from typing import Optional

from openai import OpenAI


def get_client(config: dict) -> OpenAI:
    """Return an OpenAI client configured via the given config dict.

    Uses `llm.base_url` and `llm.api_key`.
    Defaults to api_key="ollama" for local server compatibility.

    Args:
        config: A dict containing at minimum `llm.base_url`.

    Returns:
        A configured openai.OpenAI instance.
    """
    llm_conf = config.get("llm", {})

    base_url: Optional[str] = llm_conf.get("base_url")
    # API key is required by OpenAI constructor even if backend ignores it.
    api_key: str = llm_conf.get("api_key", "ollama")

    return OpenAI(base_url=base_url, api_key=api_key)
