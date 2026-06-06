"""Configuration manager — load, save, validate config.json.

Single source of truth for LLM endpoint, Whisper endpoint, and portrait settings.
No global state. All modules receive config as a plain dict via these functions.
Sensitive keys (API keys) must never appear in logs or repr.
"""
import json
from pathlib import Path
from typing import Any

# Defaults
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.json"
REQUIRED_KEYS = [
    "llm.base_url",
    "llm.model",
    "transcription.base_url",
    "transcription.model",
]

_SENSITIVE_FIELDS = {"api_key", "apikey", "api-key"}


class ConfigDict(dict):
    """Dict subclass that redacts sensitive fields in __str__ and __repr__."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # Recursively wrap nested dicts
        for key, value in self.items():
            if isinstance(value, dict) and not isinstance(value, ConfigDict):
                self[key] = ConfigDict(value)

    def _redacted_copy(self) -> dict:
        """Return a shallow copy with sensitive values masked."""
        result = {}
        for key, value in self.items():
            if isinstance(value, ConfigDict):
                result[key] = value._redacted_copy()
            elif key in _SENSITIVE_FIELDS and isinstance(value, str) and value:
                result[key] = "***REDACTED***"
            else:
                result[key] = value
        return result

    def __str__(self) -> str:
        return json.dumps(self._redacted_copy(), indent=2)

    def __repr__(self) -> str:
        return f"ConfigDict({json.dumps(self._redacted_copy(), indent=2)})"


def _get_nested(data: dict, dotted_key: str) -> Any:
    """Get a value from a nested dict using a dotted key like 'llm.model'."""
    keys = dotted_key.split(".")
    current = data
    for k in keys:
        if isinstance(current, dict) and k in current:
            current = current[k]
        else:
            return None
    return current


def _validate_config(config: dict) -> None:
    """Validate that all required keys are present and non-empty.

    Raises:
        ValueError with descriptive message listing all missing keys.
    """
    missing = []
    for key in REQUIRED_KEYS:
        value = _get_nested(config, key)
        if value is None or (isinstance(value, str) and not value.strip()):
            missing.append(key)
    if missing:
        raise ValueError(f"Missing or empty required config keys: {', '.join(missing)}")


def load_config(path: str | Path | None = None) -> ConfigDict:
    """Load and validate a config.json file.

    Args:
        path: Path to config.json. Defaults to project root config.json.

    Returns:
        ConfigDict with all required keys present.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If the file is not valid JSON or required keys are missing.
    """
    if path is None:
        path = DEFAULT_CONFIG_PATH
    else:
        path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    raw = path.read_text(encoding="utf-8")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {path}: {e}") from e

    if not isinstance(data, dict):
        raise ValueError(f"Config must be a JSON object, got {type(data).__name__}")

    _validate_config(data)
    return ConfigDict(data)


def save_config(config: dict, path: str | Path | None = None) -> None:
    """Validate and save a config dict to disk.

    Args:
        config: Configuration dict to save.
        path: Destination path. Defaults to project root config.json.

    Raises:
        ValueError: If the config dict fails validation.
    """
    if path is None:
        path = DEFAULT_CONFIG_PATH
    else:
        path = Path(path)

    _validate_config(config)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


def sanitize_config(config: dict) -> dict:
    """Return a copy of the config dict with sensitive values masked.

    Useful for logging. Does not mutate the original.
    """
    result = {}
    for key, value in config.items():
        if isinstance(value, dict):
            result[key] = sanitize_config(value)
        elif key in _SENSITIVE_FIELDS and isinstance(value, str) and value:
            result[key] = "***REDACTED***"
        else:
            result[key] = value
    return result


__all__ = ["load_config", "save_config", "sanitize_config", "ConfigDict"]
