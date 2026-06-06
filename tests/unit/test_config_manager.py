import json
from pathlib import Path

import pytest

from providers.config_manager import ConfigDict, load_config, save_config, sanitize_config


def valid_config() -> dict:
    return {
        "llm": {
            "base_url": "http://localhost:11434/v1",
            "model": "llama3",
            "api_key": "ollama-secret",
        },
        "transcription": {
            "base_url": "https://api.openai.com/v1",
            "model": "whisper-1",
            "api_key": "sk-test-secret-1234567890abcdef",
        },
        "portrait": {
            "face_backend": "opencv",
            "split_enabled": True,
            "split_active_threshold": 0.15,
            "split_window_ratio": 0.6,
            "split_hysteresis_sec": 3.0,
            "body_head_pad_ratio": 0.30,
            "body_lower_pad_ratio": 1.20,
        },
    }


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def test_load_valid_config(tmp_path):
    path = tmp_path / "config.json"
    cfg = valid_config()
    write_json(path, cfg)

    loaded = load_config(path)

    assert isinstance(loaded, ConfigDict)
    assert loaded["llm"]["base_url"] == cfg["llm"]["base_url"]
    assert loaded["llm"]["model"] == cfg["llm"]["model"]
    assert loaded["transcription"]["base_url"] == cfg["transcription"]["base_url"]
    assert loaded["transcription"]["model"] == cfg["transcription"]["model"]


def test_load_missing_file_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "missing.json")


def test_load_missing_required_key_raises_value_error(tmp_path):
    path = tmp_path / "config.json"
    cfg = valid_config()
    del cfg["llm"]["model"]
    write_json(path, cfg)

    with pytest.raises(ValueError) as exc_info:
        load_config(path)

    assert "llm.model" in str(exc_info.value)


def test_load_malformed_json_raises_value_error(tmp_path):
    path = tmp_path / "config.json"
    path.write_text("{bad json", encoding="utf-8")

    with pytest.raises(ValueError) as exc_info:
        load_config(path)

    assert "Invalid JSON" in str(exc_info.value)


def test_save_and_reload_roundtrip_identical(tmp_path):
    path = tmp_path / "nested" / "config.json"
    cfg = valid_config()

    save_config(cfg, path)
    loaded = load_config(path)

    assert dict(loaded) == cfg


def test_api_key_not_in_repr_or_str(tmp_path):
    path = tmp_path / "config.json"
    cfg = valid_config()
    write_json(path, cfg)

    loaded = load_config(path)
    rendered = str(loaded)
    repr_rendered = repr(loaded)

    assert "ollama-secret" not in rendered
    assert "sk-test-secret-1234567890abcdef" not in rendered
    assert "ollama-secret" not in repr_rendered
    assert "sk-test-secret-1234567890abcdef" not in repr_rendered
    assert "***REDACTED***" in rendered


def test_sanitize_config_redacts_nested_sensitive_values():
    cfg = valid_config()

    sanitized = sanitize_config(cfg)

    assert sanitized["llm"]["api_key"] == "***REDACTED***"
    assert sanitized["transcription"]["api_key"] == "***REDACTED***"
    assert sanitized["llm"]["model"] == "llama3"


def test_save_config_refuses_invalid_config(tmp_path):
    cfg = valid_config()
    cfg["transcription"]["base_url"] = ""

    with pytest.raises(ValueError) as exc_info:
        save_config(cfg, tmp_path / "config.json")

    assert "transcription.base_url" in str(exc_info.value)


def test_load_config_returns_plain_dict_compatible_type(tmp_path):
    path = tmp_path / "config.json"
    cfg = valid_config()
    write_json(path, cfg)

    loaded = load_config(path)

    assert isinstance(loaded, dict)
    assert loaded["portrait"]["face_backend"] == "opencv"


def test_default_path_can_load_project_config():
    loaded = load_config()

    assert loaded["llm"]["base_url"]
    assert loaded["llm"]["model"]
    assert loaded["transcription"]["base_url"]
    assert loaded["transcription"]["model"]
