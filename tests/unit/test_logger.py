import logging
import os
import tempfile
import uuid
from pathlib import Path
import pytest

from utils.logger import setup_logger, logger as default_logger


@pytest.fixture
def tmp_log_dir():
    """Provide a temporary log directory for each test."""
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def clean_logger_name():
    """Return a unique logger name per test to avoid handler pollution.

    setup_logger() caches handlers on first call; if two tests share the
    same logger name, the second test's log_dir is ignored because the
    existing handlers still point to the first test's (now-deleted) tmpdir.
    Using uuid4() guarantees a fresh logger per test.
    """
    return f"ytclipper_test_{uuid.uuid4().hex}"


def test_setup_logger_creates_log_file(tmp_log_dir, clean_logger_name):
    """setup_logger should create app.log in the specified directory."""
    log = setup_logger(name=clean_logger_name, log_dir=tmp_log_dir)
    log.info("test message")
    
    log_file = Path(tmp_log_dir) / "app.log"
    assert log_file.exists(), "app.log should exist after logging"
    content = log_file.read_text(encoding="utf-8")
    assert "test message" in content


def test_logger_writes_to_app_log(tmp_log_dir, clean_logger_name):
    """logger instance should be able to write to app.log."""
    log = setup_logger(name=clean_logger_name, log_dir=tmp_log_dir)
    log.info("hello world")
    log_path = Path(tmp_log_dir) / "app.log"
    assert log_path.exists()
    assert "hello world" in log_path.read_text(encoding="utf-8")


def test_logger_handles_large_file_rotation(tmp_log_dir, clean_logger_name):
    """RotatingFileHandler should rotate log file when size exceeds limit."""
    log = setup_logger(name=clean_logger_name, log_dir=tmp_log_dir)
    
    # Write enough data to exceed 10MB
    large_line = "x" * 1024  # 1KB per line
    for _ in range(15 * 1024):  # 15MB
        log.info(large_line)
    
    log_path = Path(tmp_log_dir) / "app.log"
    backup_path = Path(tmp_log_dir) / "app.log.1"
    
    # After rotation, app.log.1 should exist
    assert log_path.exists()
    assert backup_path.exists(), "Log rotation should have created app.log.1"


def test_logger_redacts_api_key_value(tmp_log_dir, clean_logger_name):
    """logger should redact api_key values from log messages."""
    log = setup_logger(name=clean_logger_name, log_dir=tmp_log_dir)
    log.info("config api_key=sk-1234567890abcdefghij")
    
    log_path = Path(tmp_log_dir) / "app.log"
    content = log_path.read_text(encoding="utf-8")
    assert "sk-1234567890abcdefghij" not in content
    assert "REDACTED" in content


def test_logger_redacts_json_api_key(tmp_log_dir, clean_logger_name):
    """logger should redact 'api_key' in JSON-style config dumps."""
    log = setup_logger(name=clean_logger_name, log_dir=tmp_log_dir)
    secret = "sk-jso...7890"
    log.info(f'{{"api_key": "{secret}", "model": "gpt-4"}}')
    
    log_path = Path(tmp_log_dir) / "app.log"
    content = log_path.read_text(encoding="utf-8")
    assert secret not in content
    assert "REDACTED" in content


def test_logger_redacts_openai_sk_key(tmp_log_dir, clean_logger_name):
    """logger should redact raw sk- OpenAI style keys."""
    log = setup_logger(name=clean_logger_name, log_dir=tmp_log_dir)
    secret = "sk-proXXXXXXXXXXXXXXXXXXXX"
    log.info(f"Using {secret} for auth")
    
    log_path = Path(tmp_log_dir) / "app.log"
    content = log_path.read_text(encoding="utf-8")
    assert secret not in content
    assert "REDACTED" in content


def test_default_logger_is_singleton():
    """Default `logger` instance should be reusable and have handlers."""
    assert default_logger is not None
    assert isinstance(default_logger, logging.Logger)
    assert default_logger.name == "ytclipper"
    assert len(default_logger.handlers) > 0


def test_logger_supports_all_levels(tmp_log_dir, clean_logger_name):
    """Logger should support debug, info, warning, error."""
    log = setup_logger(name=clean_logger_name, log_dir=tmp_log_dir)
    log.debug("debug message")
    log.info("info message")
    log.warning("warning message")
    log.error("error message")
    
    log_path = Path(tmp_log_dir) / "app.log"
    content = log_path.read_text(encoding="utf-8")
    assert "debug message" in content
    assert "info message" in content
    assert "warning message" in content
    assert "error message" in content
