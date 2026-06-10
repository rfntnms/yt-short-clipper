"""Test script for utils/logger.py — RFN-27 verification"""
import os
import sys
import shutil
import logging

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import setup_logger

# Import creates the global logger and adds handlers pointing to cwd.
# Clear them so we can test setup_logger with a custom log_dir.
logging.getLogger("ytclipper").handlers.clear()

TEST_LOG_DIR = "/tmp/test_rfn27_logger"
if os.path.exists(TEST_LOG_DIR):
    shutil.rmtree(TEST_LOG_DIR)
os.makedirs(TEST_LOG_DIR, exist_ok=True)

# Test 1: Create logger with explicit log_dir
log = setup_logger("ytclipper", log_dir=TEST_LOG_DIR)

log.debug("Debug: system initialized")
log.info("Info: Pipeline started for video_id=abc123")
log.warning("Warning: GPU not found, falling back to CPU")
log.error("Error: Failed to transcribe video - timeout after 30s")

# Test 2: Sensitive data redaction
log.info("Using api_key=sk-abc...0000 for OpenAI")
log.info('Config: {"api_key": "sk-pro...cdef"}')

# Test 3: Verify file output
log_file = os.path.join(TEST_LOG_DIR, "app.log")
assert os.path.exists(log_file), f"FAIL: app.log not created at {log_file}"

with open(log_file, "r", encoding="utf-8") as f:
    content = f.read()

lines = content.strip().splitlines()
print(f"Log file: {log_file}")
print(f"File size: {os.path.getsize(log_file)} bytes")
print(f"Log lines: {len(lines)}")
print("--- File contents ---")
print(content)

# Assertions
assert len(lines) >= 6, f"FAIL: expected >= 6 log lines, got {len(lines)}"
assert "***REDACTED***" in content, "FAIL: sensitive data not redacted"
assert "sk-abc123" not in content, "FAIL: raw API key leaked in logs"
assert "Pipeline started" in content, "FAIL: info message missing"
assert "ytclipper" in content, "FAIL: logger name missing"

# Test 4: Verify rotation config (10MB, 3 backups)
from logging.handlers import RotatingFileHandler
found_rotating = False
for h in log.handlers:
    if isinstance(h, RotatingFileHandler):
        assert h.maxBytes == 10 * 1024 * 1024, f"FAIL: maxBytes={h.maxBytes}, expected 10485760"
        assert h.backupCount == 3, f"FAIL: backupCount={h.backupCount}, expected 3"
        print(f"Rotation: maxBytes={h.maxBytes} (10MB), backupCount={h.backupCount}")
        found_rotating = True
        break
assert found_rotating, "FAIL: No RotatingFileHandler found"

# Test 5: Verify console handler exists
found_console = any(isinstance(h, logging.StreamHandler) and not isinstance(h, RotatingFileHandler) for h in log.handlers)
assert found_console, "FAIL: No console handler found"

# Test 6: Verify logger properties
assert log.name == "ytclipper", f"FAIL: logger name is {log.name}"
assert log.level == logging.DEBUG, f"FAIL: logger level is {log.level}"
assert log.propagate is False, "FAIL: propagate should be False"
print(f"Logger: name={log.name}, level=DEBUG, handlers={len(log.handlers)}, propagate=False")

# Test 7: Verify singleton behavior (second call returns same logger)
log2 = setup_logger("ytclipper", log_dir="/some/other/path")
assert log2 is log, "FAIL: setup_logger should return same instance"
print("Singleton: OK (second call returns cached instance)")

# Cleanup
shutil.rmtree(TEST_LOG_DIR, ignore_errors=True)
print("\n=== ALL CHECKS PASSED ===")
