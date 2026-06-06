"""Unit tests for pipeline/transcriber.py."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pipeline.transcriber import TranscriptionError, transcribe


@pytest.fixture
def video_path(tmp_path: Path) -> Path:
    v = tmp_path / "test_video.mp4"
    v.write_bytes(b"\x00" * 1024)
    return v


# ── test 1: returns word-level JSON on success ──────────────────────────
@patch("pipeline.transcriber.get_client")
def test_transcribe_returns_word_level_json(
    mock_get_client: MagicMock, video_path: Path, tmp_path: Path
) -> None:
    mock_client = mock_get_client.return_value
    mock_client.audio.transcriptions.create.return_value = MagicMock(
        text="hello world",
        words=[
            MagicMock(word="hello", start=0.0, end=0.5),
            MagicMock(word="world", start=0.5, end=1.0),
        ],
    )

    srt_path = video_path.with_suffix(".srt")
    config = {"transcription": {"base_url": "http://localhost:9999/v1", "model": "whisper-1", "api_key": "test"}}

    result = transcribe(video_path, config)

    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0]["word"] == "hello"
    assert result[0]["start"] == 0.0
    assert result[0]["end"] == 0.5


# ── test 2: raises TranscriptionError on API failure ────────────────────
@patch("pipeline.transcriber.get_client")
def test_raises_on_api_failure(mock_get_client: MagicMock, video_path: Path) -> None:
    mock_client = mock_get_client.return_value
    mock_client.audio.transcriptions.create.side_effect = Exception("401 Unauthorized")

    config = {"transcription": {"base_url": "http://localhost:9999/v1", "model": "whisper-1", "api_key": "bad"}}

    with pytest.raises(TranscriptionError, match="401 Unauthorized"):
        transcribe(video_path, config)


# ── test 3: raises on invalid/missing video file ────────────────────────
def test_raises_on_missing_file(tmp_path: Path) -> None:
    ghost = tmp_path / "nonexistent.mp4"
    config = {"transcription": {"base_url": "http://localhost:9999/v1", "model": "whisper-1", "api_key": "test"}}

    with pytest.raises(TranscriptionError, match="not found"):
        transcribe(ghost, config)


# ── test 4: uses ai_client.get_client (no direct openai import) ─────────
@patch("pipeline.transcriber.get_client")
def test_uses_ai_client(mock_get_client: MagicMock, video_path: Path) -> None:
    mock_client = mock_get_client.return_value
    mock_client.audio.transcriptions.create.return_value = MagicMock(
        text="ok",
        words=[MagicMock(word="ok", start=0.0, end=1.0)],
    )

    config = {"transcription": {"base_url": "http://localhost:9999/v1", "model": "whisper-1", "api_key": "test"}}
    transcribe(video_path, config)

    mock_get_client.assert_called_once()
