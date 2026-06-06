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


# ── test 5: aggregates words from segments[].words when top-level empty ──
@patch("pipeline.transcriber.get_client")
def test_aggregates_words_from_segments(
    mock_get_client: MagicMock, video_path: Path
) -> None:
    """Covers OpenAI-compatible providers (faster-whisper-server, whisper.cpp,
    Groq whisper-turbo) that return words nested in segments instead of
    top-level response.words."""
    mock_client = mock_get_client.return_value
    mock_client.audio.transcriptions.create.return_value = MagicMock(
        text="hello world",
        words=None,  # top-level words missing
        segments=[
            MagicMock(
                id=0,
                words=[
                    MagicMock(word="hello", start=0.0, end=0.5),
                    MagicMock(word="world", start=0.5, end=1.0),
                ],
            ),
        ],
    )

    config = {"transcription": {"base_url": "http://localhost:9999/v1", "model": "whisper-1", "api_key": "test"}}
    result = transcribe(video_path, config)

    assert len(result) == 2
    assert result[0]["word"] == "hello"
    assert result[1]["word"] == "world"


# ── test 6: handles dict-style response with segments ───────────────────
@patch("pipeline.transcriber.get_client")
def test_handles_dict_response_with_segments(
    mock_get_client: MagicMock, video_path: Path
) -> None:
    """Some OpenAI-compatible proxy/echo setups return raw dicts."""
    mock_client = mock_get_client.return_value
    mock_client.audio.transcriptions.create.return_value = {
        "text": "foo bar",
        "words": None,
        "segments": [
            {"id": 0, "words": [{"word": "foo", "start": 0.0, "end": 0.3}]},
            {"id": 1, "words": [{"word": "bar", "start": 0.3, "end": 0.7}]},
        ],
    }

    config = {"transcription": {"base_url": "http://localhost:9999/v1", "model": "whisper-1", "api_key": "***"}}
    result = transcribe(video_path, config)

    assert len(result) == 2
    assert result[0]["word"] == "foo"
    assert result[1]["word"] == "bar"


# ── test 7: existing SRT does NOT skip word-level transcription ─────────
@patch("pipeline.transcriber.get_client")
def test_existing_srt_does_not_skip_whisper(
    mock_get_client: MagicMock, video_path: Path
) -> None:
    """Contract guard: even if downloader produced an .srt, transcribe must
    still call Whisper to obtain word-level JSON. Returning [] would break
    downstream highlight/caption stages that expect timed word data."""
    # Pre-create a sibling .srt file the way the downloader would.
    srt_path = video_path.with_suffix(".srt")
    srt_path.write_text("1\n00:00:00,000 --> 00:00:01,000\nhello world\n")

    mock_client = mock_get_client.return_value
    mock_client.audio.transcriptions.create.return_value = MagicMock(
        text="hello world",
        words=[
            MagicMock(word="hello", start=0.0, end=0.5),
            MagicMock(word="world", start=0.5, end=1.0),
        ],
    )

    config = {"transcription": {"base_url": "http://localhost:9999/v1", "model": "whisper-1", "api_key": "***"}}
    result = transcribe(video_path, config)

    # Whisper was called despite the SRT being on disk.
    mock_client.audio.transcriptions.create.assert_called_once()
    assert len(result) == 2
    assert result[0]["word"] == "hello"


# ── test 8: missing word timestamps raises TranscriptionError ───────────
@patch("pipeline.transcriber.get_client")
def test_raises_when_no_word_timestamps(
    mock_get_client: MagicMock, video_path: Path
) -> None:
    """If the provider succeeds but returns neither top-level words nor
    segments[].words, surface a typed error instead of silently returning []."""
    mock_client = mock_get_client.return_value
    mock_client.audio.transcriptions.create.return_value = MagicMock(
        text="...",
        words=None,           # no top-level words
        segments=[            # segments exist but with no words
            MagicMock(id=0, words=None),
        ],
    )

    config = {"transcription": {"base_url": "http://localhost:9999/v1", "model": "whisper-1", "api_key": "***"}}

    with pytest.raises(TranscriptionError, match="word-level timestamps"):
        transcribe(video_path, config)


# ── test 9: invalid word entry (missing start) raises TranscriptionError ─
@patch("pipeline.transcriber.get_client")
def test_raises_on_invalid_word_missing_start(
    mock_get_client: MagicMock, video_path: Path
) -> None:
    """A word dict with start=None must not silently become 0.0."""
    mock_client = mock_get_client.return_value
    mock_client.audio.transcriptions.create.return_value = MagicMock(
        text="hi",
        words=[
            MagicMock(word="hi", start=None, end=1.0),
        ],
    )

    config = {"transcription": {"base_url": "http://localhost:9999/v1", "model": "whisper-1", "api_key": "***"}}

    with pytest.raises(TranscriptionError, match="invalid word timestamps"):
        transcribe(video_path, config)


# ── test 10: invalid word entry (end < start) raises TranscriptionError ──
@patch("pipeline.transcriber.get_client")
def test_raises_on_invalid_word_timestamp_order(
    mock_get_client: MagicMock, video_path: Path
) -> None:
    """end < start is a corrupt timestamp; raise rather than propagate."""
    mock_client = mock_get_client.return_value
    mock_client.audio.transcriptions.create.return_value = MagicMock(
        text="hi",
        words=[
            MagicMock(word="hi", start=2.0, end=1.0),
        ],
    )

    config = {"transcription": {"base_url": "http://localhost:9999/v1", "model": "whisper-1", "api_key": "***"}}

    with pytest.raises(TranscriptionError, match="invalid word timestamp order"):
        transcribe(video_path, config)


# ── test 11: get_client called with sanitized config (no model leak) ─────
@patch("pipeline.transcriber.get_client")
def test_get_client_receives_only_base_url_and_api_key(
    mock_get_client: MagicMock, video_path: Path
) -> None:
    """Ensure transcription_cfg.model does NOT leak into get_client's llm dict.
    get_client() should receive only base_url + api_key so transcription model
    is never interpreted as an LLM model."""
    mock_client = mock_get_client.return_value
    mock_client.audio.transcriptions.create.return_value = MagicMock(
        text="ok",
        words=[MagicMock(word="ok", start=0.0, end=1.0)],
    )

    config = {
        "transcription": {
            "base_url": "http://localhost:9999/v1",
            "model": "whisper-1",
            "api_key": "test-key",
        }
    }
    transcribe(video_path, config)

    mock_get_client.assert_called_once()
    call_arg = mock_get_client.call_args[0]
    llm_cfg = call_arg[0]["llm"]
    assert "model" not in llm_cfg, "model field must not leak into get_client llm config"
    assert llm_cfg["base_url"] == "http://localhost:9999/v1"
    assert llm_cfg["api_key"] == "test-key"


# ── test 12: non-numeric timestamp raises TranscriptionError (not ValueError) ─
@patch("pipeline.transcriber.get_client")
def test_raises_on_non_numeric_timestamp(
    mock_get_client: MagicMock, video_path: Path
) -> None:
    """float('unknown') raises ValueError; we must surface a typed
    TranscriptionError so callers see a consistent exception type."""
    mock_client = mock_get_client.return_value
    mock_client.audio.transcriptions.create.return_value = MagicMock(
        text="hi",
        words=[
            MagicMock(word="hi", start="unknown", end=1.0),
        ],
    )

    config = {"transcription": {"base_url": "http://localhost:9999/v1", "model": "whisper-1", "api_key": "***"}}

    with pytest.raises(TranscriptionError, match="non-numeric word timestamps"):
        transcribe(video_path, config)


# ── test 13: NaN / Inf timestamps rejected ───────────────────────────────
@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")])
@patch("pipeline.transcriber.get_client")
def test_raises_on_non_finite_timestamp(
    mock_get_client: MagicMock, video_path: Path, bad_value: float
) -> None:
    """NaN/Inf timestamps silently passed through float() and would break
    downstream highlight/caption logic. They must raise TranscriptionError."""
    mock_client = mock_get_client.return_value
    mock_client.audio.transcriptions.create.return_value = MagicMock(
        text="hi",
        words=[
            MagicMock(word="hi", start=bad_value, end=1.0),
        ],
    )

    config = {"transcription": {"base_url": "http://localhost:9999/v1", "model": "whisper-1", "api_key": "***"}}

    with pytest.raises(TranscriptionError, match="non-finite word timestamps"):
        transcribe(video_path, config)
