"""Whisper transcription module.

Second step in the orchestrator pipeline.
Calls Whisper endpoint via providers/ai_client (ADR-003 compliant).
Returns word-level JSON: [{"word": str, "start": float, "end": float}, ...]
"""
import json
from pathlib import Path
from typing import Any

from providers.ai_client import get_client
from utils.logger import logger


class TranscriptionError(Exception):
    """Raised when transcription fails (invalid file, API error)."""


def transcribe(
    video_path: Path,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    """Transcribe a video file using Whisper via OpenAI-compatible endpoint.

    Args:
        video_path: Path to the video/audio file.
        config: Full application config dict. Uses config["transcription"].

    Returns:
        List of dicts: [{"word": str, "start": float, "end": float}, ...]

    Raises:
        TranscriptionError: If the file doesn't exist or the API call fails.
    """
    if not video_path.exists():
        raise TranscriptionError(f"Video file not found: {video_path}")

    transcription_cfg: dict[str, Any] = config.get("transcription", {})
    model: str = transcription_cfg.get("model", "whisper-1")

    # Skip expensive API call if downloader already produced captions.
    expected_srt = video_path.with_suffix(".srt")
    if expected_srt.exists():
        logger.info("Skipping transcription because SRT already exists: %s", expected_srt.name)
        return []

    # Reuse the generic OpenAI-compatible client factory (ADR-003).
    # get_client() currently reads config["llm"], so adapt transcription config into
    # that shape without importing openai directly here.
    client = get_client({"llm": transcription_cfg})

    logger.info("Starting transcription for %s (model=%s)", video_path.name, model)

    try:
        with open(video_path, "rb") as audio_file:
            response = client.audio.transcriptions.create(
                model=model,
                file=audio_file,
                response_format="verbose_json",
                timestamp_granularities=["word"],
            )
    except Exception as exc:
        logger.error("Transcription failed for %s: %s", video_path.name, exc)
        raise TranscriptionError(str(exc)) from exc

    # Build word-level JSON from Whisper response.
    # Whisper verbose_json may expose words via response.words or response.segments.
    words: list[dict[str, Any]] = []
    raw_words = getattr(response, "words", None)
    if raw_words:
        for w in raw_words:
            # Support both attribute-style (dataclass) and dict-style access
            if isinstance(w, dict):
                word_text: str = w.get("word", "")
                start: float = float(w.get("start", 0.0))
                end: float = float(w.get("end", 0.0))
            else:
                word_text = getattr(w, "word", "")
                start = float(getattr(w, "start", 0.0))
                end = float(getattr(w, "end", 0.0))
            words.append({"word": word_text.strip(), "start": start, "end": end})

    logger.info("Transcription complete: %d words", len(words))
    return words


__all__ = ["transcribe", "TranscriptionError"]
