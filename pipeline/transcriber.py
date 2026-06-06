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

    # Downloader may have produced an .srt, but we still need word-level
    # JSON so downstream steps (highlight_detector, caption_generator) get
    # timed word data. Log the skip candidate but continue to Whisper call.
    expected_srt = video_path.with_suffix(".srt")
    if expected_srt.exists():
        logger.info("Existing SRT found, but word-level transcription is still required: %s", expected_srt.name)

    # Reuse the generic OpenAI-compatible client factory (ADR-003).
    # get_client() only reads llm.base_url and llm.api_key, but pass only
    # those keys explicitly so a transcription-only "model" field cannot
    # leak into client construction if get_client()'s contract expands.
    client = get_client(
        {
            "llm": {
                "base_url": transcription_cfg.get("base_url"),
                "api_key": transcription_cfg.get("api_key", "ollama"),
            }
        }
    )

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
    # Whisper verbose_json may expose words via response.words (top-level) or
    # nested inside response.segments[].words. The OpenAI Whisper API returns
    # top-level words when timestamp_granularities=["word"], but several
    # OpenAI-compatible providers (faster-whisper-server, whisper.cpp, Groq
    # whisper-turbo) only return words nested in segments. Normalize both
    # shapes so we never silently return an empty list on a successful call.
    if isinstance(response, dict):
        raw_words = response.get("words")
        raw_segments = response.get("segments", []) or []
    else:
        raw_words = getattr(response, "words", None)
        raw_segments = getattr(response, "segments", []) or []

    if not raw_words:
        # Fall back: aggregate words from segments[].words
        aggregated: list[Any] = []
        for segment in raw_segments:
            if isinstance(segment, dict):
                segment_words = segment.get("words", []) or []
            else:
                segment_words = getattr(segment, "words", []) or []
            aggregated.extend(segment_words)
        raw_words = aggregated

    if not raw_words:
        # Both top-level response.words and segments[].words were missing.
        # Surface a typed error so callers do not proceed with an apparently
        # successful but unusable transcription (no word timestamps to drive
        # highlight detection or caption generation).
        raise TranscriptionError(
            "Transcription response did not include word-level timestamps"
        )

    words: list[dict[str, Any]] = []
    for w in raw_words or []:
        # Support both attribute-style (dataclass) and dict-style access
        if isinstance(w, dict):
            word_text = str(w.get("word", "")).strip()
            raw_start = w.get("start")
            raw_end = w.get("end")
        else:
            word_text = str(getattr(w, "word", "")).strip()
            raw_start = getattr(w, "start", None)
            raw_end = getattr(w, "end", None)

        if not word_text or raw_start is None or raw_end is None:
            raise TranscriptionError(
                "Transcription response contained invalid word timestamps"
            )

        start = float(raw_start)
        end = float(raw_end)
        if end < start:
            raise TranscriptionError(
                "Transcription response contained invalid word timestamp order"
            )

        words.append({"word": word_text, "start": start, "end": end})

    logger.info("Transcription complete: %d words", len(words))
    return words


__all__ = ["transcribe", "TranscriptionError"]
