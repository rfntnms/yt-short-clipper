"""Full pipeline runner for a single clipping job.

The orchestrator is the only pipeline module that coordinates downloader,
transcriber, highlight detection, video processing, and caption burn-in.
It writes durable job metadata to output/<job_id>/data.json.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Generator

from pipeline import downloader, highlight_detector, transcriber, video_processor
from utils.logger import logger


class JobStatus(Enum):
    """Pipeline job lifecycle states."""

    PENDING = "PENDING"
    DOWNLOADING = "DOWNLOADING"
    TRANSCRIBING = "TRANSCRIBING"
    DETECTING_HIGHLIGHTS = "DETECTING_HIGHLIGHTS"
    PROCESSING_CLIPS = "PROCESSING_CLIPS"
    WRITING_OUTPUT = "WRITING_OUTPUT"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class PipelineError(Exception):
    """Raised when any pipeline step fails with mapped context."""

    def __init__(
        self,
        message: str,
        *,
        step: str | None = None,
        original: Exception | None = None,
    ) -> None:
        super().__init__(message)
        self.step = step
        self.original = original


@dataclass
class JobConfig:
    """Input configuration for one pipeline run."""

    url: str
    config: dict[str, Any]
    id: str = field(default_factory=lambda: uuid.uuid4().hex)


@dataclass
class JobResult:
    """Serializable job result metadata."""

    id: str
    url: str
    status: JobStatus
    clips: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None
    output_dir: str | None = None
    created_at: str = field(default_factory=lambda: _utc_now())
    updated_at: str = field(default_factory=lambda: _utc_now())

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["status"] = self.status.value
        return data


def run_job(job: JobConfig) -> JobResult:
    """Run the full pipeline for a single video job.

    Steps:
      1. download video
      2. transcribe to word-level JSON
      3. detect highlights
      4. cut, portrait-convert, and burn captions for each highlight
      5. write output/<job_id>/data.json

    Raises:
        PipelineError: If any mapped pipeline step fails. A FAILED data.json is
        still written before raising.
    """
    output_root = Path(job.config.get("output_dir", "output"))
    job_dir = output_root / job.id
    result = JobResult(
        id=job.id,
        url=job.url,
        status=JobStatus.PENDING,
        output_dir=str(job_dir),
    )

    try:
        # ── Step 1: Download ──────────────────────────────────────────────
        result.status = JobStatus.DOWNLOADING
        _write_data_json(job_dir, result)
        video_path, _srt_path = downloader.download(
            job.url,
            job_dir,
            _optional_path(job.config.get("cookies_path")),
        )

        # ── Step 2: Transcribe ───────────────────────────────────────────
        result.status = JobStatus.TRANSCRIBING
        _write_data_json(job_dir, result)
        words = transcriber.transcribe(video_path, job.config)

        # ── Step 3: Highlight Detection ──────────────────────────────────
        result.status = JobStatus.DETECTING_HIGHLIGHTS
        _write_data_json(job_dir, result)
        transcript_text = _words_to_transcript(words)
        highlights = highlight_detector.find_highlights(transcript_text, job.config)

        # ── Step 4: Process clips ────────────────────────────────────────
        result.status = JobStatus.PROCESSING_CLIPS
        _write_data_json(job_dir, result)
        for index, highlight in enumerate(highlights, start=1):
            raw_clip = str(job_dir / f"clip_{index:02d}_raw.mp4")
            portrait_clip = str(job_dir / f"clip_{index:02d}_portrait.mp4")

            cut_path = video_processor.cut(str(video_path), highlight, raw_clip)
            portrait_path = video_processor.convert_to_portrait(
                cut_path, job.config, portrait_clip
            )

            result.clips.append(
                {
                    "index": index,
                    "path": portrait_path,
                    "start": float(highlight.start),
                    "end": float(highlight.end),
                    "hook_text": str(highlight.hook_text),
                    "score": int(highlight.score),
                }
            )
            _write_data_json(job_dir, result)

        # ── Step 5: Finalize ─────────────────────────────────────────────
        result.status = JobStatus.WRITING_OUTPUT
        _write_data_json(job_dir, result)
        result.status = JobStatus.COMPLETED
        _write_data_json(job_dir, result)
        logger.info("Pipeline job completed: %s", job.id)
        return result

    except Exception as exc:
        mapped = _map_exception(exc)
        result.status = JobStatus.FAILED
        result.error = str(mapped)
        _write_data_json(job_dir, result)
        logger.error(
            "Pipeline job failed: %s step=%s error=%s",
            job.id,
            mapped.step,
            mapped,
        )
        raise mapped from exc


def run_job_streaming(
    url: str, config: dict[str, Any]
) -> Generator[dict[str, Any], None, JobResult]:
    """Run a job and yield status dictionaries for UI/event consumers.

    The current implementation emits deterministic start/end/error events while
    delegating execution to run_job(). Callers can iterate over this generator
    in Gradio or a batch runner.
    """
    job = JobConfig(url=url, config=config)
    yield {"job_id": job.id, "status": JobStatus.PENDING.value}
    try:
        result = run_job(job)
    except PipelineError as exc:
        yield {
            "job_id": job.id,
            "status": JobStatus.FAILED.value,
            "error": str(exc),
        }
        raise
    yield {
        "job_id": job.id,
        "status": result.status.value,
        "clips": result.clips,
    }
    return result


# ── Helpers ──────────────────────────────────────────────────────────────────


def _write_data_json(job_dir: Path, result: JobResult) -> None:
    job_dir.mkdir(parents=True, exist_ok=True)
    result.updated_at = _utc_now()
    data_file = job_dir / "data.json"
    data_file.write_text(
        json.dumps(result.to_dict(), indent=2), encoding="utf-8"
    )


def _map_exception(exc: Exception) -> PipelineError:
    if isinstance(exc, PipelineError):
        return exc

    mappings: tuple[tuple[type[Exception], str], ...] = (
        (downloader.DownloadError, "download"),
        (transcriber.TranscriptionError, "transcription"),
        (highlight_detector.HighlightDetectionError, "highlight_detection"),
        (video_processor.VideoProcessingError, "video_processing"),
    )
    for error_type, step in mappings:
        if isinstance(exc, error_type):
            return PipelineError(
                f"{step} failed: {exc}", step=step, original=exc
            )
    return PipelineError(
        f"pipeline failed: {exc}", step="pipeline", original=exc
    )


def _optional_path(value: Any) -> Path | None:
    if not value:
        return None
    return Path(str(value))


def _words_to_transcript(words: list[dict[str, Any]]) -> str:
    return " ".join(
        str(word.get("word", "")).strip() for word in words
    ).strip()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


__all__ = [
    "JobConfig",
    "JobResult",
    "JobStatus",
    "PipelineError",
    "run_job",
    "run_job_streaming",
]
