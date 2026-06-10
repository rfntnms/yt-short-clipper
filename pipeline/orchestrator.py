"""Single-video pipeline orchestrator.

Integrates downloader, transcriber, highlight_detector, video_processor, and
caption_generator into one job flow with streamable status updates.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Generator, Optional

from pipeline.downloader import download
from pipeline.highlight_detector import Highlight, find_highlights
from pipeline.transcriber import transcribe
from pipeline.video_processor import convert_to_portrait, cut
from utils.logger import logger


def _generate_and_burn(clip_path: str, word_json: list[dict[str, Any]], config: dict[str, Any]) -> str:
    from pipeline.caption_generator import generate_and_burn

    return generate_and_burn(clip_path, word_json, config)


class OrchestrationError(Exception):
    """Raised when a full pipeline job fails."""


@dataclass
class JobConfig:
    """Input config for a single video pipeline job."""

    url: str
    job_id: str
    output_dir: str | Path = "output"
    config: dict[str, Any] = field(default_factory=dict)
    cookies_path: Optional[str | Path] = None
    force_transcribe: bool = False


@dataclass
class JobStatus:
    """Streamed job status for UI/batch consumers."""

    status: str
    progress: float
    message: str
    job_id: str
    output_clips: list[str] = field(default_factory=list)
    error: Optional[str] = None


def run_job_streaming(job: JobConfig) -> Generator[JobStatus, None, None]:
    """Run the full single-video pipeline and yield progress updates.

    Flow:
    1. download video + optional subtitle
    2. transcribe when subtitle is missing or forced
    3. detect highlights
    4. cut, portrait-convert, and caption every highlight
    5. write output/<job_id>/data.json
    """
    job_dir = Path(job.output_dir) / job.job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Starting pipeline job %s", job.job_id)

    try:
        yield _status(job, 0.05, "Downloading video")
        video_path, subtitle_path = download(
            job.url,
            job_dir,
            Path(job.cookies_path) if job.cookies_path else None,
        )

        yield _status(job, 0.25, "Preparing transcript")
        word_json: list[dict[str, Any]] = []
        if subtitle_path is not None and not job.force_transcribe:
            transcript_text = Path(subtitle_path).read_text(encoding="utf-8")
            word_json = transcribe(Path(video_path), job.config)
        else:
            word_json = transcribe(Path(video_path), job.config)
            transcript_text = _words_to_transcript(word_json)

        yield _status(job, 0.45, "Detecting highlights")
        highlights = find_highlights(transcript_text, job.config)
        if not highlights:
            raise OrchestrationError("No highlights detected")

        output_clips: list[str] = []
        for index, highlight in enumerate(highlights, start=1):
            base_progress = 0.45 + (0.45 * (index - 1) / len(highlights))
            yield _status(job, base_progress, f"Processing highlight {index}/{len(highlights)}")

            raw_clip = job_dir / f"clip_{index:02d}_raw.mp4"
            portrait_clip = job_dir / f"clip_{index:02d}_portrait.mp4"

            cut(str(video_path), highlight, str(raw_clip))
            convert_to_portrait(str(raw_clip), job.config, str(portrait_clip))

            clip_word_data = _slice_word_data(word_json, highlight.start, highlight.end)
            final_clip = _generate_and_burn(str(portrait_clip), clip_word_data, job.config)
            output_clips.append(final_clip)

        yield _status(job, 0.95, "Writing job metadata", output_clips)
        _write_metadata(job_dir, job, video_path, subtitle_path, highlights, output_clips)

        yield _status(job, 1.0, "Job completed", output_clips, status="DONE")
    except Exception as exc:
        logger.error("Pipeline job %s failed: %s", job.job_id, exc, exc_info=True)
        yield _status(job, 1.0, "Job failed", status="FAILED", error=str(exc))


def run_job(job: JobConfig) -> JobStatus:
    """Run a job to completion and return the terminal status."""
    terminal_status: JobStatus | None = None
    for status in run_job_streaming(job):
        terminal_status = status
    if terminal_status is None:
        raise OrchestrationError("Job produced no status updates")
    return terminal_status


class Orchestrator:
    """Small compatibility wrapper around module-level orchestration API."""

    def run_job_streaming(self, job: JobConfig) -> Generator[JobStatus, None, None]:
        yield from run_job_streaming(job)

    def run_job(self, job: JobConfig) -> JobStatus:
        return run_job(job)


def _status(
    job: JobConfig,
    progress: float,
    message: str,
    output_clips: Optional[list[str]] = None,
    status: str = "RUNNING",
    error: Optional[str] = None,
) -> JobStatus:
    return JobStatus(
        status=status,
        progress=progress,
        message=message,
        job_id=job.job_id,
        output_clips=output_clips or [],
        error=error,
    )


def _words_to_transcript(words: list[dict[str, Any]]) -> str:
    return " ".join(str(word.get("word", "")).strip() for word in words).strip()


def _slice_word_data(
    word_json: list[dict[str, Any]],
    clip_start: float,
    clip_end: float,
) -> list[dict[str, Any]]:
    """Filter word-level data to the clip's time range and offset timestamps.

    Keeps words whose time range overlaps [clip_start, clip_end) and
    offsets their timestamps so they are relative to the clip start.
    """
    sliced: list[dict[str, Any]] = []
    for word in word_json:
        w_start = word.get("start", 0.0)
        w_end = word.get("end", 0.0)
        if w_start < clip_end and w_end > clip_start:
            sliced.append({
                "word": word.get("word", ""),
                "start": max(w_start - clip_start, 0.0),
                "end": max(min(w_end, clip_end) - clip_start, 0.0),
            })
    return sliced


def _write_metadata(
    job_dir: Path,
    job: JobConfig,
    video_path: str | Path,
    subtitle_path: str | Path | None,
    highlights: list[Highlight],
    output_clips: list[str],
) -> None:
    metadata = {
        "job_id": job.job_id,
        "url": job.url,
        "video_path": str(video_path),
        "subtitle_path": str(subtitle_path) if subtitle_path else None,
        "output_clips": output_clips,
        "highlights": [asdict(highlight) for highlight in highlights],
    }
    (job_dir / "data.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


__all__ = [
    "JobConfig",
    "JobStatus",
    "OrchestrationError",
    "Orchestrator",
    "run_job",
    "run_job_streaming",
]
