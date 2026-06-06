"""Video processing module — FFmpeg-based clip cutting and portrait conversion.

Fourth step in the orchestrator pipeline.
Two public functions:
  * cut()           — extract a clip using seek + stream copy
  * convert_to_portrait() — crop to 9:16 center, scale to 1080x1920

All external calls go via subprocess (no ffmpeg-python dependency).
GPU hwaccel flags are injected when utils/gpu_detector reports CUDA + nvenc.
"""

from __future__ import annotations

import math
import shutil
import subprocess
from pathlib import Path
from typing import Any

from pipeline.highlight_detector import Highlight
from utils.gpu_detector import detect_cuda, get_gpu_flags
from utils.logger import logger

# ── constants ────────────────────────────────────────────────────────────
_FFMPEG: str = shutil.which("ffmpeg") or "ffmpeg"
_FFPROBE: str = shutil.which("ffprobe") or "ffprobe"
_OUTPUT_WIDTH: int = 1080
_OUTPUT_HEIGHT: int = 1920
_ASPECT_NUM: int = 9   # width ratio
_ASPECT_DEN: int = 16  # height ratio


class VideoProcessingError(Exception):
    """Raised when an FFmpeg/FFprobe operation fails."""


# ── public API ───────────────────────────────────────────────────────────


def cut(
    video_path: str,
    highlight: Highlight,
    output_path: str,
) -> str:
    """Extract a clip from a video between highlight.start and highlight.end.

    Uses fast seek (-ss before -c copy) for speed.

    Args:
        video_path: Path to the source video (mp4).
        highlight: Highlight dataclass with start/end timestamps.
        output_path: Destination path for the clip (should end in .mp4).

    Returns:
        output_path on success.

    Raises:
        VideoProcessingError: If timestamps are invalid or FFmpeg fails.
    """
    _validate_timestamps(highlight.start, highlight.end)
    duration = highlight.end - highlight.start

    cmd: list[str] = [
        _FFMPEG,
        "-y",                    # overwrite output
        "-ss", str(highlight.start),  # seek before input (fast)
        "-i", video_path,
        "-t", str(duration),
        "-c", "copy",
        output_path,
    ]

    _run_ffmpeg(cmd, context=f"cut({highlight.start}s..{highlight.end}s)")
    return output_path


def convert_to_portrait(
    clip_path: str,
    config: dict[str, Any],
    output_path: str,
) -> str:
    """Convert a clip to 9:16 portrait (1080x1920) with center crop.

    SINGLE mode only in MVP (Milestone 2).
    Detects input resolution via ffprobe, computes a center crop that yields
    a 9:16 region, then scales to 1080x1920.
    Applies GPU hwaccel flags if CUDA + h264_nvenc are available.

    Args:
        clip_path: Path to the source clip (mp4).
        config: Full application config dict. Only "portrait" sub-dict is used.
        output_path: Destination path. Always 1080x1920 regardless of input.

    Returns:
        output_path on success.

    Raises:
        VideoProcessingError: If ffprobe or FFmpeg fails.
    """
    # Probe input resolution
    width, height = probe_dimensions(clip_path)

    # Compute center crop for 9:16 aspect
    cw, ch, cx, cy = compute_center_crop(width, height)

    # GPU flags
    gpu_info = detect_cuda()
    flags = get_gpu_flags(gpu_info)
    logger.info("Portrait conversion: %s", flags["description"])

    # Build filter: crop center, then scale to 1080x1920
    # scale=-2:1920 keeps aspect; with explicit crop we know it's 9:16
    # Using 1080:-2 ensures width is exactly 1080, height adjusts
    # But we want strictly 1080x1920 so we pad if needed after crop
    # use force_original_aspect_ratio inside scale to pad black if crop != exact
    vf = (
        f"crop={cw}:{ch}:{cx}:{cy},"
        f"scale={_OUTPUT_WIDTH}:{_OUTPUT_HEIGHT}:force_original_aspect_ratio=decrease,"
        f"pad={_OUTPUT_WIDTH}:{_OUTPUT_HEIGHT}:(ow-iw)/2:(oh-ih)/2"
    )

    hwaccel = flags.get("hwaccel", [])
    encoder = flags.get("encoder", "libx264")

    # Quality flag must match the encoder family:
    #   libx264      → -crf (constant rate factor)
    #   h264_nvenc   → -cq  (constant quality; -crf is rejected)
    if encoder == "h264_nvenc":
        quality_args = ["-cq", "23"]
    else:
        quality_args = ["-crf", "23"]

    cmd: list[str] = [
        _FFMPEG,
        "-y",
        *hwaccel,
        "-i", clip_path,
        "-vf", vf,
        "-c:v", encoder,
        "-preset", "medium",
        *quality_args,
        "-c:a", "copy",
        output_path,
    ]

    _run_ffmpeg(cmd, context=f"portrait({clip_path})")
    return output_path


# ── helpers ──────────────────────────────────────────────────────────────


def probe_dimensions(video_path: str) -> tuple[int, int]:
    """Get video width and height via ffprobe.

    Raises:
        VideoProcessingError: If ffprobe fails or dimensions cannot be parsed.
    """
    cmd: list[str] = [
        _FFPROBE,
        "-v", "quiet",
        "-print_format", "json",
        "-show_streams",
        "-select_streams", "v:0",
        video_path,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30, check=False)
    except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
        raise VideoProcessingError(f"ffprobe failed: {exc}") from exc

    if proc.returncode != 0:
        raise VideoProcessingError(
            f"ffprobe failed (rc={proc.returncode}): {proc.stderr.strip()}"
        )

    import json

    try:
        data = json.loads(proc.stdout)
        streams = data.get("streams", [])
        if not streams:
            raise VideoProcessingError(f"no video streams found in {video_path}")

        w = int(streams[0]["width"])
        h = int(streams[0]["height"])
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise VideoProcessingError(
            f"could not parse ffprobe dimensions for {video_path}: {exc}"
        ) from exc

    return w, h


def compute_center_crop(
    src_width: int, src_height: int
) -> tuple[int, int, int, int]:
    """Compute a center crop region that matches a 9:16 aspect ratio.

    Returns (crop_w, crop_h, crop_x, crop_y).
    The crop is always centered horizontally. Vertical offset is 0
    (anchors top) unless the source is already 9:16 or taller, in which
    case the full height is kept.
    """
    # Target aspect: 9/16
    src_aspect = src_width / src_height
    target_aspect = _ASPECT_NUM / _ASPECT_DEN  # 9/16 ≈ 0.5625

    if src_aspect > target_aspect:
        # Source is wider than 9:16 → crop width
        crop_h = src_height
        crop_w = int(src_height * target_aspect)
    else:
        # Source is taller than 9:16 → crop height
        crop_w = src_width
        crop_h = int(src_width / target_aspect)

    # Round down to even values — required for YUV 4:2:0 pixel formats.
    crop_w = max(2, crop_w - (crop_w % 2))
    crop_h = max(2, crop_h - (crop_h % 2))

    crop_x = (src_width - crop_w) // 2
    crop_y = (src_height - crop_h) // 2

    return crop_w, crop_h, crop_x, crop_y


# ── private utilities ────────────────────────────────────────────────────


def _validate_timestamps(start: float, end: float) -> None:
    """Validate that start/end form a valid, non-negative time range.

    Raises:
        VideoProcessingError: If values are non-finite, negative, or inverted.
    """
    if not (math.isfinite(start) and math.isfinite(end)):
        raise VideoProcessingError(
            f"Timestamps must be finite: start={start}, end={end}"
        )
    if start < 0:
        raise VideoProcessingError(
            f"Start timestamp must be non-negative: {start}"
        )
    if end <= start:
        raise VideoProcessingError(
            f"End must be > start: {start} >= {end}"
        )


def _run_ffmpeg(cmd: list[str], context: str) -> None:
    """Execute an FFmpeg command and raise VideoProcessingError on failure.

    No shell=True — all arguments are properly escaped by subprocess.
    """
    logger.info("Running FFmpeg: %s", " ".join(cmd))
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
        raise VideoProcessingError(
            f"FFmpeg execution failed ({context}): {exc}"
        ) from exc

    if proc.returncode != 0:
        stderr_tail = proc.stderr.strip().splitlines()
        # last 5 lines of stderr for error message
        detail = "\n".join(stderr_tail[-5:]) if stderr_tail else "no stderr"
        raise VideoProcessingError(
            f"FFmpeg failed ({context}): rc={proc.returncode}, stderr:\n{detail}"
        )


__all__ = [
    "VideoProcessingError",
    "cut",
    "convert_to_portrait",
    "compute_center_crop",
    "probe_dimensions",
]
