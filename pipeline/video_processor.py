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
import tempfile
from pathlib import Path
from typing import Any

from pipeline.highlight_detector import Highlight
from pipeline import speaker_layout
from pipeline.speaker_layout import LayoutMode
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

    # Use output-side seek + re-encode for frame-accurate cuts.
    # Input-side -ss with -c copy is fast but keyframe-inaccurate.
    cmd: list[str] = [
        _FFMPEG,
        "-y",
        "-i", video_path,
        "-ss", str(highlight.start),
        "-t", str(duration),
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-crf", "18",
        "-c:a", "aac",
        "-b:a", "128k",
        output_path,
    ]

    _run_ffmpeg(cmd, context=f"cut({highlight.start}s..{highlight.end}s)")
    return output_path


def convert_to_portrait(
    clip_path: str,
    config: dict[str, Any],
    output_path: str,
) -> str:
    """Convert a clip to 9:16 portrait (1080x1920).

    Uses speaker_layout.analyze() to detect speakers and decide layout mode:
    - SINGLE mode: crop to the dominant speaker's body-safe region
    - SPLIT mode: dual-panel vstack with top-2 speakers
    - Fallback: center crop when no speakers detected or SPLIT disabled

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

    # Analyze speaker layout
    segments: list[speaker_layout.LayoutSegment] = speaker_layout.analyze(
        clip_path, config
    )

    # GPU flags
    gpu_info = detect_cuda()
    flags = get_gpu_flags(gpu_info)
    logger.info("Portrait conversion: %s", flags["description"])

    if _needs_segment_rendering(segments):
        _render_segmented_portrait(clip_path, output_path, segments, config, width, height, flags)
    else:
        vf = _build_portrait_filter(segments, config, width, height)
        _render_portrait_clip(clip_path, output_path, vf, flags)

    return output_path


def _build_portrait_filter(
    segments: list[speaker_layout.LayoutSegment],
    config: dict[str, Any],
    width: int,
    height: int,
) -> str:
    """Build the FFmpeg video filter for SINGLE or SPLIT mode."""
    portrait_cfg = config.get("portrait", {})
    split_enabled = portrait_cfg.get("split_enabled", True)

    use_split = False
    use_single = False
    crops = []

    if segments:
        seg = segments[0]
        if seg.mode == LayoutMode.SPLIT and split_enabled and len(seg.speaker_crops) >= 2:
            use_split = True
            crops = seg.speaker_crops[:2]
        elif seg.mode == LayoutMode.SINGLE and len(seg.speaker_crops) >= 1:
            use_single = True
            crops = [seg.speaker_crops[0]]

    # Helper formatters
    def scale_and_pad(w: int, h: int) -> str:
        return f"scale={w}:{h}:force_original_aspect_ratio=decrease,pad={w}:{h}:(ow-iw)/2:(oh-ih)/2"

    if use_split:
        # Dual-panel vertical stack
        c1, c2 = crops[0], crops[1]
        
        # Unpack tuple crops or use CropRegion old API if present
        if isinstance(c1, tuple):
            t1, b1, x1 = c1
            h1 = b1 - t1
            w1 = 1080
        else:
            t1, x1, w1, h1 = c1.top, c1.left, c1.width, c1.height

        if isinstance(c2, tuple):
            t2, b2, x2 = c2
            h2 = b2 - t2
            w2 = 1080
        else:
            t2, x2, w2, h2 = c2.top, c2.left, c2.width, c2.height

        vf = (
            f"[0:v]crop={w1}:{h1}:{x1}:{t1},{scale_and_pad(_OUTPUT_WIDTH, 960)}[top];"
            f"[0:v]crop={w2}:{h2}:{x2}:{t2},{scale_and_pad(_OUTPUT_WIDTH, 960)}[bottom];"
            f"[top][bottom]vstack=inputs=2,{scale_and_pad(_OUTPUT_WIDTH, _OUTPUT_HEIGHT)}"
        )
        return vf

    elif use_single:
        # Single speaker crop
        c = crops[0]
        if isinstance(c, tuple):
            t, b, x = c
            h = b - t
            w = 1080
        else:
            t, x, w, h = c.top, c.left, c.width, c.height

        vf = f"crop={w}:{h}:{x}:{t},{scale_and_pad(_OUTPUT_WIDTH, _OUTPUT_HEIGHT)}"
        return vf

    else:
        # Fallback to center crop
        cw, ch, cx, cy = compute_center_crop(width, height)
        vf = f"crop={cw}:{ch}:{cx}:{cy},{scale_and_pad(_OUTPUT_WIDTH, _OUTPUT_HEIGHT)}"
        return vf

def _needs_segment_rendering(segments: list[speaker_layout.LayoutSegment]) -> bool:
    """Return True when timed layout segments must be rendered separately."""
    if len(segments) > 1:
        return True
    return bool(segments and segments[0].start_sec > 0)


def _render_segmented_portrait(
    clip_path: str,
    output_path: str,
    segments: list[speaker_layout.LayoutSegment],
    config: dict[str, Any],
    width: int,
    height: int,
    flags: dict[str, Any],
) -> None:
    """Render each layout segment separately, then concatenate the segments.

    FFmpeg filter graphs cannot switch arbitrary SINGLE/SPLIT crop geometry over
    time cleanly with the simple crop/vstack chain, so each LayoutSegment is cut
    with -ss/-t, rendered to a uniform 1080x1920 H.264/AAC segment, then joined
    with concat demuxer as required by AGENTS.md §3a.
    """
    normalized_segments = [seg for seg in segments if seg.end_sec > seg.start_sec]
    if not normalized_segments:
        vf = _build_portrait_filter([], config, width, height)
        _render_portrait_clip(clip_path, output_path, vf, flags)
        return

    output = Path(output_path)
    with tempfile.TemporaryDirectory(prefix="ytclipper_portrait_") as tmp:
        tmp_dir = Path(tmp)
        rendered_paths: list[Path] = []
        for idx, segment in enumerate(normalized_segments):
            segment_path = tmp_dir / f"segment_{idx:04d}.mp4"
            vf = _build_portrait_filter([segment], config, width, height)
            _render_portrait_clip(
                clip_path,
                str(segment_path),
                vf,
                flags,
                start_sec=segment.start_sec,
                duration=segment.end_sec - segment.start_sec,
                force_aac=True,
            )
            rendered_paths.append(segment_path)

        concat_list = tmp_dir / "segments.txt"
        concat_list.write_text(
            "".join(f"file '{path.as_posix()}'\n" for path in rendered_paths),
            encoding="utf-8",
        )
        cmd = [
            _FFMPEG,
            "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", str(concat_list),
            "-c", "copy",
            str(output),
        ]
        _run_ffmpeg(cmd, context=f"concat_portrait({clip_path})")


def _render_portrait_clip(
    clip_path: str,
    output_path: str,
    vf: str,
    flags: dict[str, Any],
    start_sec: float | None = None,
    duration: float | None = None,
    force_aac: bool = False,
) -> None:
    """Render one portrait clip or one timed segment with shared codec flags."""
    hwaccel = flags.get("hwaccel", [])
    encoder = flags.get("encoder", "libx264")
    quality_args = ["-cq", "23"] if encoder == "h264_nvenc" else ["-crf", "23"]

    timing_args: list[str] = []
    if start_sec is not None:
        timing_args.extend(["-ss", str(start_sec)])
    if duration is not None:
        timing_args.extend(["-t", str(duration)])

    audio_args = ["-c:a", "aac", "-b:a", "128k"] if force_aac else ["-c:a", "copy"]
    cmd: list[str] = [
        _FFMPEG,
        "-y",
        *hwaccel,
        *timing_args,
        "-i", clip_path,
        "-vf", vf,
        "-c:v", encoder,
        "-preset", "medium",
        *quality_args,
        *audio_args,
        output_path,
    ]
    _run_ffmpeg(cmd, context=f"portrait({clip_path})")


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

        stream = streams[0]
        w = int(stream["width"])
        h = int(stream["height"])

        # FFmpeg filters see autorotated frames, so dimensions must match rotation
        rotation = 0
        tags = stream.get("tags") or {}
        if "rotate" in tags:
            rotation = int(tags["rotate"])
        else:
            for side_data in stream.get("side_data_list") or []:
                if "rotation" in side_data:
                    rotation = int(side_data["rotation"])
                    break

        if abs(rotation) % 180 == 90:
            w, h = h, w

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
