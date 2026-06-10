"""Speaker layout analysis module.

Detects speaker count per segment and decides layout mode (SINGLE/SPLIT).
Provides body-safe crop calculations that account for head and body padding
so people are never cut in half in the output panels.

This module is read-only relative to the video — it only analyzes frames,
never writes them.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Any

try:
    import cv2
except ImportError:
    cv2 = None  # type: ignore

# ── constants ────────────────────────────────────────────────────────────

HEAD_PAD_RATIO: float = 0.30  # fraction of face height above head (min 0.2)
BODY_PAD_RATIO: float = 1.20  # fraction of face height below chin (min 1.0)
PANEL_W: int = 1080
PANEL_H: int = 960
SPLIT_ACTIVE_THRESHOLD: float = 0.15
SPLIT_WINDOW_RATIO: float = 0.6
HYSTERESIS_SEC: float = 3.0
FRAME_SAMPLE_INTERVAL: int = 5


class Mode(enum.Enum):
    """Layout mode for a video segment."""

    SINGLE = "single"
    SPLIT = "split"
    CENTER_FALLBACK = "center_fallback"


# v2-compatible alias.
LayoutMode = Mode


@dataclass(frozen=True)
class CropRegion:
    """A body-safe crop region in source frame coordinates."""

    top: int
    bottom: int
    left: int
    right: int

    @property
    def height(self) -> int:
        return self.bottom - self.top

    @property
    def width(self) -> int:
        return self.right - self.left


@dataclass(frozen=True)
class FaceInfo:
    """Detected face with bounding box and activity score."""

    bbox: tuple[int, int, int, int]
    active_score: float


@dataclass(frozen=True)
class LayoutSegment:
    """A time segment with a decided layout mode and speaker crop info."""

    start_sec: float
    end_sec: float
    mode: Mode
    speaker_crops: list[Any]


def calculate_body_safe_crop(
    face_bbox: tuple[int, int, int, int],
    src_height: int,
    src_width: int,
) -> CropRegion:
    """Compute a body-safe crop region around a face bbox.

    Applies HEAD_PAD_RATIO above the face and BODY_PAD_RATIO below. Result is
    clamped to frame bounds. Padding is never reduced before clamping.
    """
    x, y, w, h = face_bbox

    crop_top = max(0, int(y - h * HEAD_PAD_RATIO))
    crop_bottom = min(src_height, int(y + h + h * BODY_PAD_RATIO))

    cx = x + w // 2
    max_left = max(0, src_width - PANEL_W)
    crop_left = max(0, min(max_left, cx - PANEL_W // 2))
    crop_right = min(src_width, crop_left + PANEL_W)

    return CropRegion(
        top=crop_top,
        bottom=crop_bottom,
        left=crop_left,
        right=crop_right,
    )


def compute_body_safe_crop(
    bbox: tuple[int, int, int, int],
    src_width: int,
    src_height: int,
    head_pad_ratio: float = HEAD_PAD_RATIO,
    body_pad_ratio: float = BODY_PAD_RATIO,
    panel_w: int = PANEL_W,
    panel_h: int = PANEL_H,
) -> tuple[int, int, int]:
    """v2-compatible body-safe crop helper.

    Returns (crop_top, crop_bottom, crop_x).
    """
    x, y, w, h = bbox
    crop_top = max(0, int(y - h * head_pad_ratio))
    crop_bottom = min(src_height, int(y + h + h * body_pad_ratio))
    crop_cx = x + w // 2
    crop_x = max(0, min(max(0, src_width - panel_w), crop_cx - panel_w // 2))
    return crop_top, crop_bottom, crop_x


def select_top_speakers(
    faces: list[tuple[tuple[int, int, int, int], float]],
    n: int = 2,
) -> list[tuple[tuple[int, int, int, int], float]]:
    """Select top-N speakers by active score, descending."""
    return sorted(faces, key=lambda f: f[1], reverse=True)[:n]


def score_active_faces(faces: list[FaceInfo], threshold: float) -> list[FaceInfo]:
    """Return faces whose active score meets threshold."""
    return [face for face in faces if face.active_score >= threshold]


def classify_window(
    frames: list[list[FaceInfo]],
    threshold: float,
    window_ratio: float,
) -> Mode:
    """Classify one window as SPLIT when > window_ratio frames have 2+ active faces."""
    if not frames:
        return Mode.SINGLE

    dual_active = 0
    for frame_faces in frames:
        if len(score_active_faces(frame_faces, threshold)) >= 2:
            dual_active += 1

    return Mode.SPLIT if (dual_active / len(frames)) > window_ratio else Mode.SINGLE


def apply_hysteresis(raw_modes: list[Mode], min_consecutive: int) -> list[Mode]:
    """Smooth mode sequence: require min_consecutive same-mode windows to flip.

    Algorithm: track a pending candidate mode. When min_consecutive
    consecutive windows agree on the new mode, the flip is committed
    and takes effect from the NEXT window (not the current one).

    If the original mode reappears before min_consecutive is reached,
    the pending counter resets.
    """
    if not raw_modes:
        return []
    if min_consecutive <= 0:
        return list(raw_modes)

    result: list[Mode] = []
    current: Mode = raw_modes[0]
    pending_mode: Mode = raw_modes[0]
    pending_count: int = 0
    flip_pending: bool = False

    for mode in raw_modes:
        if flip_pending:
            current = pending_mode
            pending_count = 0
            flip_pending = False

        if mode == current:
            pending_count = 0
            result.append(current)
        else:
            if mode == pending_mode:
                pending_count += 1
            else:
                pending_mode = mode
                pending_count = 1

            if pending_count >= min_consecutive:
                flip_pending = True
                result.append(current)
            else:
                result.append(current)

    return result


def analyze(clip_path: str, config: dict[str, Any]) -> list[LayoutSegment]:
    """Analyze a video clip and return per-segment layout decisions.

    Tests patch _compute_active_scores to avoid real OpenCV I/O.
    """
    validate_config(config)

    import cv2  # local import — avoids hard dependency for unit tests

    cap = cv2.VideoCapture(clip_path)
    if not cap.isOpened():
        return []
    fps = int(cap.get(cv2.CAP_PROP_FPS) or 25)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    src_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1920)
    src_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1080)
    cap.release()

    if frame_count == 0:
        return []

    frames_faces = _compute_active_scores(clip_path, fps, frame_count)
    if not frames_faces:
        return [LayoutSegment(0.0, 0.0, Mode.CENTER_FALLBACK, [])]

    portrait_cfg = config.get("portrait", {})
    split_enabled = portrait_cfg.get("split_enabled", True)
    threshold = portrait_cfg.get("split_active_threshold", SPLIT_ACTIVE_THRESHOLD)
    window_ratio = portrait_cfg.get("split_window_ratio", SPLIT_WINDOW_RATIO)
    hysteresis_sec = portrait_cfg.get("split_hysteresis_sec", HYSTERESIS_SEC)

    frames_per_window = max(1, fps)
    raw_modes = []
    for start in range(0, len(frames_faces), frames_per_window):
        window = frames_faces[start : start + frames_per_window]
        raw_modes.append(classify_window(window, threshold, window_ratio))

    min_consecutive = max(1, round(hysteresis_sec / (frames_per_window / fps)))
    smooth_modes = apply_hysteresis(raw_modes, min_consecutive)

    window_duration = frames_per_window / fps if fps > 0 else 1.0
    head_pad = portrait_cfg.get("body_head_pad_ratio", HEAD_PAD_RATIO)
    body_pad = portrait_cfg.get("body_lower_pad_ratio", BODY_PAD_RATIO)

    segments = []
    for i, mode in enumerate(smooth_modes):
        start_sec = i * window_duration
        end_sec = min((i + 1) * window_duration, frame_count / fps)

        window_start = i * frames_per_window
        window_faces = frames_faces[window_start : window_start + frames_per_window]

        if mode == Mode.SPLIT and split_enabled:
            # Pick representative faces — top-2 active
            all_active = []
            for frame in window_faces:
                all_active.extend(score_active_faces(frame, threshold))
            all_active.sort(key=lambda f: f.active_score, reverse=True)
            top_two = all_active[:2]
            crops = [
                compute_body_safe_crop(
                    f.bbox, src_width, src_height, head_pad, body_pad,
                )
                for f in top_two
            ]
            segments.append(LayoutSegment(start_sec, end_sec, Mode.SPLIT, crops))
        else:
            # SINGLE mode — track the highest-score face
            all_faces = [f for frame in window_faces for f in frame]
            if all_faces:
                best = max(all_faces, key=lambda f: f.active_score)
                crop = compute_body_safe_crop(
                    best.bbox, src_width, src_height, head_pad, body_pad,
                )
                segments.append(LayoutSegment(start_sec, end_sec, Mode.SINGLE, [crop]))
            else:
                segments.append(LayoutSegment(start_sec, end_sec, Mode.SINGLE, []))

    return segments


def validate_config(config: dict[str, Any]) -> None:
    """Validate portrait padding config minimums."""
    portrait = config.get("portrait", {})
    head_pad = portrait.get("body_head_pad_ratio", HEAD_PAD_RATIO)
    body_pad = portrait.get("body_lower_pad_ratio", BODY_PAD_RATIO)
    if head_pad < 0.2:
        raise ValueError("head_pad_ratio must be >= 0.2")
    if body_pad < 1.0:
        raise ValueError("body_lower_pad_ratio must be >= 1.0")


def _compute_active_scores(
    clip_path: str,
    fps: int,
    frame_count: int,
) -> list[list[FaceInfo]]:
    """Placeholder OpenCV scoring hook.

    Real frame scoring will be expanded later. Tests patch this function.
    """
    return []


SINGLE = Mode.SINGLE
SPLIT = Mode.SPLIT
CENTER_FALLBACK = Mode.CENTER_FALLBACK

__all__ = [
    "Mode",
    "LayoutMode",
    "SINGLE",
    "SPLIT",
    "CENTER_FALLBACK",
    "CropRegion",
    "FaceInfo",
    "LayoutSegment",
    "calculate_body_safe_crop",
    "compute_body_safe_crop",
    "select_top_speakers",
    "score_active_faces",
    "classify_window",
    "apply_hysteresis",
    "analyze",
    "validate_config",
]
