"""Unit tests for pipeline/speaker_layout.py.

Tests face scoring, window classification, hysteresis debounce,
body-safe crop calculator, and end-to-end analyze() with mocked cv2.
Uses synthetic face bounding boxes — no real video or OpenCV face detection needed.

AGENTS.md Section 3a is the authoritative spec.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from pipeline.speaker_layout import (
    FaceInfo,
    LayoutMode,
    LayoutSegment,
    analyze,
    apply_hysteresis,
    classify_window,
    compute_body_safe_crop,
    score_active_faces,
    validate_config,
)

# ── helpers ──────────────────────────────────────────────────────────────

DEFAULT_CONFIG = {
    "portrait": {
        "face_backend": "opencv",
        "split_enabled": True,
        "split_active_threshold": 0.15,
        "split_window_ratio": 0.6,
        "split_hysteresis_sec": 3.0,
        "body_head_pad_ratio": 0.30,
        "body_lower_pad_ratio": 1.20,
    }
}


def _face(x: int, y: int, w: int, h: int, score: float = 0.5) -> FaceInfo:
    return FaceInfo(bbox=(x, y, w, h), active_score=score)


def _silent(x: int, y: int, w: int, h: int) -> FaceInfo:
    return _face(x, y, w, h, score=0.05)


def _active(x: int, y: int, w: int, h: int) -> FaceInfo:
    return _face(x, y, w, h, score=0.4)


# ═══════════════════════════════════════════════════════════════════════
# Data model tests
# ═══════════════════════════════════════════════════════════════════════

def test_face_info_stores_bbox_and_score():
    f = FaceInfo(bbox=(100, 200, 80, 100), active_score=0.75)
    assert f.bbox == (100, 200, 80, 100)
    assert f.active_score == 0.75


def test_layout_segment_has_mode_and_crops():
    seg = LayoutSegment(
        start_sec=0.0, end_sec=3.0,
        mode=LayoutMode.SINGLE,
        speaker_crops=[(100, 200, 80, 100)],
    )
    assert seg.mode == LayoutMode.SINGLE
    assert len(seg.speaker_crops) == 1


def test_layout_segment_split_has_two_crops():
    seg = LayoutSegment(
        start_sec=0.0, end_sec=5.0,
        mode=LayoutMode.SPLIT,
        speaker_crops=[(100, 100, 80, 80), (400, 100, 80, 80)],
    )
    assert seg.mode == LayoutMode.SPLIT
    assert len(seg.speaker_crops) == 2


def test_layout_mode_enum_values():
    assert LayoutMode.SINGLE.value == "single"
    assert LayoutMode.SPLIT.value == "split"


# ═══════════════════════════════════════════════════════════════════════
# score_active_faces() tests
# ═══════════════════════════════════════════════════════════════════════

def test_score_active_faces_tags_above_threshold():
    faces = [_face(100, 100, 60, 80, score=0.3), _face(400, 100, 60, 80, score=0.05)]
    active = score_active_faces(faces, threshold=0.15)
    assert len(active) == 1
    assert active[0].bbox == (100, 100, 60, 80)


def test_score_active_faces_empty_input():
    assert score_active_faces([], threshold=0.15) == []


def test_score_active_faces_all_silent():
    faces = [_silent(100, 100, 60, 80), _silent(400, 100, 60, 80)]
    active = score_active_faces(faces, threshold=0.15)
    assert active == []


def test_score_active_faces_custom_threshold():
    """AGENTS.md: Face below threshold → SILENT."""
    faces = [_face(100, 100, 60, 80, score=0.14)]
    assert score_active_faces(faces, threshold=0.15) == []


def test_score_active_faces_exact_threshold():
    """Score exactly at threshold → ACTIVE."""
    faces = [_face(100, 100, 60, 80, score=0.15)]
    assert len(score_active_faces(faces, threshold=0.15)) == 1


# ═══════════════════════════════════════════════════════════════════════
# classify_window() tests
# ═══════════════════════════════════════════════════════════════════════

def test_classify_window_single_face_is_single():
    frames = [[_active(100, 100, 60, 80)] for _ in range(10)]
    assert classify_window(frames, 0.15, 0.6) == LayoutMode.SINGLE


def test_classify_window_two_active_is_split():
    """AGENTS.md: 2 faces side-by-side → SPLIT."""
    frames = [[_active(100, 100, 60, 80), _active(400, 100, 60, 80)] for _ in range(10)]
    assert classify_window(frames, 0.15, 0.6) == LayoutMode.SPLIT


def test_classify_window_below_ratio_is_single():
    """50% dual-active frames < 60% → SINGLE."""
    dual = [[_active(100, 100, 60, 80), _active(400, 100, 60, 80)] for _ in range(5)]
    single = [[_active(300, 100, 60, 80)] for _ in range(5)]
    assert classify_window(dual + single, 0.15, 0.6) == LayoutMode.SINGLE


def test_classify_window_above_ratio_is_split():
    """70% dual-active > 60% → SPLIT."""
    dual = [[_active(100, 100, 60, 80), _active(400, 100, 60, 80)] for _ in range(7)]
    single = [[_active(300, 100, 60, 80)] for _ in range(3)]
    assert classify_window(dual + single, 0.15, 0.6) == LayoutMode.SPLIT


def test_classify_window_no_faces_is_single():
    """No faces → SINGLE (center-crop fallback)."""
    assert classify_window([[] for _ in range(10)], 0.15, 0.6) == LayoutMode.SINGLE


def test_classify_window_one_active_one_silent():
    """AGENTS.md: Face below threshold → not counted."""
    frames = [[_active(100, 100, 60, 80), _silent(400, 100, 60, 80)] for _ in range(10)]
    assert classify_window(frames, 0.15, 0.6) == LayoutMode.SINGLE


def test_classify_window_three_active():
    """AGENTS.md: 3+ active → SPLIT."""
    frames = [[
        _face(100, 100, 60, 80, score=0.5),
        _face(400, 100, 60, 80, score=0.4),
        _face(700, 100, 60, 80, score=0.3),
    ] for _ in range(10)]
    assert classify_window(frames, 0.15, 0.6) == LayoutMode.SPLIT


# ═══════════════════════════════════════════════════════════════════════
# apply_hysteresis() tests
# ═══════════════════════════════════════════════════════════════════════

def test_hysteresis_all_same():
    raw = [LayoutMode.SINGLE] * 6
    assert apply_hysteresis(raw, 3) == [LayoutMode.SINGLE] * 6


def test_hysteresis_flip_requires_three_consecutive():
    """AGENTS.md: require 3 consecutive windows to flip."""
    raw = ([LayoutMode.SINGLE] * 15 + [LayoutMode.SPLIT] * 3 + [LayoutMode.SPLIT] * 5)
    result = apply_hysteresis(raw, 3)
    assert all(m == LayoutMode.SINGLE for m in result[:18])
    assert all(m == LayoutMode.SPLIT for m in result[18:])


def test_hysteresis_rejects_two_consecutive():
    """Two SPLIT then SINGLE → no flip."""
    raw = [LayoutMode.SINGLE] * 15 + [LayoutMode.SPLIT] * 2 + [LayoutMode.SINGLE] * 10
    assert all(m == LayoutMode.SINGLE for m in apply_hysteresis(raw, 3))


def test_hysteresis_empty():
    assert apply_hysteresis([], 3) == []


def test_hysteresis_single():
    assert apply_hysteresis([LayoutMode.SINGLE], 3) == [LayoutMode.SINGLE]


def test_hysteresis_flip_back_to_single():
    raw = ([LayoutMode.SPLIT] * 15 + [LayoutMode.SINGLE] * 3 + [LayoutMode.SINGLE] * 5)
    result = apply_hysteresis(raw, 3)
    assert all(m == LayoutMode.SPLIT for m in result[:18])
    assert all(m == LayoutMode.SINGLE for m in result[18:])


def test_hysteresis_two_element():
    assert apply_hysteresis([LayoutMode.SINGLE, LayoutMode.SPLIT], 3) == [LayoutMode.SINGLE, LayoutMode.SINGLE]


def test_hysteresis_transition_stability():
    """One speaker leaves mid-segment → switch to SINGLE after hysteresis."""
    raw = ([LayoutMode.SINGLE] * 10 + [LayoutMode.SPLIT] * 3 + [LayoutMode.SINGLE] * 7)
    result = apply_hysteresis(raw, 3)
    assert all(m == LayoutMode.SINGLE for m in result[:13])
    assert all(m == LayoutMode.SPLIT for m in result[13:16])
    assert all(m == LayoutMode.SINGLE for m in result[16:])


# ═══════════════════════════════════════════════════════════════════════
# compute_body_safe_crop() tests
# ═══════════════════════════════════════════════════════════════════════

def test_body_safe_crop_center():
    crop = compute_body_safe_crop((860, 400, 100, 120), 1920, 1080, 0.30, 1.20, 1080, 960)
    top, bottom, x = crop
    assert top == 364
    assert bottom == 664
    assert 0 <= x <= 1920 - 1080


def test_body_safe_crop_top_edge():
    """Face near top → clamp crop_top=0."""
    crop = compute_body_safe_crop((860, 10, 100, 120), 1920, 1080, 0.30, 1.20, 1080, 960)
    assert crop[0] == 0
    assert crop[1] == 274


def test_body_safe_crop_bottom_edge():
    """Face near bottom → clamp to src_height."""
    crop = compute_body_safe_crop((860, 900, 100, 120), 1920, 1080, 0.30, 1.20, 1080, 960)
    assert crop[0] == 864
    assert crop[1] == 1080


def test_body_safe_crop_left_edge():
    """Face far left → x clamped to 0."""
    crop = compute_body_safe_crop((10, 400, 100, 120), 1920, 1080, 0.30, 1.20, 1080, 960)
    assert crop[2] == 0


def test_body_safe_crop_right_edge():
    """Face far right → x clamped to src_w-panel_w."""
    crop = compute_body_safe_crop((1850, 400, 100, 120), 1920, 1080, 0.30, 1.20, 1080, 960)
    assert crop[2] == 840


def test_body_safe_crop_preserves_padding():
    """AGENTS.md: NEVER reduce padding."""
    crop = compute_body_safe_crop((860, 10, 100, 120), 1920, 1080, 0.30, 1.20, 1080, 960)
    assert crop[1] - crop[0] == 274  # full range


def test_body_safe_crop_returns_3_tuple():
    result = compute_body_safe_crop((500, 400, 80, 100), 1920, 1080, 0.30, 1.20, 1080, 960)
    assert len(result) == 3
    assert all(isinstance(v, int) for v in result)


def test_body_safe_crop_square_face():
    crop = compute_body_safe_crop((960, 500, 100, 100), 1920, 1080, 0.30, 1.20, 1080, 960)
    assert crop[0] == 470
    assert crop[1] == 720


def test_body_safe_crop_at_origin():
    crop = compute_body_safe_crop((0, 0, 50, 60), 1920, 1080, 0.30, 1.20, 1080, 960)
    assert crop[0] == 0
    assert crop[1] == 60 + 72


# ═══════════════════════════════════════════════════════════════════════
# validate_config() tests
# ═══════════════════════════════════════════════════════════════════════

def test_validate_config_rejects_low_head_pad():
    config = {"portrait": {**DEFAULT_CONFIG["portrait"], "body_head_pad_ratio": 0.1}}
    with pytest.raises(ValueError, match="head_pad_ratio"):
        validate_config(config)


def test_validate_config_rejects_low_body_pad():
    config = {"portrait": {**DEFAULT_CONFIG["portrait"], "body_lower_pad_ratio": 0.8}}
    with pytest.raises(ValueError, match="body_lower_pad_ratio"):
        validate_config(config)


def test_validate_config_accepts_valid():
    validate_config(DEFAULT_CONFIG)


def test_validate_config_minimum_accepted():
    config = {"portrait": {**DEFAULT_CONFIG["portrait"], "body_head_pad_ratio": 0.20, "body_lower_pad_ratio": 1.0}}
    validate_config(config)


# ═══════════════════════════════════════════════════════════════════════
# analyze() tests — with mocked cv2
# ═══════════════════════════════════════════════════════════════════════

def _mock_cap(fps=25.0, frame_count=150, width=1920, height=1080):
    cap = MagicMock()
    cap.isOpened.return_value = True
    cap.get.side_effect = lambda prop: {
        5: fps, 7: frame_count, 3: width, 4: height,
    }.get(prop, 0)
    cap.read.return_value = (True, MagicMock())
    return cap


def _patch_cv2():
    """Context manager patches for cv2.VideoCapture and constants."""
    return [
        patch("pipeline.speaker_layout.cv2.VideoCapture", return_value=_mock_cap()),
        patch("pipeline.speaker_layout.cv2.CAP_PROP_FPS", 5),
        patch("pipeline.speaker_layout.cv2.CAP_PROP_FRAME_COUNT", 7),
        patch("pipeline.speaker_layout.cv2.CAP_PROP_FRAME_WIDTH", 3),
        patch("pipeline.speaker_layout.cv2.CAP_PROP_FRAME_HEIGHT", 4),
        patch("pipeline.speaker_layout.cv2.CAP_PROP_POS_FRAMES", 1),
    ]


def test_analyze_single_speaker():
    patches = _patch_cv2()
    with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5]:
        with patch("pipeline.speaker_layout._compute_active_scores") as mock:
            mock.return_value = [[FaceInfo(bbox=(200, 300, 80, 100), active_score=0.4)]] * 30
            segments = analyze("test.mp4", DEFAULT_CONFIG)
    assert len(segments) >= 1
    for seg in segments:
        assert seg.mode == LayoutMode.SINGLE


def test_analyze_split_disabled():
    config = {"portrait": {**DEFAULT_CONFIG["portrait"], "split_enabled": False}}
    patches = _patch_cv2()
    with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5]:
        with patch("pipeline.speaker_layout._compute_active_scores") as mock:
            mock.return_value = [[
                FaceInfo(bbox=(200, 300, 80, 100), active_score=0.4),
                FaceInfo(bbox=(600, 300, 80, 100), active_score=0.4),
            ]] * 30
            segments = analyze("test.mp4", config)
    for seg in segments:
        assert seg.mode == LayoutMode.SINGLE


def test_analyze_no_faces():
    patches = _patch_cv2()
    with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5]:
        with patch("pipeline.speaker_layout._compute_active_scores") as mock:
            mock.return_value = [[]] * 30
            segments = analyze("test.mp4", DEFAULT_CONFIG)
    for seg in segments:
        assert seg.mode == LayoutMode.SINGLE
        assert seg.speaker_crops == []


def test_analyze_split_top_two():
    """AGENTS.md: 3+ active → SPLIT, top-2."""
    patches = _patch_cv2()
    faces3 = [[
        FaceInfo(bbox=(100, 100, 60, 80), active_score=0.5),
        FaceInfo(bbox=(400, 100, 60, 80), active_score=0.4),
        FaceInfo(bbox=(700, 100, 60, 80), active_score=0.3),
    ]] * 30
    with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5]:
        with patch("pipeline.speaker_layout._compute_active_scores") as mock:
            mock.return_value = faces3
            segments = analyze("test.mp4", DEFAULT_CONFIG)
    for seg in segments:
        if seg.mode == LayoutMode.SPLIT:
            assert len(seg.speaker_crops) <= 2


def test_analyze_returns_layout_segment_list():
    patches = _patch_cv2()
    with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5]:
        with patch("pipeline.speaker_layout._compute_active_scores") as mock:
            mock.return_value = [[FaceInfo(bbox=(200, 300, 80, 100), active_score=0.4)]] * 30
            segments = analyze("test.mp4", DEFAULT_CONFIG)
    assert isinstance(segments, list)
    for seg in segments:
        assert isinstance(seg, LayoutSegment)
        assert seg.start_sec < seg.end_sec


def test_analyze_video_open_failure():
    cap = MagicMock()
    cap.isOpened.return_value = False
    with patch("pipeline.speaker_layout.cv2.VideoCapture", return_value=cap):
        segments = analyze("nonexistent.mp4", DEFAULT_CONFIG)
    assert segments == []


def test_analyze_zero_frames():
    cap = _mock_cap(frame_count=0)
    with patch("pipeline.speaker_layout.cv2.VideoCapture", return_value=cap):
        segments = analyze("empty.mp4", DEFAULT_CONFIG)
    assert segments == []


# ═══════════════════════════════════════════════════════════════════════
# Edge cases from AGENTS.md
# ═══════════════════════════════════════════════════════════════════════

def test_hysteresis_alternating_no_commit():
    raw = [LayoutMode.SINGLE, LayoutMode.SPLIT] * 10
    assert apply_hysteresis(raw, 3) == [LayoutMode.SINGLE] * 20


def test_analyze_mixed_frames_single():
    """Some frames with faces, some without → still SINGLE."""
    patches = _patch_cv2()
    with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5]:
        with patch("pipeline.speaker_layout._compute_active_scores") as mock:
            mock.return_value = [[FaceInfo(bbox=(200, 300, 80, 100), active_score=0.4)]] * 15 + [[]] * 15
            segments = analyze("test.mp4", DEFAULT_CONFIG)
    assert len(segments) >= 1
    for seg in segments:
        assert seg.mode == LayoutMode.SINGLE
