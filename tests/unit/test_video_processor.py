"""Unit tests for pipeline/video_processor.py."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pipeline.highlight_detector import Highlight
from pipeline.video_processor import (
    VideoProcessingError,
    compute_center_crop,
    convert_to_portrait,
    cut,
    probe_dimensions,
)


# ── helpers ──────────────────────────────────────────────────────────────


def _make_highlight(
    start: float = 10.0, end: float = 25.0, score: int = 8, hook: str = "Test"
) -> Highlight:
    return Highlight(start=start, end=end, hook_text=hook, score=score)


def _completed_proc(returncode: int = 0, stdout: str = "", stderr: str = "") -> MagicMock:
    proc = MagicMock(spec=subprocess.CompletedProcess)
    proc.returncode = returncode
    proc.stdout = stdout
    proc.stderr = stderr
    return proc


@pytest.fixture
def fake_config() -> dict:
    return {
        "llm": {"base_url": "http://localhost:11434/v1", "model": "llama3", "api_key": "ollama"},
        "transcription": {
            "base_url": "https://api.openai.com/v1",
            "model": "whisper-1",
            "api_key": "***",
        },
        "portrait": {"face_backend": "opencv"},
    }


# ── test 1: compute_center_crop for landscape 16:9 input ────────────────
def test_center_crop_wide_input() -> None:
    """1920x1080 → crop width to 9:16, full height kept."""
    cw, ch, cx, cy = compute_center_crop(1920, 1080)
    assert (cw, ch) == (607, 1080)  # 1080 * 9/16 = 607.5 → int truncates
    assert cx == (1920 - cw) // 2
    assert cy == 0


# ── test 2: compute_center_crop for portrait 9:16 input ─────────────────
def test_center_crop_tall_input() -> None:
    """1080x1920 → no crop needed (already 9:16)."""
    cw, ch, cx, cy = compute_center_crop(1080, 1920)
    assert (cw, ch) == (1080, 1920)
    assert (cx, cy) == (0, 0)


# ── test 3: compute_center_crop for square input ────────────────────────
def test_center_crop_square_input() -> None:
    """1000x1000 → crop width (input is too wide vs 9:16)."""
    cw, ch, cx, cy = compute_center_crop(1000, 1000)
    assert (cw, ch) == (562, 1000)  # 1000 * 9/16 = 562.5
    assert cx == (1000 - 562) // 2
    assert cy == 0


# ── test 4: compute_center_crop for ultra-wide input ────────────────────
def test_center_crop_ultrawide_input() -> None:
    """3840x1080 → heavy horizontal crop to 9:16."""
    cw, ch, cx, cy = compute_center_crop(3840, 1080)
    assert cw == 607
    assert ch == 1080
    assert cy == 0
    assert cx > 0


# ── test 5: probe_dimensions parses ffprobe output ──────────────────────
@patch("pipeline.video_processor.subprocess.run")
def test_probe_dimensions_parses_ffprobe(mock_run: MagicMock) -> None:
    mock_run.return_value = _completed_proc(
        returncode=0, stdout='{"streams": [{"width": 1920, "height": 1080}]}'
    )
    w, h = probe_dimensions("/path/to/video.mp4")
    assert (w, h) == (1920, 1080)
    mock_run.assert_called_once()
    args = mock_run.call_args[0][0]
    assert Path(args[0]).name == "ffprobe"
    assert "/path/to/video.mp4" in args


# ── test 6: probe_dimensions raises on ffprobe failure ──────────────────
@patch("pipeline.video_processor.subprocess.run")
def test_probe_dimensions_failure_raises(mock_run: MagicMock) -> None:
    mock_run.return_value = _completed_proc(returncode=1, stderr="No such file")
    with pytest.raises(VideoProcessingError, match="ffprobe"):
        probe_dimensions("/missing.mp4")


# ── test 7: cut uses seek + duration + codec copy ──────────────────────
@patch("pipeline.video_processor.subprocess.run")
def test_cut_uses_seek_duration_and_copy(
    mock_run: MagicMock, tmp_path: Path
) -> None:
    mock_run.return_value = _completed_proc(returncode=0)
    output = tmp_path / "out.mp4"
    result = cut("/path/in.mp4", _make_highlight(start=10.0, end=25.0), str(output))

    assert result == str(output)
    args = mock_run.call_args[0][0]
    assert Path(args[0]).name == "ffmpeg"
    # seek before input (fast)
    assert "-ss" in args
    assert "10.0" in args
    # duration
    assert "-t" in args
    assert "15.0" in args
    # codec copy
    assert "-c" in args
    assert "copy" in args
    # input + output
    assert "/path/in.mp4" in args
    assert str(output) in args


# ── test 8: cut raises on ffmpeg failure with stderr ───────────────────
@patch("pipeline.video_processor.subprocess.run")
def test_cut_raises_on_ffmpeg_error(mock_run: MagicMock, tmp_path: Path) -> None:
    mock_run.return_value = _completed_proc(
        returncode=1, stderr="Invalid data found when processing input"
    )
    with pytest.raises(VideoProcessingError, match="Invalid data"):
        cut("/path/in.mp4", _make_highlight(), str(tmp_path / "out.mp4"))


# ── test 9: cut validates end <= start ────────────────────────────────
@patch("pipeline.video_processor.subprocess.run")
def test_cut_rejects_inverted_range(mock_run: MagicMock, tmp_path: Path) -> None:
    with pytest.raises(VideoProcessingError, match="[Ee]nd must be >"):
        cut(
            "/path/in.mp4",
            _make_highlight(start=30.0, end=10.0),
            str(tmp_path / "out.mp4"),
        )
    mock_run.assert_not_called()


# ── test 10: cut rejects negative timestamp ────────────────────────────
@patch("pipeline.video_processor.subprocess.run")
def test_cut_rejects_negative_timestamp(mock_run: MagicMock, tmp_path: Path) -> None:
    with pytest.raises(VideoProcessingError, match="non-negative"):
        cut(
            "/path/in.mp4",
            _make_highlight(start=-1.0, end=5.0),
            str(tmp_path / "out.mp4"),
        )
    mock_run.assert_not_called()


# ── test 11: cut rejects NaN timestamp ────────────────────────────────
@patch("pipeline.video_processor.subprocess.run")
def test_cut_rejects_nan_timestamp(mock_run: MagicMock, tmp_path: Path) -> None:
    with pytest.raises(VideoProcessingError, match="finite"):
        cut(
            "/path/in.mp4",
            _make_highlight(start=float("nan"), end=5.0),
            str(tmp_path / "out.mp4"),
        )
    mock_run.assert_not_called()


# ── test 12: convert_to_portrait — CPU path, single mode, scale to 1080x1920 ──
@patch("pipeline.video_processor.subprocess.run")
@patch("pipeline.video_processor.probe_dimensions")
@patch("pipeline.video_processor.detect_cuda")
def test_convert_to_portrait_single_mode_scale(
    mock_cuda: MagicMock,
    mock_probe: MagicMock,
    mock_run: MagicMock,
    fake_config: dict,
    tmp_path: Path,
) -> None:
    mock_cuda.return_value = {
        "available": False,
        "name": None,
        "h264_nvenc_available": False,
    }
    mock_probe.return_value = (1920, 1080)
    mock_run.return_value = _completed_proc(returncode=0)

    output = tmp_path / "portrait.mp4"
    result = convert_to_portrait(str(tmp_path / "in.mp4"), fake_config, str(output))

    assert result == str(output)
    # Verify FFmpeg command
    args = mock_run.call_args[0][0]
    assert Path(args[0]).name == "ffmpeg"
    # Find the -vf filter
    vf_idx = args.index("-vf")
    filter_str = args[vf_idx + 1]
    # Should contain crop and scale=1080:1920
    assert "crop=" in filter_str
    assert "scale=1080:1920" in filter_str
    # CPU encoder
    assert "libx264" in args
    # No GPU flags
    assert "-hwaccel" not in args
    assert "h264_nvenc" not in args


# ── test 13: convert_to_portrait — GPU path applies hwaccel and nvenc ────
@patch("pipeline.video_processor.subprocess.run")
@patch("pipeline.video_processor.probe_dimensions")
@patch("pipeline.video_processor.detect_cuda")
def test_convert_to_portrait_gpu_flags(
    mock_cuda: MagicMock,
    mock_probe: MagicMock,
    mock_run: MagicMock,
    fake_config: dict,
    tmp_path: Path,
) -> None:
    mock_cuda.return_value = {
        "available": True,
        "name": "RTX 4090",
        "h264_nvenc_available": True,
    }
    mock_probe.return_value = (1920, 1080)
    mock_run.return_value = _completed_proc(returncode=0)

    result = convert_to_portrait(
        str(tmp_path / "in.mp4"), fake_config, str(tmp_path / "portrait.mp4")
    )

    assert result is not None
    args = mock_run.call_args[0][0]
    # GPU hwaccel flag present
    assert "-hwaccel" in args
    assert "cuda" in args
    # NVENC encoder
    assert "h264_nvenc" in args
    # libx264 should NOT be the encoder
    assert "libx264" not in args


# ── test 14: convert_to_portrait — output resolution is 1080x1920 ──────
@patch("pipeline.video_processor.subprocess.run")
@patch("pipeline.video_processor.probe_dimensions")
@patch("pipeline.video_processor.detect_cuda")
def test_convert_to_portrait_output_resolution_always_1080x1920(
    mock_cuda: MagicMock,
    mock_probe: MagicMock,
    mock_run: MagicMock,
    fake_config: dict,
    tmp_path: Path,
) -> None:
    mock_cuda.return_value = {
        "available": False,
        "name": None,
        "h264_nvenc_available": False,
    }
    # Try various input resolutions
    for input_w, input_h in [(3840, 2160), (1920, 1080), (1280, 720), (1080, 1920)]:
        mock_probe.return_value = (input_w, input_h)
        mock_run.return_value = _completed_proc(returncode=0)
        convert_to_portrait(
            str(tmp_path / "in.mp4"),
            fake_config,
            str(tmp_path / f"out_{input_w}x{input_h}.mp4"),
        )
        args = mock_run.call_args[0][0]
        vf_idx = args.index("-vf")
        filter_str = args[vf_idx + 1]
        assert "scale=1080:1920" in filter_str, f"missing 1080x1920 for {input_w}x{input_h}"


# ── test 15: convert_to_portrait — raises on ffmpeg failure ─────────────
@patch("pipeline.video_processor.subprocess.run")
@patch("pipeline.video_processor.probe_dimensions")
@patch("pipeline.video_processor.detect_cuda")
def test_convert_to_portrait_raises_on_ffmpeg_error(
    mock_cuda: MagicMock,
    mock_probe: MagicMock,
    mock_run: MagicMock,
    fake_config: dict,
    tmp_path: Path,
) -> None:
    mock_cuda.return_value = {"available": False, "name": None, "h264_nvenc_available": False}
    mock_probe.return_value = (1920, 1080)
    mock_run.return_value = _completed_proc(returncode=1, stderr="Conversion failed!")

    with pytest.raises(VideoProcessingError, match="Conversion failed"):
        convert_to_portrait(
            str(tmp_path / "in.mp4"),
            fake_config,
            str(tmp_path / "out.mp4"),
        )


# ── test 16: convert_to_portrait — GPU available but no nvenc → CPU fallback ──
@patch("pipeline.video_processor.subprocess.run")
@patch("pipeline.video_processor.probe_dimensions")
@patch("pipeline.video_processor.detect_cuda")
def test_convert_to_portrait_falls_back_to_cpu_when_nvenc_missing(
    mock_cuda: MagicMock,
    mock_probe: MagicMock,
    mock_run: MagicMock,
    fake_config: dict,
    tmp_path: Path,
) -> None:
    """GPU detected but FFmpeg lacks h264_nvenc → use libx264."""
    mock_cuda.return_value = {
        "available": True,
        "name": "GTX 1080",
        "h264_nvenc_available": False,
    }
    mock_probe.return_value = (1920, 1080)
    mock_run.return_value = _completed_proc(returncode=0)

    convert_to_portrait(
        str(tmp_path / "in.mp4"),
        fake_config,
        str(tmp_path / "out.mp4"),
    )

    args = mock_run.call_args[0][0]
    assert "libx264" in args
    assert "h264_nvenc" not in args
    assert "-hwaccel" not in args


# ── test 17: convert_to_portrait — probe failure raises immediately ────
@patch("pipeline.video_processor.probe_dimensions")
@patch("pipeline.video_processor.detect_cuda")
def test_convert_to_portrait_probe_failure(
    mock_cuda: MagicMock,
    mock_probe: MagicMock,
    fake_config: dict,
    tmp_path: Path,
) -> None:
    mock_cuda.return_value = {"available": False, "name": None, "h264_nvenc_available": False}
    mock_probe.side_effect = VideoProcessingError("probe failed: file not found")

    with pytest.raises(VideoProcessingError, match="probe failed"):
        convert_to_portrait(
            str(tmp_path / "missing.mp4"),
            fake_config,
            str(tmp_path / "out.mp4"),
        )


# ── test 18: cut is shell-safe — no shell=True ─────────────────────────
def test_cut_uses_shell_false() -> None:
    """Static check: ensure cut() never invokes shell=True (command injection guard)."""
    import inspect

    from pipeline import video_processor

    source = inspect.getsource(video_processor)
    # Regex to catch subprocess.run(..., shell=True)
    import re
    assert not re.search(r"subprocess\.run\([^)]*shell\s*=\s*True", source)


# ── test 19: convert_to_portrait is shell-safe ──────────────────────────
def test_convert_to_portrait_uses_shell_false() -> None:
    import inspect

    from pipeline import video_processor

    source = inspect.getsource(video_processor)
    import re
    assert not re.search(r"subprocess\.run\([^)]*shell\s*=\s*True", source)


# ── test 20: output extension is mp4 ────────────────────────────────────
@patch("pipeline.video_processor.subprocess.run")
def test_cut_output_path_returned_unchanged(
    mock_run: MagicMock, tmp_path: Path
) -> None:
    mock_run.return_value = _completed_proc(returncode=0)
    out = str(tmp_path / "my_clip.mp4")
    result = cut("/in.mp4", _make_highlight(), out)
    assert result == out
