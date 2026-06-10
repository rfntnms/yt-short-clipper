"""Test caption generator."""
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
import os
import subprocess
from pipeline.caption_generator import generate_ass_content, generate_and_burn, CaptioningError

def test_generate_ass_content():
    word_json = [
        {"word": "Hello", "start": 0.0, "end": 0.5},
        {"word": "world", "start": 0.5, "end": 1.0}
    ]
    config = {
        "caption_style": {
            "font_name": "Roboto",
            "font_size": 16,
            "highlight_color": "&H0000FFFF"
        }
    }
    
    ass_content = generate_ass_content(word_json, config)
    
    assert "ScriptType: v4.00+" in ass_content
    assert "Roboto,16" in ass_content
    assert "Dialogue: 0,0:00:00.00,0:00:00.50" in ass_content
    assert "Dialogue: 0,0:00:00.50,0:00:01.00" in ass_content
    assert "{\\c&H0000FFFF&}Hello" in ass_content
    assert "{\\c&H0000FFFF&}world" in ass_content


def test_generate_ass_content_bad_config():
    word_json = [
        {"word": "Hi", "start": 0.0, "end": 1.0}
    ]
    config = {
        "caption_style": {
            "font_size": "not_an_int",
            "highlight_color": "bad_color"
        }
    }
    
    ass_content = generate_ass_content(word_json, config)
    assert "Arial,14" in ass_content  # Fallback to defaults
    assert "&H0000FFFF" in ass_content  # Default highlight color


@patch("pipeline.caption_generator.subprocess.run")
@patch("pipeline.caption_generator.get_gpu_flags")
@patch("pipeline.caption_generator.detect_cuda")
def test_generate_and_burn_success(mock_detect_cuda, mock_get_gpu_flags, mock_run, tmp_path):
    mock_detect_cuda.return_value = {"available": False}
    mock_get_gpu_flags.return_value = {"encoder": "libx264"}
    clip_path = tmp_path / "clip.mp4"
    clip_path.touch()
    
    word_json = [{"word": "Hi", "start": 0.0, "end": 1.0}]
    
    mock_proc = MagicMock()
    mock_proc.returncode = 0
    mock_run.return_value = mock_proc
    
    out_path = generate_and_burn(str(clip_path), word_json, {})
    
    assert out_path == str(tmp_path / "clip_captioned.mp4")
    mock_run.assert_called_once()
    cmd = mock_run.call_args[0][0]
    assert "-vf" in cmd
    assert "ass=" in " ".join(cmd)


@patch("pipeline.caption_generator.subprocess.run")
@patch("pipeline.caption_generator.get_gpu_flags")
@patch("pipeline.caption_generator.detect_cuda")
def test_generate_and_burn_ffmpeg_error(mock_detect_cuda, mock_get_gpu_flags, mock_run, tmp_path):
    mock_detect_cuda.return_value = {"available": False}
    mock_get_gpu_flags.return_value = {"encoder": "libx264"}
    clip_path = tmp_path / "clip.mp4"
    clip_path.touch()
    
    word_json = [{"word": "Hi", "start": 0.0, "end": 1.0}]
    
    mock_proc = MagicMock()
    mock_proc.returncode = 1
    mock_proc.stderr = "FFmpeg failed"
    mock_run.return_value = mock_proc
    
    with pytest.raises(CaptioningError, match="FFmpeg failed"):
        generate_and_burn(str(clip_path), word_json, {})


def test_generate_and_burn_missing_clip(tmp_path):
    clip_path = tmp_path / "missing.mp4"
    
    with pytest.raises(CaptioningError, match="Input clip not found"):
        generate_and_burn(str(clip_path), [], {})


@patch("pipeline.caption_generator.subprocess.run")
@patch("pipeline.caption_generator.get_gpu_flags")
@patch("pipeline.caption_generator.detect_cuda")
def test_generate_and_burn_timeout(mock_detect_cuda, mock_get_gpu_flags, mock_run, tmp_path):
    mock_detect_cuda.return_value = {"available": False}
    mock_get_gpu_flags.return_value = {"encoder": "libx264"}
    clip_path = tmp_path / "clip.mp4"
    clip_path.touch()
    
    mock_run.side_effect = subprocess.TimeoutExpired(cmd=["ffmpeg"], timeout=1800)
    
    with pytest.raises(CaptioningError, match="timed out"):
        generate_and_burn(str(clip_path), [], {"ffmpeg_timeout_sec": 1800})
