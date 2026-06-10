"""Unit tests for pipeline components."""
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
from pipeline.downloader import download, DownloadError
from pipeline.transcriber import transcribe, TranscriptionError
from pipeline.highlight_detector import find_highlights, Highlight, HighlightDetectionError
from pipeline.video_processor import cut, convert_to_portrait, probe_dimensions, compute_center_crop, VideoProcessingError
from yt_dlp.utils import DownloadError as YTDlpError


# --- downloader.py tests ---

@patch("pipeline.downloader.yt_dlp.YoutubeDL")
def test_download_success(mock_ytdl_class, tmp_path):
    mock_ydl = MagicMock()
    mock_ytdl_class.return_value.__enter__.return_value = mock_ydl
    
    mock_ydl.extract_info.return_value = {"requested_subtitles": {"en": {}}}
    mock_ydl.prepare_filename.return_value = str(tmp_path / "video.mp4")
    
    video_file = tmp_path / "video.mp4"
    video_file.touch()
    srt_file = tmp_path / "video.en.srt"
    srt_file.touch()
    
    video_path, srt_path = download("https://youtube.com/watch?v=123", tmp_path)
    
    assert video_path == video_file
    assert srt_path == srt_file
    mock_ydl.extract_info.assert_called_once_with("https://youtube.com/watch?v=123", download=True)


@patch("pipeline.downloader.yt_dlp.YoutubeDL")
def test_download_yt_dlp_error(mock_ytdl_class, tmp_path):
    mock_ydl = MagicMock()
    mock_ytdl_class.return_value.__enter__.return_value = mock_ydl
    
    mock_ydl.extract_info.side_effect = YTDlpError("Network error")
    
    with pytest.raises(DownloadError, match="Network error"):
        download("https://youtube.com/watch?v=123", tmp_path)


def test_download_missing_cookies(tmp_path):
    missing_cookie_path = tmp_path / "missing_cookies.txt"
    with pytest.raises(DownloadError, match="Cookies file not found"):
        download("https://youtube.com/watch?v=123", tmp_path, cookies_path=missing_cookie_path)


@patch("pipeline.downloader.yt_dlp.YoutubeDL")
def test_download_file_not_found(mock_ytdl_class, tmp_path):
    mock_ydl = MagicMock()
    mock_ytdl_class.return_value.__enter__.return_value = mock_ydl
    
    mock_ydl.extract_info.return_value = {}
    mock_ydl.prepare_filename.return_value = str(tmp_path / "video.mp4")
    
    # Do not create the video file to trigger the missing file check
    with pytest.raises(DownloadError, match="Downloaded video path not found"):
        download("https://youtube.com/watch?v=123", tmp_path)


# --- transcriber.py tests ---

def test_transcribe_missing_video(tmp_path):
    with pytest.raises(TranscriptionError, match="Video file not found"):
        transcribe(tmp_path / "missing.mp4", {})


@patch("pipeline.transcriber.get_client")
def test_transcribe_success_top_level_words(mock_get_client, tmp_path):
    video_path = tmp_path / "video.mp4"
    video_path.touch()
    
    mock_client = MagicMock()
    mock_get_client.return_value = mock_client
    
    mock_response = MagicMock()
    mock_response.words = [
        {"word": "Hello", "start": 0.0, "end": 0.5},
        {"word": "world", "start": 0.5, "end": 1.0}
    ]
    mock_client.audio.transcriptions.create.return_value = mock_response
    
    words = transcribe(video_path, {})
    assert len(words) == 2
    assert words[0]["word"] == "Hello"


@patch("pipeline.transcriber.get_client")
def test_transcribe_success_nested_segments(mock_get_client, tmp_path):
    video_path = tmp_path / "video.mp4"
    video_path.touch()
    
    mock_client = MagicMock()
    mock_get_client.return_value = mock_client
    
    mock_response = MagicMock()
    mock_response.words = None
    mock_segment = MagicMock()
    mock_segment.words = [
        {"word": "Nested", "start": 1.0, "end": 1.5}
    ]
    mock_response.segments = [mock_segment]
    mock_client.audio.transcriptions.create.return_value = mock_response
    
    words = transcribe(video_path, {})
    assert len(words) == 1
    assert words[0]["word"] == "Nested"


@patch("pipeline.transcriber.get_client")
def test_transcribe_api_error(mock_get_client, tmp_path):
    video_path = tmp_path / "video.mp4"
    video_path.touch()
    
    mock_client = MagicMock()
    mock_get_client.return_value = mock_client
    
    mock_client.audio.transcriptions.create.side_effect = Exception("API down")
    
    with pytest.raises(TranscriptionError, match="API down"):
        transcribe(video_path, {})


@patch("pipeline.transcriber.get_client")
def test_transcribe_missing_words(mock_get_client, tmp_path):
    video_path = tmp_path / "video.mp4"
    video_path.touch()
    
    mock_client = MagicMock()
    mock_get_client.return_value = mock_client
    
    mock_response = MagicMock()
    mock_response.words = None
    mock_response.segments = []
    mock_client.audio.transcriptions.create.return_value = mock_response
    
    with pytest.raises(TranscriptionError, match="did not include word-level timestamps"):
        transcribe(video_path, {})


@patch("pipeline.transcriber.get_client")
def test_transcribe_invalid_timestamps(mock_get_client, tmp_path):
    video_path = tmp_path / "video.mp4"
    video_path.touch()
    
    mock_client = MagicMock()
    mock_get_client.return_value = mock_client
    
    mock_response = MagicMock()
    mock_response.words = [
        {"word": "Bad", "start": 1.0, "end": 0.5}  # end < start
    ]
    mock_client.audio.transcriptions.create.return_value = mock_response
    
    with pytest.raises(TranscriptionError, match="invalid word timestamp order"):
        transcribe(video_path, {})


# --- highlight_detector.py tests ---

@patch("pipeline.highlight_detector._load_system_prompt")
@patch("pipeline.highlight_detector.get_client")
def test_find_highlights_success(mock_get_client, mock_load_prompt):
    mock_load_prompt.return_value = "System prompt"
    
    mock_client = MagicMock()
    mock_get_client.return_value = mock_client
    
    mock_choice = MagicMock()
    # Provide valid JSON output
    mock_choice.message.content = '''
    [
      {"start": 10.5, "end": 20.0, "hook_text": "Awesome clip", "score": 9},
      {"start": 30.0, "end": 45.0, "hook_text": "Good clip", "score": 7}
    ]
    '''
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_client.chat.completions.create.return_value = mock_response
    
    highlights = find_highlights("transcript", {})
    
    assert len(highlights) == 2
    assert highlights[0].score == 9  # Sorted descending
    assert highlights[1].score == 7


@patch("pipeline.highlight_detector._load_system_prompt")
@patch("pipeline.highlight_detector.get_client")
def test_find_highlights_retry_on_bad_json(mock_get_client, mock_load_prompt):
    mock_load_prompt.return_value = "System prompt"
    
    mock_client = MagicMock()
    mock_get_client.return_value = mock_client
    
    # First call returns invalid JSON
    bad_choice = MagicMock()
    bad_choice.message.content = "I found some highlights for you, but this is not JSON"
    bad_response = MagicMock()
    bad_response.choices = [bad_choice]
    
    # Second call returns valid JSON
    good_choice = MagicMock()
    good_choice.message.content = '[{"start": 10.0, "end": 20.0, "hook_text": "Fixed clip", "score": 8}]'
    good_response = MagicMock()
    good_response.choices = [good_choice]
    
    mock_client.chat.completions.create.side_effect = [bad_response, good_response]
    
    highlights = find_highlights("transcript", {})
    assert len(highlights) == 1
    assert highlights[0].hook_text == "Fixed clip"
    assert mock_client.chat.completions.create.call_count == 2


@patch("pipeline.highlight_detector._load_system_prompt")
@patch("pipeline.highlight_detector.get_client")
def test_find_highlights_fail_after_retries(mock_get_client, mock_load_prompt):
    mock_load_prompt.return_value = "System prompt"
    
    mock_client = MagicMock()
    mock_get_client.return_value = mock_client
    
    bad_choice = MagicMock()
    bad_choice.message.content = "Still not JSON"
    bad_response = MagicMock()
    bad_response.choices = [bad_choice]
    
    mock_client.chat.completions.create.return_value = bad_response
    
    with pytest.raises(HighlightDetectionError, match="Malformed LLM response"):
        find_highlights("transcript", {})
    
    assert mock_client.chat.completions.create.call_count == 3  # Initial + 2 retries


def test_find_highlights_missing_prompt():
    with patch("pipeline.highlight_detector._PROMPT_PATH") as mock_path:
        mock_path.exists.return_value = False
        with pytest.raises(HighlightDetectionError, match="System prompt not found"):
            find_highlights("transcript", {})


# --- video_processor.py tests ---

def test_compute_center_crop():
    # 1920x1080 (16:9) -> crop width to match 9:16 target (607x1080)
    w, h, x, y = compute_center_crop(1920, 1080)
    assert h == 1080
    assert w == 606  # rounded to even
    assert y == 0
    assert x == (1920 - 606) // 2

    # 1080x1920 (9:16) -> no crop
    w, h, x, y = compute_center_crop(1080, 1920)
    assert w == 1080
    assert h == 1920
    assert x == 0
    assert y == 0
    
    # 1080x1080 (1:1) -> crop width
    w, h, x, y = compute_center_crop(1080, 1080)
    assert h == 1080
    assert w == 606
    assert x == (1080 - 606) // 2
    assert y == 0


@patch("pipeline.video_processor.subprocess.run")
def test_probe_dimensions_success(mock_run):
    mock_proc = MagicMock()
    mock_proc.returncode = 0
    mock_proc.stdout = '{"streams": [{"width": 1920, "height": 1080}]}'
    mock_run.return_value = mock_proc
    
    w, h = probe_dimensions("video.mp4")
    assert w == 1920
    assert h == 1080


@patch("pipeline.video_processor.subprocess.run")
def test_probe_dimensions_rotated(mock_run):
    mock_proc = MagicMock()
    mock_proc.returncode = 0
    # Provide rotation tag
    mock_proc.stdout = '{"streams": [{"width": 1920, "height": 1080, "tags": {"rotate": "90"}}]}'
    mock_run.return_value = mock_proc
    
    w, h = probe_dimensions("video.mp4")
    assert w == 1080
    assert h == 1920


@patch("pipeline.video_processor.subprocess.run")
def test_probe_dimensions_ffprobe_error(mock_run):
    mock_proc = MagicMock()
    mock_proc.returncode = 1
    mock_proc.stderr = "Invalid file"
    mock_run.return_value = mock_proc
    
    with pytest.raises(VideoProcessingError, match="ffprobe failed"):
        probe_dimensions("video.mp4")


@patch("pipeline.video_processor._run_ffmpeg")
def test_cut_success(mock_run_ffmpeg):
    highlight = Highlight(start=10.0, end=15.0, hook_text="hook", score=8)
    res = cut("video.mp4", highlight, "clip.mp4")
    
    assert res == "clip.mp4"
    mock_run_ffmpeg.assert_called_once()
    args, kwargs = mock_run_ffmpeg.call_args
    cmd = args[0]
    assert "-ss" in cmd
    assert "10.0" in cmd
    assert "-t" in cmd
    assert "5.0" in cmd  # duration


def test_cut_invalid_timestamps():
    highlight = Highlight(start=15.0, end=10.0, hook_text="hook", score=8)
    with pytest.raises(VideoProcessingError, match="End must be > start"):
        cut("video.mp4", highlight, "clip.mp4")


@patch("pipeline.video_processor.probe_dimensions")
@patch("pipeline.video_processor.detect_cuda")
@patch("pipeline.video_processor._run_ffmpeg")
def test_convert_to_portrait_success(mock_run_ffmpeg, mock_detect_cuda, mock_probe):
    mock_probe.return_value = (1920, 1080)
    mock_detect_cuda.return_value = {"cuda_available": False}
    
    res = convert_to_portrait("clip.mp4", {}, "portrait.mp4")
    
    assert res == "portrait.mp4"
    mock_run_ffmpeg.assert_called_once()
    args, kwargs = mock_run_ffmpeg.call_args
    cmd = args[0]
    assert "-vf" in cmd
    assert "-c:v" in cmd
    assert "libx264" in cmd
