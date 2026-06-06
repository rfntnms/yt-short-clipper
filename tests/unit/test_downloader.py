"""Unit tests for pipeline/downloader.py."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from yt_dlp.utils import DownloadError as YTDlpError

from pipeline.downloader import DownloadError, download


@pytest.fixture
def temp_output(tmp_path: Path) -> Path:
    """Fixture to provide a temporary output directory."""
    out_dir = tmp_path / "output"
    out_dir.mkdir()
    return out_dir


# ── test 1: returns correct paths on success ────────────────────────────
@patch("pipeline.downloader.yt_dlp.YoutubeDL")
def test_download_success(mock_ytdl_cls: MagicMock, temp_output: Path) -> None:
    # Setup mock yt-dlp to pretend it succeeded
    mock_instance = mock_ytdl_cls.return_value
    mock_instance.__enter__.return_value = mock_instance
    mock_instance.extract_info.return_value = {
        "id": "abc123xyz",
        "title": "Test Video",
        "ext": "mp4",
        "requested_subtitles": {
            "en": {"ext": "vtt", "url": "https://example.com/sub.vtt"}
        },
    }
    mock_instance.prepare_filename.return_value = str(temp_output / "Test Video.mp4")

    # Call the downloader
    video_path, srt_path = download("https://youtube.com/watch?v=abc123xyz", temp_output)

    # Assertions
    assert isinstance(video_path, Path)
    assert video_path.name == "Test Video.mp4"
    assert video_path.parent == temp_output
    # srt_path is returned from requested_subtitles side-effect (converted to .srt conceptually)
    assert srt_path is not None
    assert srt_path.name.endswith(".srt")


# ── test 2: raises typed DownloadError on yt-dlp exception ───────────────
@patch("pipeline.downloader.yt_dlp.YoutubeDL")
def test_download_raises_downloaderror(mock_ytdl_cls: MagicMock, temp_output: Path) -> None:
    mock_instance = mock_ytdl_cls.return_value
    mock_instance.__enter__.return_value = mock_instance
    # Simulate network failure or invalid URL
    mock_instance.extract_info.side_effect = YTDlpError("Sign in to confirm you're not a bot")

    with pytest.raises(DownloadError, match="Sign in to confirm"):
        download("https://youtube.com/watch?v=invalid", temp_output)


# ── test 3: uses cookies.txt when provided ──────────────────────────────
@patch("pipeline.downloader.yt_dlp.YoutubeDL")
def test_download_uses_cookies(mock_ytdl_cls: MagicMock, temp_output: Path) -> None:
    mock_instance = mock_ytdl_cls.return_value
    mock_instance.__enter__.return_value = mock_instance
    mock_instance.extract_info.return_value = {
        "id": "abc123xyz",
        "title": "Test Video",
        "ext": "mp4",
    }
    mock_instance.prepare_filename.return_value = str(temp_output / "Test Video.mp4")

    cookies_path = temp_output / "cookies.txt"
    cookies_path.write_text("dummy cookies")

    download("https://youtube.com/watch?v=abc123xyz", temp_output, cookies_path=cookies_path)

    # Verify YoutubeDL was instantiated with cookiefile inside the params dict
    # (yt-dlp's YoutubeDL constructor accepts `params` as a positional argument)
    call_args = mock_ytdl_cls.call_args
    params = call_args.args[0] if call_args.args else call_args.kwargs.get("params", {})
    assert params.get("cookiefile") == str(cookies_path)


# ── test 4: handles no subtitles gracefully ────────────────────────────
@patch("pipeline.downloader.yt_dlp.YoutubeDL")
def test_download_no_subtitles(mock_ytdl_cls: MagicMock, temp_output: Path) -> None:
    mock_instance = mock_ytdl_cls.return_value
    mock_instance.__enter__.return_value = mock_instance
    mock_instance.extract_info.return_value = {
        "id": "abc123xyz",
        "title": "Test Video",
        "ext": "mp4",
        "requested_subtitles": None, # No subtitles available
    }
    mock_instance.prepare_filename.return_value = str(temp_output / "Test Video.mp4")

    video_path, srt_path = download("https://youtube.com/watch?v=abc123xyz", temp_output)

    assert video_path.name == "Test Video.mp4"
    assert srt_path is None
