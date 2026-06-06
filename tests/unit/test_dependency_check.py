import shutil
import pytest

from utils.dependency_check import check_ffmpeg, check_yt_dlp, check_all


def test_check_ffmpeg_raises_when_missing(monkeypatch):
    """check_ffmpeg should raise RuntimeError if ffmpeg is not in PATH."""
    monkeypatch.setattr(shutil, "which", lambda x: None if x == "ffmpeg" else f"/usr/bin/{x}")
    
    with pytest.raises(RuntimeError) as exc_info:
        check_ffmpeg()
    
    assert "FFmpeg not found in PATH" in str(exc_info.value)
    assert "apt-get install ffmpeg" in str(exc_info.value)


def test_check_ffmpeg_passes_when_present(monkeypatch):
    """check_ffmpeg should not raise if ffmpeg is found."""
    monkeypatch.setattr(shutil, "which", lambda x: f"/usr/bin/{x}" if x == "ffmpeg" else None)
    
    # Should not raise
    check_ffmpeg()


def test_check_yt_dlp_raises_when_missing(monkeypatch):
    """check_yt_dlp should raise RuntimeError if yt-dlp is not in PATH."""
    monkeypatch.setattr(shutil, "which", lambda x: None if x == "yt-dlp" else f"/usr/bin/{x}")
    
    with pytest.raises(RuntimeError) as exc_info:
        check_yt_dlp()
    
    assert "yt-dlp not found in PATH" in str(exc_info.value)


def test_check_yt_dlp_passes_when_present(monkeypatch):
    """check_yt_dlp should not raise if yt-dlp is found."""
    monkeypatch.setattr(shutil, "which", lambda x: f"/usr/bin/{x}" if x == "yt-dlp" else None)
    
    # Should not raise
    check_yt_dlp()


def test_check_all_raises_when_ffmpeg_missing(monkeypatch):
    """check_all should raise on first missing dependency (ffmpeg)."""
    monkeypatch.setattr(shutil, "which", lambda x: None)
    
    with pytest.raises(RuntimeError) as exc_info:
        check_all()
    
    assert "FFmpeg" in str(exc_info.value)


def test_check_all_raises_when_yt_dlp_missing(monkeypatch):
    """check_all should raise when yt-dlp is missing but ffmpeg is present."""
    monkeypatch.setattr(
        shutil, "which", 
        lambda x: "/usr/bin/ffmpeg" if x == "ffmpeg" else None
    )
    
    with pytest.raises(RuntimeError) as exc_info:
        check_all()
    
    assert "yt-dlp" in str(exc_info.value)


def test_check_all_passes_when_all_present(monkeypatch):
    """check_all should not raise when both dependencies are present."""
    monkeypatch.setattr(
        shutil, "which", 
        lambda x: f"/usr/bin/{x}" if x in ("ffmpeg", "yt-dlp") else None
    )
    
    # Should not raise
    check_all()
