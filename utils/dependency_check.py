import shutil

def check_ffmpeg() -> None:
    """Validate that ffmpeg is available in PATH.
    
    Raises:
        RuntimeError: If ffmpeg is not found.
    """
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        raise RuntimeError(
            "FFmpeg not found in PATH. "
            "Please install FFmpeg on your system:\n"
            "  Ubuntu/Debian: sudo apt-get install ffmpeg\n"
            "  macOS: brew install ffmpeg\n"
            "  Docker: already included in the base image"
        )

def check_yt_dlp() -> None:
    """Validate that yt-dlp is available in PATH.
    
    Raises:
        RuntimeError: If yt-dlp is not found.
    """
    yt_dlp_path = shutil.which("yt-dlp")
    if yt_dlp_path is None:
        raise RuntimeError(
            "yt-dlp not found in PATH. "
            "Please install yt-dlp:\n"
            "  Ubuntu/Debian: sudo apt-get install yt-dlp\n"
            "  pip: pip install yt-dlp\n"
            "  Docker: already included in the base image"
        )

def check_all() -> None:
    """Run all dependency checks. Raises RuntimeError on first missing dependency."""
    check_ffmpeg()
    check_yt_dlp()

__all__ = ['check_ffmpeg', 'check_yt_dlp', 'check_all']
