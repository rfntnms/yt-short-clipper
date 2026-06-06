"""yt-dlp download wrapper for YouTube video downloading.

First module in the orchestrator pipeline.
Returns (video_path, srt_path) where srt_path is None when no captions exist.
"""
from pathlib import Path
from typing import Any, Optional, cast

import yt_dlp
from yt_dlp.utils import DownloadError as YTDlpError

from utils.logger import logger


class DownloadError(Exception):
    """Raised when video download fails (invalid URL, network, age-restriction)."""


def download(
    url: str,
    output_dir: Path,
    cookies_path: Optional[Path] = None,
) -> tuple[Path, Optional[Path]]:
    """Download a YouTube video and its auto-generated subtitles.

    Args:
        url: YouTube video URL.
        output_dir: Directory to write the downloaded file(s).
        cookies_path: Optional path to a cookies.txt file for age-restricted videos.

    Returns:
        Tuple (video_path, srt_path_or_None).

    Raises:
        DownloadError: If the download fails for any reason.
    """
    # Build yt-dlp options
    ydl_opts: dict[str, Any] = {
        "format": "bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/best[height<=1080][ext=mp4]/best",
        "outtmpl": str(output_dir / "%(title)s.%(ext)s"),
        "merge_output_format": "mp4",
        "writesubtitles": True,
        "writeautomaticsub": True,
        "subtitleslangs": ["en"],
        "quiet": True,
        "no_warnings": True,
    }

    if cookies_path is not None:
        if not cookies_path.exists():
            raise DownloadError(f"Cookies file not found: {cookies_path}")
        ydl_opts["cookiefile"] = str(cookies_path)

    logger.info("Starting download for %s", url)

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
    except YTDlpError as exc:
        logger.error("Download failed for %s: %s", url, exc)
        raise DownloadError(str(exc)) from exc

    # Resolve the output video path
    video_title: str = info.get("title", "video")
    video_ext: str = info.get("ext", "mp4")
    video_path = output_dir / f"{video_title}.{video_ext}"

    # Resolve the subtitle path if available
    srt_path: Optional[Path] = None
    subtitles_info = info.get("requested_subtitles")
    if subtitles_info and "en" in subtitles_info:
        sub_url = subtitles_info["en"].get("url")
        if sub_url:
            srt_path = output_dir / f"{video_title}.en.srt"

    if not video_path.exists():
        # Fallback: try to find any mp4 in output_dir
        candidates = list(output_dir.glob("*.mp4"))
        if candidates:
            video_path = candidates[0]

    logger.info("Download complete: %s", video_path.name)
    return video_path, srt_path


__all__ = ["download", "DownloadError"]
