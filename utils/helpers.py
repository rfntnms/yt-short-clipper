"""
Helper utility functions for YT Short Clipper
"""

import sys
import re
import shutil
from pathlib import Path


def get_app_dir():
    """Get application data directory
    
    On macOS (.app bundle): ~/Library/Application Support/YTShortClipper/
    On Windows/Linux or dev mode: directory containing the executable/script
    
    This ensures user data (config, downloads, output) persists across app updates on macOS.
    """
    if getattr(sys, 'frozen', False):
        if sys.platform == "darwin":
            # macOS: use Application Support (survives .app replacement)
            app_support = Path.home() / "Library" / "Application Support" / "YTShortClipper"
            app_support.mkdir(parents=True, exist_ok=True)
            return app_support
        return Path(sys.executable).parent
    return Path(__file__).parent.parent


def get_bundle_dir():
    """Get bundled resources directory"""
    if getattr(sys, 'frozen', False):
        return Path(sys._MEIPASS) if hasattr(sys, '_MEIPASS') else get_app_dir()
    return get_app_dir()


def get_ffmpeg_path():
    """Get FFmpeg executable path
    
    Checks in order:
    1. Bundled ffmpeg in app_dir/ffmpeg/ folder (downloaded via Library page)
    2. ffmpeg in system PATH
    3. Default "ffmpeg" command
    """
    app_dir = get_app_dir()
    
    # Check bundled ffmpeg (works for both frozen and development)
    if sys.platform.startswith('win'):
        bundled = app_dir / "ffmpeg" / "ffmpeg.exe"
    else:
        bundled = app_dir / "ffmpeg" / "ffmpeg"
    
    if bundled.exists():
        return str(bundled)
    
    # Try to find ffmpeg in PATH
    ffmpeg_in_path = shutil.which("ffmpeg")
    if ffmpeg_in_path:
        return ffmpeg_in_path
    
    # Fallback to command name
    return "ffmpeg"


def get_ytdlp_path():
    """Get yt-dlp executable path or check if module is available
    
    Checks in order:
    1. yt-dlp Python module (preferred - bundled with PyInstaller)
    2. Bundled yt-dlp.exe (Windows)
    3. yt-dlp in system PATH
    4. Default "yt-dlp" command
    
    Returns:
        str: Path to yt-dlp executable, or "yt_dlp_module" if using Python module
    """
    # First check if yt-dlp is available as Python module
    try:
        import yt_dlp
        return "yt_dlp_module"  # Special marker to use module instead of subprocess
    except ImportError:
        pass
    
    if getattr(sys, 'frozen', False):
        bundled = get_app_dir() / "yt-dlp.exe"
        if bundled.exists():
            return str(bundled)
    
    # Try to find yt-dlp in PATH
    yt_dlp_path = shutil.which("yt-dlp")
    if yt_dlp_path:
        return yt_dlp_path
    
    # Fallback to command name (will work if it's in system PATH)
    return "yt-dlp"


def is_ytdlp_module_available():
    """Check if yt-dlp Python module is available"""
    try:
        import yt_dlp
        return True
    except ImportError:
        return False


def get_deno_path():
    """Get Deno executable path (required for yt-dlp --remote-components)
    
    Checks in order:
    1. Bundled deno in app_dir/bin/ folder (downloaded via Library page)
    2. deno in system PATH
    3. None if not found
    """
    app_dir = get_app_dir()
    
    # Check bundled deno (works for both frozen and development)
    if sys.platform.startswith('win'):
        bundled = app_dir / "bin" / "deno.exe"
    else:
        bundled = app_dir / "bin" / "deno"
    
    if bundled.exists():
        return str(bundled)
    
    # Try to find deno in PATH
    deno_path = shutil.which("deno")
    if deno_path:
        return deno_path
    
    # Not found
    return None


def extract_video_id(url: str) -> str:
    """Extract YouTube video ID from URL"""
    patterns = [
        r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',
        r'(?:youtu\.be\/)([0-9A-Za-z_-]{11})'
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


def hex_to_rgb(hex_color: str):
    """Convert a #RRGGBB or #RGB string to an (R, G, B) tuple. Falls back to white on bad input."""
    if not isinstance(hex_color, str):
        return (255, 255, 255)
    s = hex_color.strip().lstrip("#")
    if len(s) == 3:
        s = "".join(ch * 2 for ch in s)
    if len(s) != 6:
        return (255, 255, 255)
    try:
        return (int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16))
    except ValueError:
        return (255, 255, 255)


def ensure_binaries_in_path():
    """Ensure Deno and FFmpeg bundled binaries are added to the environment PATH.
    
    Returns:
        dict: Information about which paths were found/added
            - deno_path: path to Deno binary or None
            - ffmpeg_path: path to FFmpeg binary or None
    """
    import os
    
    result = {
        "deno_path": get_deno_path(),
        "ffmpeg_path": get_ffmpeg_path()
    }
    
    if result["deno_path"] and Path(result["deno_path"]).exists():
        deno_dir = str(Path(result["deno_path"]).parent)
        if "PATH" in os.environ:
            if deno_dir not in os.environ["PATH"]:
                os.environ["PATH"] = f"{deno_dir}{os.pathsep}{os.environ['PATH']}"
        else:
            os.environ["PATH"] = deno_dir
            
    return result


def parse_timestamp(ts: str) -> float:
    """Convert timestamp to seconds"""
    ts = ts.replace(",", ".")
    parts = ts.split(":")
    return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
