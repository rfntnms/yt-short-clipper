import os
import subprocess
import tempfile
import json
from typing import List, Dict, Any, Optional

from utils.logger import logger
from utils.gpu_detector import get_ffmpeg_hwaccel_flags

class CaptioningError(Exception):
    """Raised when subtitle generation or burn-in fails."""
    pass

def generate_ass_content(word_json: List[Dict[str, Any]], config: dict) -> str:
    """
    Generates ASS subtitle content from word-level JSON timestamps.
    Highlighting the active word with a specific color.
    """
    # Extract config or defaults
    style_cfg = config.get("caption_style", {})
    font_name = style_cfg.get("font_name", "Arial")
    font_size = style_cfg.get("font_size", 14)
    primary_color = style_cfg.get("primary_color", "&H00FFFFFF")  # White
    highlight_color = style_cfg.get("highlight_color", "&H0000FFFF")  # Yellow
    outline_color = style_cfg.get("outline_color", "&H00000000")
    back_color = style_cfg.get("back_color", "&H80000000")
    
    # ASS Header
    ass_lines = [
        "[Script Info]",
        "ScriptType: v4.00+",
        "PlayResX: 1080",
        "PlayResY: 1920",
        "WrapStyle: 1",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
        f"Style: Default,{font_name},{font_size},{primary_color},{primary_color},{outline_color},{back_color},-1,0,0,0,100,100,0,0,1,2,0,2,10,10,150,1",
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text"
    ]
    
    # Simple chunking by sentence/phrase for demo purposes
    # In a full robust implementation, we group words into logical lines
    # Here, we will just display words in small groups (e.g. 3-5 words)
    
    words = word_json
    
    def format_time(seconds: float) -> str:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        cs = int((seconds % 1) * 100)
        return f"{h}:{m:02d}:{s:02d}.{cs:02d}"

    # Simple grouper
    lines = []
    current_line = []
    
    for w in words:
        current_line.append(w)
        # break lines every 5 words
        if len(current_line) >= 5:
            lines.append(current_line)
            current_line = []
    if current_line:
        lines.append(current_line)

    for line_words in lines:
        if not line_words: continue
        start_time = format_time(line_words[0]["start"])
        end_time = format_time(line_words[-1]["end"])
        
        # We create a karaoke-like effect by generating overlapping lines 
        # or inline override tags. For simplicity, we use override tags.
        # Format: {\c&H0000FFFF&}highlighted{\c&H00FFFFFF&} normal normal
        
        # We need to emit one event per word highlight
        for i, active_word in enumerate(line_words):
            event_start = format_time(active_word["start"])
            event_end = format_time(active_word["end"])
            
            text_parts = []
            for j, w in enumerate(line_words):
                word_text = (
                    w["word"]
                    .strip()
                    .replace("\\", "\\\\")
                    .replace("{", "\\{")
                    .replace("}", "\\}")
                    .replace("\r", " ")
                    .replace("\n", " ")
                )
                if i == j:
                    text_parts.append(f"{{\\c{highlight_color}&}}{word_text}{{\\c{primary_color}&}}")
                else:
                    text_parts.append(word_text)
            
            full_text = " ".join(text_parts)
            ass_lines.append(f"Dialogue: 0,{event_start},{event_end},Default,,0,0,0,,{full_text}")

    return "\n".join(ass_lines)


def generate_and_burn(clip_path: str, word_json: List[Dict[str, Any]], config: dict) -> str:
    """
    Generates ASS subtitles from word_json and burns them into clip_path.
    Returns the path to the final video with burned-in subtitles.
    """
    if not os.path.exists(clip_path):
        raise CaptioningError(f"Input clip not found: {clip_path}")
        
    ass_content = generate_ass_content(word_json, config)
    
    # Create temp ASS file
    fd, ass_path = tempfile.mkstemp(suffix=".ass", text=True)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            f.write(ass_content)
            
        logger.info(f"Generated ASS subtitles at {ass_path}")
        
        base, ext = os.path.splitext(clip_path)
        output_path = f"{base}_captioned{ext}"
        
        # Prepare FFmpeg command
        hw_flags = get_ffmpeg_hwaccel_flags()
        input_flags = hw_flags.get("input_flags", [])
        vcodec = hw_flags.get("vcodec", "libx264")
        preset = hw_flags.get("preset", "fast")
        
        # Subtitles filter needs paths escaped properly for FFmpeg
        # We wrap in single quotes to protect spaces, but must escape inner single quotes
        escaped_ass_path = (
            ass_path
            .replace("\\", "\\\\")
            .replace(":", "\\:")
            .replace("'", "\\'")
        )
        
        cmd = [
            "ffmpeg", "-y"
        ]
        cmd.extend(input_flags)
        cmd.extend([
            "-i", clip_path,
            "-vf", f"ass='{escaped_ass_path}'",
            "-c:v", vcodec,
            "-preset", preset,
            "-c:a", "copy",
            output_path
        ])
        
        logger.debug(f"Running subtitle burn-in: {' '.join(cmd)}")
        
        timeout_sec = config.get("ffmpeg_timeout_sec", 1800)
        try:
            result = subprocess.run(
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout_sec
            )
        except subprocess.TimeoutExpired as e:
            logger.error(f"FFmpeg captioning timed out after {timeout_sec}s")
            raise CaptioningError(f"FFmpeg captioning timed out after {timeout_sec}s") from e
        
        if result.returncode != 0:
            logger.error(f"FFmpeg captioning failed. Stderr:\n{result.stderr}")
            raise CaptioningError(f"FFmpeg captioning failed: {result.stderr}")
            
        logger.info(f"Successfully burned subtitles into {output_path}")
        return output_path
        
    finally:
        # Cleanup temporary ASS file
        if os.path.exists(ass_path):
            os.remove(ass_path)
            logger.debug(f"Cleaned up temp subtitle file: {ass_path}")

