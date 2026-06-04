"""
Auto Clipper Core - Processing logic
Refactored to use OpenAI Whisper API instead of local model
"""

import subprocess
import os
import re
import threading
import json
import cv2
import numpy as np
import tempfile
import sys
import time
from pathlib import Path
from datetime import datetime
from openai import OpenAI, APIConnectionError, RateLimitError, APIStatusError
from utils.helpers import ensure_binaries_in_path, hex_to_rgb

# Setup Deno and FFmpeg in PATH before importing yt-dlp
ensure_binaries_in_path()


# Hide console window on Windows
SUBPROCESS_FLAGS = 0
if sys.platform == "win32":
    SUBPROCESS_FLAGS = subprocess.CREATE_NO_WINDOW


class SubtitleNotFoundError(Exception):
    """Raised when no subtitle is available for the video.
    
    Carries context needed to offer Whisper transcription fallback.
    """
    def __init__(self, message: str, video_path: str = None, video_info: dict = None, session_dir: str = None):
        super().__init__(message)
        self.video_path = video_path
        self.video_info = video_info or {}
        self.session_dir = session_dir



class AutoClipperCore:
    """Core processing logic for Auto Clipper"""
    
    def __init__(
        self,
        client: OpenAI,
        ffmpeg_path: str = "ffmpeg",
        ytdlp_path: str = "yt-dlp",
        output_dir: str = "output",
        model: str = "gpt-4.1",
        tts_model: str = "tts-1",
        temperature: float = 1.0,
        system_prompt: str = None,
        watermark_settings: dict = None,
        credit_watermark_settings: dict = None,
        hook_style_settings: dict = None,
        face_tracking_mode: str = "opencv",
        mediapipe_settings: dict = None,
        ai_providers: dict = None,
        subtitle_language: str = "id",
        performance_settings: dict = None,
        log_callback=None,
        progress_callback=None,
        token_callback=None,
        cancel_check=None
    ):
        # Multi-provider support
        self.ai_providers = ai_providers or {}
        
        # Create separate clients for each provider
        if self.ai_providers:
            # Highlight Finder client
            hf_config = self.ai_providers.get("highlight_finder", {})
            self.highlight_client = OpenAI(
                api_key=hf_config.get("api_key", ""),
                base_url=hf_config.get("base_url", "https://api.openai.com/v1")
            )
            self.model = hf_config.get("model", model)
            
            # Caption Maker client (Whisper) — use longer timeout for large audio uploads
            cm_config = self.ai_providers.get("caption_maker", {})
            self.caption_client = OpenAI(
                api_key=cm_config.get("api_key", ""),
                base_url=cm_config.get("base_url", "https://api.openai.com/v1"),
                timeout=600.0  # 10 minutes for large audio files
            )
            self.whisper_model = cm_config.get("model", "whisper-1")
            
            # Hook Maker client (TTS)
            hm_config = self.ai_providers.get("hook_maker", {})
            self.tts_client = OpenAI(
                api_key=hm_config.get("api_key", ""),
                base_url=hm_config.get("base_url", "https://api.openai.com/v1")
            )
            self.tts_model = hm_config.get("model", tts_model)
        else:
            # Fallback to single client (backward compatibility)
            self.highlight_client = client
            self.caption_client = client
            self.tts_client = client
            self.model = model
            self.tts_model = tts_model
            self.whisper_model = "whisper-1"
        
        # Keep original client for backward compatibility
        self.client = client
        
        self.ffmpeg_path = ffmpeg_path
        self.ytdlp_path = ytdlp_path
        self.output_dir = Path(output_dir)
        self.temperature = temperature
        self.system_prompt = system_prompt or self.get_default_prompt()
        self.watermark_settings = watermark_settings or {"enabled": False}
        self.credit_watermark_settings = credit_watermark_settings or {"enabled": False}
        self.hook_style_settings = hook_style_settings or {}
        self.channel_name = ""  # Will be set after download
        self.face_tracking_mode = face_tracking_mode
        self.mediapipe_settings = mediapipe_settings or {
            "lip_activity_threshold": 0.15,
            "switch_threshold": 0.3,
            "min_shot_duration": 90,
            "center_weight": 0.3
        }
        self.subtitle_language = subtitle_language
        self.performance_settings = self._normalize_performance_settings(performance_settings)
        self.performance_profile = self.performance_settings.get("profile", "balanced")
        self.detection_engine = self.performance_settings.get("detection_engine") or self.face_tracking_mode
        self.log = log_callback or print
        self.set_progress = progress_callback or (lambda s, p: None)
        self.report_tokens = token_callback or (lambda gi, go, w, t: None)
        self.is_cancelled = cancel_check or (lambda: False)
        
        # GPU acceleration settings
        self.gpu_enabled = False
        self.gpu_encoder_args = []
        self.gpu_codec = self.performance_settings.get("codec", "h264")
        self.gpu_decode_enabled = bool(self.performance_settings.get("decode_enabled", True))
        
        # MediaPipe Face Mesh (lazy loaded)
        self.mp_face_mesh = None
        self.mp_drawing = None
        
        # Create temp directory
        self.temp_dir = self.output_dir / "_temp"
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        from services.download_service import DownloadService
        self.download_service = DownloadService(
            temp_dir=self.temp_dir,
            output_dir=self.output_dir,
            cookies_file=None,
            subtitle_language=self.subtitle_language,
            ytdlp_path=self.ytdlp_path,
            log_callback=self.log,
            progress_callback=self.set_progress,
            is_cancelled_callback=self.is_cancelled,
            youtube_api_key="",
            performance_settings=self.performance_settings
        )
        self.download_service.performance_settings["prefer_gpu"] = False

    def _normalize_performance_settings(self, settings: dict = None) -> dict:
        """Return backward-compatible performance settings with safe defaults."""
        defaults = {
            "profile": "balanced",
            "encoder": "auto",
            "codec": "h264",
            "detection_engine": self.face_tracking_mode,
            "detection_interval": 10,
            "prefer_gpu": True,
            "fallback_to_cpu": True,
            "decode_enabled": True,
            "test_encoder": False,
            "yolo_model_path": "",
            "allow_yolo_download": False,
        }
        merged = defaults.copy()
        if isinstance(settings, dict):
            merged.update({k: v for k, v in settings.items() if v is not None})

        if merged.get("profile") not in ("quality", "balanced", "fast"):
            merged["profile"] = "balanced"
        if merged.get("codec") not in ("h264", "hevc"):
            merged["codec"] = "h264"
        if not isinstance(merged.get("detection_interval"), int) or merged["detection_interval"] <= 0:
            merged["detection_interval"] = {"quality": 5, "balanced": 10, "fast": 30}[merged["profile"]]
        return merged
    
    def enable_gpu_acceleration(self, enabled: bool = True):
        """Enable or disable GPU acceleration for video encoding"""
        enabled = enabled and bool(self.performance_settings.get("prefer_gpu", True))
        self.gpu_enabled = enabled
        self.gpu_info = {'type': None}
        if hasattr(self, "download_service"):
            self.download_service.performance_settings["prefer_gpu"] = False
        
        if enabled:
            try:
                from utils.gpu_detector import GPUDetector
                detector = GPUDetector(self.ffmpeg_path)
                preferred_encoder = self.performance_settings.get("encoder", "auto")
                if preferred_encoder == "cpu":
                    self.gpu_enabled = False
                    self.gpu_encoder_args = []
                    self.log("  CPU encoding selected in performance settings")
                    return

                self.gpu_encoder_args = detector.get_encoder_args(
                    use_gpu=True,
                    preferred_codec=self.gpu_codec,
                    encoder=preferred_encoder
                )
                self.gpu_info = detector.detect_gpu()
                selected_encoder = self._extract_video_encoder(self.gpu_encoder_args)
                if selected_encoder in ("libx264", "libx265"):
                    self.log("  No usable GPU encoder found; using CPU encoding")
                    self.gpu_enabled = False
                    self.gpu_encoder_args = []
                    return
                if self.performance_settings.get("test_encoder", False) and selected_encoder:
                    test_result = detector.test_encoder(selected_encoder)
                    if not test_result.get("available"):
                        self.log(f"  GPU encoder test failed: {test_result.get('reason', 'unknown')}")
                        self.log("  Falling back to CPU encoding")
                        self.gpu_enabled = False
                        self.gpu_encoder_args = []
                        return
                if hasattr(self, "download_service"):
                    self.download_service.performance_settings["prefer_gpu"] = self.gpu_enabled
                self.log(f"  ⚡ GPU Acceleration: ENABLED ({self.gpu_info.get('name', 'Unknown')})")
                self.log(f"  Selected encoder: {selected_encoder or 'libx264'}")
                self.log(f"  Encoder args: {' '.join(self.gpu_encoder_args)}")
                self.log(f"  Hardware decode: {'enabled' if self.gpu_decode_enabled else 'disabled'}")
            except Exception as e:
                self.log(f"  ⚠ GPU Acceleration failed to initialize: {e}")
                self.log(f"  Falling back to CPU encoding")
                self.gpu_enabled = False
                self.gpu_encoder_args = []
        else:
            self.log(f"  💻 GPU Acceleration: DISABLED (using CPU)")
            self.gpu_encoder_args = []

    @staticmethod
    def _extract_video_encoder(args: list) -> str:
        """Extract the selected FFmpeg video encoder from an arg list."""
        if not args:
            return ""
        for idx, token in enumerate(args[:-1]):
            if token == "-c:v":
                return args[idx + 1]
        return ""
    
    def get_video_encoder_args(self) -> list:
        """Get video encoder arguments based on GPU settings"""
        if self.gpu_enabled and self.gpu_encoder_args:
            return self.gpu_encoder_args
        else:
            # Default CPU encoding
            if getattr(self, "gpu_codec", "h264") == "hevc":
                return ['-c:v', 'libx265', '-preset', 'fast', '-crf', '22']
            return ['-c:v', 'libx264', '-preset', 'fast', '-crf', '18']
            
    def get_hwaccel_args(self) -> list:
        """Get hardware decode arguments if enabled"""
        if getattr(self, "gpu_enabled", False) and getattr(self, "gpu_decode_enabled", True):
            gpu_type = getattr(self, "gpu_info", {}).get("type")
            if gpu_type == "nvidia":
                return ["-hwaccel", "cuda"]
            elif gpu_type == "intel":
                return ["-hwaccel", "qsv"]
            elif gpu_type == "apple":
                return ["-hwaccel", "videotoolbox"]
            else:
                return ["-hwaccel", "auto"]
        return []

    # ------------------------------------------------------------------
    # GPU encoder safety net
    # ------------------------------------------------------------------
    # Some GPU encoders (h264_qsv, h264_nvenc, h264_amf) reject specific
    # preset/option combinations depending on the FFmpeg build, driver
    # version, or GPU model. When that happens, the FFmpeg call fails very
    # early with messages like:
    #   "Unable to parse "preset" option value ..."
    #   "Error setting option preset to value ..."
    #   "Error applying encoder options"
    # We detect these signatures, swap the GPU encoder args inside the
    # command for plain libx264 (CPU), and retry once. Subsequent calls in
    # the same session also fall back to CPU automatically.
    _CPU_FALLBACK_ARGS = ['-c:v', 'libx264', '-preset', 'fast', '-crf', '18']

    _GPU_ENCODER_NAMES = (
        'h264_nvenc', 'hevc_nvenc',
        'h264_qsv', 'hevc_qsv',
        'h264_amf', 'hevc_amf',
        'h264_videotoolbox', 'hevc_videotoolbox',
        'h264_mf', 'hevc_mf',
    )

    @classmethod
    def _is_gpu_encoder_error(cls, stderr: str) -> bool:
        """Heuristically detect FFmpeg failures caused by GPU encoder options."""
        if not stderr:
            return False
        text = stderr.lower()
        # Mention of any hardware encoder + a known option/init failure phrase
        mentions_hw = any(enc in text for enc in cls._GPU_ENCODER_NAMES)
        failure_phrases = (
            'error applying encoder options',
            'error setting option',
            'unable to parse',
            'no nvenc capable devices found',
            'cannot load nvcuda',
            'cannot load nvencodeapi',
            'failed loading nvenc',
            'device creation failed',
            'no device available',
            'impossible to convert between',
            'function not implemented',
        )
        mentions_failure = any(p in text for p in failure_phrases)
        return mentions_hw and mentions_failure

    @classmethod
    def _swap_cmd_to_cpu_encoder(cls, cmd: list) -> list:
        """Return a copy of cmd with any GPU encoder block replaced by CPU args.

        This walks the command, finds every ``-c:v <hw_encoder>`` and removes
        the encoder + any GPU-specific options that follow it (until the next
        FFmpeg flag or input/output token). It then injects the CPU fallback
        args in the same position. Audio codec args (``-c:a``) are preserved.
        """
        if not cmd:
            return cmd

        # Options that are known to belong to GPU encoders. We strip them
        # together with their value so libx264 doesn't choke on them.
        gpu_only_opts = {
            '-preset', '-rc', '-cq', '-qp', '-qp_i', '-qp_p', '-qp_b',
            '-quality', '-global_quality', '-look_ahead', '-rc_lookahead',
            '-spatial_aq', '-temporal_aq', '-aq-strength', '-tune',
            '-profile:v', '-level', '-b:v', '-maxrate', '-bufsize',
            '-pix_fmt',
        }

        new_cmd = []
        i = 0
        replaced = False
        while i < len(cmd):
            token = cmd[i]
            if token in ('-hwaccel', '-hwaccel_output_format', '-init_hw_device', '-filter_hw_device'):
                i += 2
                continue
            if token == '-c:v' and i + 1 < len(cmd) and cmd[i + 1] in cls._GPU_ENCODER_NAMES:
                # Inject CPU fallback once
                if not replaced:
                    new_cmd.extend(cls._CPU_FALLBACK_ARGS)
                    replaced = True
                # Skip '-c:v <hw_encoder>'
                i += 2
                # Skip any trailing GPU-specific options
                while i < len(cmd) - 1 and cmd[i] in gpu_only_opts:
                    i += 2
                continue
            new_cmd.append(token)
            i += 1

        # If no GPU encoder was present in cmd but caller still asked for
        # fallback, leave cmd untouched (nothing to swap).
        return new_cmd if replaced else list(cmd)

    def _disable_gpu_acceleration_runtime(self, reason: str = ""):
        """Disable GPU encoding for the rest of this processing session."""
        if not self.gpu_enabled:
            return
        self.gpu_enabled = False
        self.gpu_encoder_args = []
        msg = "  ⚠ GPU encoding disabled for the rest of this session"
        if reason:
            msg += f" ({reason})"
        self.log(msg)
        self.log("  💻 Continuing with CPU encoding (libx264)")

    def _run_ffmpeg_subprocess(self, cmd: list, **kwargs):
        """Run an FFmpeg command with automatic CPU fallback on GPU encoder errors.

        Wraps ``subprocess.run`` and, if the command fails with a signature
        that looks like a GPU encoder problem, rewrites the command to use
        libx264 and retries once. Returns the final ``CompletedProcess``.
        """
        kwargs.setdefault('capture_output', True)
        kwargs.setdefault('text', True)
        kwargs.setdefault('creationflags', SUBPROCESS_FLAGS)

        result = subprocess.run(cmd, **kwargs)
        if result.returncode == 0:
            return result

        stderr = result.stderr or ''
        if not self._is_gpu_encoder_error(stderr):
            return result

        # Looks like a GPU encoder issue: swap to CPU and retry once.
        fallback_cmd = self._swap_cmd_to_cpu_encoder(cmd)
        if fallback_cmd == list(cmd):
            # No GPU encoder found in cmd to swap; return original failure.
            return result

        self.log("  ⚠ FFmpeg failed with GPU encoder error, retrying on CPU...")
        # Pull a short reason line from stderr for the log
        reason_line = next(
            (ln.strip() for ln in stderr.splitlines()
             if 'error' in ln.lower() or 'unable' in ln.lower()),
            ''
        )
        self._disable_gpu_acceleration_runtime(reason_line[:120])

        retry = subprocess.run(fallback_cmd, **kwargs)
        return retry

    def log_ffmpeg_command(self, cmd: list, description: str = "FFmpeg"):
        """Log FFmpeg command for debugging"""
        # Format command nicely
        cmd_str = ' '.join(f'"{arg}"' if ' ' in str(arg) else str(arg) for arg in cmd)
        self.log(f"  🎬 {description} Command:")
        self.log(f"     {cmd_str}")
    
    def _find_system_font_bold(self) -> str:
        """Find a bold system font across platforms"""
        if sys.platform == "win32":
            candidates = [
                "C:/Windows/Fonts/arialbd.ttf",
                "C:/Windows/Fonts/arial.ttf",
                "C:/Windows/Fonts/segoeui.ttf",
            ]
        elif sys.platform == "darwin":
            candidates = [
                "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
                "/System/Library/Fonts/Supplemental/Arial.ttf",
                "/Library/Fonts/Arial Bold.ttf",
                "/Library/Fonts/Arial.ttf",
                "/System/Library/Fonts/Helvetica.ttc",
                "/System/Library/Fonts/SFNS.ttf",
            ]
        else:
            candidates = [
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
                "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
            ]
        
        for font in candidates:
            if os.path.exists(font):
                return font
        return None
    
    def _get_ffmpeg_font_path(self) -> str:
        """Get fontfile argument for FFmpeg drawtext filter, platform-aware"""
        font = self._find_system_font_bold()
        if font:
            if sys.platform == "win32":
                # Escape colon for FFmpeg filter on Windows
                escaped = font.replace("\\", "/").replace(":", "\\:")
                return f"fontfile='{escaped}':"
            else:
                return f"fontfile='{font}':"
        # Fallback: let FFmpeg use fontconfig default
        return "font='Arial':"
    
    @staticmethod
    def get_default_prompt():
        """Get default system prompt for highlight detection"""
        from config.ai_provider_config import DEFAULT_HIGHLIGHT_PROMPT
        return DEFAULT_HIGHLIGHT_PROMPT
    
    def process(self, url: str, num_clips: int = 5, add_captions: bool = True, add_hook: bool = True):
        """Main processing pipeline"""
        
        # Step 1: Download video
        self.set_progress("Downloading video...", 0.1)
        video_path, srt_path, video_info = self.download_service.download_video(url)
        
        # Store channel name for credit watermark
        self.channel_name = video_info.get("channel", "") if video_info else ""
        
        if self.is_cancelled():
            return
        
        if not srt_path:
            raise SubtitleNotFoundError(
                f"No subtitle available for language: {self.subtitle_language.upper()}",
                video_path=video_path,
                video_info=video_info
            )
        
        # Step 2: Find highlights
        self.set_progress("Finding highlights...", 0.3)
        transcript = self.download_service.parse_srt(srt_path)
        highlights = self.find_highlights(transcript, video_info, num_clips)
        
        if self.is_cancelled():
            return
        
        if not highlights:
            raise Exception("No valid highlights found!")
        
        # Step 3: Process each clip
        total_clips = len(highlights)
        for i, highlight in enumerate(highlights, 1):
            if self.is_cancelled():
                return
            self.process_clip(video_path, highlight, i, total_clips, add_captions=add_captions, add_hook=add_hook)
        
        # Cleanup
        self.set_progress("Cleaning up...", 0.95)
        self.cleanup()
        
        self.set_progress("Complete!", 1.0)
        self.log(f"\n✅ Created {total_clips} clips in: {self.output_dir}")
    
    def transcribe_full_video(self, video_path: str) -> str:
        """Transcribe full video audio using Whisper API (Caption Maker).
        
        Extracts audio from the video, compresses to mp3, splits into chunks
        if needed (Whisper API has ~25MB limit), and returns a transcript
        formatted like parse_srt output so find_highlights can consume it directly.
        
        Returns:
            str: Transcript with timestamps in SRT-like format:
                 [HH:MM:SS,mmm - HH:MM:SS,mmm] text
        """
        self.log("[AI Transcription] Transcribing full video with Whisper API...")
        
        # Check Caption Maker is configured
        cm_config = self.ai_providers.get("caption_maker", {})
        if not cm_config.get("api_key"):
            raise Exception(
                "Caption Maker is not configured!\n\n"
                "Please set up Caption Maker in:\n"
                "Settings → AI API Settings → Caption Maker"
            )
        
        # Extract audio as compressed mp3 to minimize file size
        audio_file = tempfile.NamedTemporaryFile(suffix='.mp3', delete=False).name
        cmd = [
            self.ffmpeg_path, "-y",
            *self.get_hwaccel_args(),
            "-i", video_path,
            "-vn",
            "-acodec", "libmp3lame",
            "-ar", "16000",
            "-ac", "1",
            "-b:a", "64k",
            audio_file
        ]
        self.log("  Extracting audio from video...")
        result = subprocess.run(cmd, capture_output=True, text=True, creationflags=SUBPROCESS_FLAGS)
        
        if result.returncode != 0:
            if os.path.exists(audio_file):
                os.unlink(audio_file)
            raise Exception(f"Failed to extract audio from video:\n{result.stderr[:200]}")
        
        file_size_mb = os.path.getsize(audio_file) / (1024 * 1024)
        self.log(f"  Audio file size: {file_size_mb:.1f} MB")
        
        # Get total audio duration
        probe_cmd = [self.ffmpeg_path, "-i", audio_file, "-f", "null", "-"]
        probe_result = subprocess.run(probe_cmd, capture_output=True, text=True, creationflags=SUBPROCESS_FLAGS)
        duration_match = re.search(r"Duration: (\d+):(\d+):(\d+\.\d+)", probe_result.stderr)
        total_duration = 0
        if duration_match:
            h, m, s = duration_match.groups()
            total_duration = int(h) * 3600 + int(m) * 60 + float(s)
        
        self.log(f"  Audio duration: {total_duration:.0f}s ({total_duration/60:.1f} min)")
        
        # Report Whisper usage
        self.report_tokens(0, 0, total_duration, 0)
        
        # Split into chunks if file is too large (>4MB to avoid proxy timeout)
        MAX_CHUNK_SIZE_MB = 4
        all_segments = []
        
        if file_size_mb <= MAX_CHUNK_SIZE_MB:
            # Single file, transcribe directly
            self.log("  Sending to Whisper API...")
            self.set_progress("Transcribing audio with AI...", 0.3)
            segments = self._whisper_transcribe_file(audio_file, 0)
            all_segments.extend(segments)
        else:
            # Split into chunks by duration
            chunk_count = int(file_size_mb / MAX_CHUNK_SIZE_MB) + 1
            chunk_duration = total_duration / chunk_count
            self.log(f"  File too large, splitting into {chunk_count} chunks (~{chunk_duration:.0f}s each)...")
            
            for i in range(chunk_count):
                if self.is_cancelled():
                    os.unlink(audio_file)
                    return ""
                
                chunk_start = i * chunk_duration
                chunk_file = tempfile.NamedTemporaryFile(suffix='.mp3', delete=False).name
                
                cmd = [
                    self.ffmpeg_path, "-y",
                    *self.get_hwaccel_args(),
                    "-i", audio_file,
                    "-ss", str(chunk_start),
                    "-t", str(chunk_duration),
                    "-acodec", "libmp3lame",
                    "-ar", "16000",
                    "-ac", "1",
                    "-b:a", "64k",
                    chunk_file
                ]
                subprocess.run(cmd, capture_output=True, text=True, creationflags=SUBPROCESS_FLAGS)
                
                chunk_size = os.path.getsize(chunk_file) / (1024 * 1024)
                self.log(f"  Transcribing chunk {i+1}/{chunk_count} ({chunk_size:.1f}MB, ~{chunk_duration:.0f}s)...")
                self.set_progress(f"Transcribing audio chunk {i+1}/{chunk_count}...", 
                                  0.3 + (0.2 * (i + 1) / chunk_count))
                
                segments = self._whisper_transcribe_file(chunk_file, chunk_start)
                all_segments.extend(segments)
                
                try:
                    os.unlink(chunk_file)
                except Exception:
                    pass
        
        # Cleanup main audio file
        try:
            os.unlink(audio_file)
        except Exception:
            pass
        
        if not all_segments:
            raise Exception("Whisper API returned empty transcription. The video may have no speech.")
        
        # Format segments into SRT-like transcript (same format as parse_srt output)
        lines = []
        for seg in all_segments:
            start_ts = self._seconds_to_srt_timestamp(seg["start"])
            end_ts = self._seconds_to_srt_timestamp(seg["end"])
            text = seg["text"].strip()
            if text:
                lines.append(f"[{start_ts} - {end_ts}] {text}")
        
        transcript = "\n".join(lines)
        self.log(f"  ✓ Transcription complete: {len(lines)} segments")
        
        return transcript
    
    def _whisper_transcribe_file(self, audio_path: str, time_offset: float = 0) -> list:
        """Transcribe a single audio file with Whisper API.
        
        Uses raw httpx POST instead of OpenAI SDK for better proxy compatibility.
        
        Args:
            audio_path: Path to audio file
            time_offset: Offset in seconds to add to all timestamps (for chunked files)
        
        Returns:
            list of dicts with 'start', 'end', 'text' keys
        """
        import time as _time
        import requests as _requests
        
        file_size_mb = os.path.getsize(audio_path) / (1024 * 1024)
        base_url = str(self.caption_client.base_url).rstrip("/")
        api_key = self.caption_client.api_key
        
        self.log(f"    Uploading {file_size_mb:.1f}MB to Whisper API ({self.whisper_model})...")
        self.log(f"    Base URL: {base_url}")
        
        # Build multipart form data
        url = f"{base_url}/audio/transcriptions"
        headers = {"Authorization": f"Bearer {api_key}"}
        
        form_data = {
            "model": self.whisper_model,
            "response_format": "verbose_json",
        }
        if self.subtitle_language and self.subtitle_language != "none":
            form_data["language"] = self.subtitle_language
        
        # Run API call in a thread so we can log heartbeat while waiting
        response_data = None
        api_error = None
        
        def _call_api():
            nonlocal response_data, api_error
            try:
                with open(audio_path, "rb") as f:
                    files = {"file": (os.path.basename(audio_path), f, "audio/mpeg")}
                    resp = _requests.post(url, headers=headers, data=form_data, files=files, timeout=600)
                    resp.raise_for_status()
                    response_data = resp.json()
            except Exception as e:
                api_error = e
        
        api_thread = threading.Thread(target=_call_api, daemon=True)
        start_time = _time.time()
        api_thread.start()
        
        # Heartbeat: log every 15s so user knows it's still working
        TIMEOUT_SECONDS = 300  # 5 minutes max per chunk
        while api_thread.is_alive():
            api_thread.join(timeout=15)
            if api_thread.is_alive():
                elapsed = _time.time() - start_time
                
                # Check cancellation
                if self.is_cancelled():
                    self.log(f"    ⚠️ Cancelled by user during Whisper API call")
                    return []
                
                if elapsed > TIMEOUT_SECONDS:
                    self.log(f"    ⏱️ Whisper API timed out after {TIMEOUT_SECONDS}s")
                    raise Exception(
                        f"Whisper API timed out after {TIMEOUT_SECONDS}s.\n\n"
                        "Possible causes:\n"
                        "1. Your AI API provider may not support the Whisper audio endpoint\n"
                        "2. The server may be overloaded or unreachable\n"
                        "3. Network connection issue\n\n"
                        "Try:\n"
                        "- Check if your Caption Maker API supports audio transcription\n"
                        "- Try again later\n"
                        "- Use a different API provider for Caption Maker"
                    )
                self.log(f"    ⏳ Waiting for Whisper API response... ({elapsed:.0f}s elapsed)")
                self.set_progress(f"Transcribing with AI... waiting for response ({elapsed:.0f}s)", 0.35)
        
        elapsed = _time.time() - start_time
        
        if api_error:
            self.log(f"  ❌ Whisper API error after {elapsed:.1f}s: {api_error}")
            raise Exception(f"Whisper transcription failed:\n{str(api_error)}")
        
        if response_data is None:
            self.log(f"  ❌ Whisper API returned no response after {elapsed:.1f}s")
            raise Exception("Whisper API returned no response. The endpoint may not support audio transcription.")
        
        self.log(f"    ✓ Whisper API responded in {elapsed:.1f}s")
        
        segments = []
        if "segments" in response_data and response_data["segments"]:
            for seg in response_data["segments"]:
                segments.append({
                    "start": seg.get("start", 0) + time_offset,
                    "end": seg.get("end", 0) + time_offset,
                    "text": seg.get("text", "")
                })
        
        return segments
    
    def _whisper_transcribe_words_api(self, audio_path: str):
        """Transcribe an audio file with word-level timestamps using raw HTTP.

        Compresses the audio to MP3 before uploading (the ytclip proxy drops
        connections for large WAV files >~1MB). Uses ``requests`` instead of
        the OpenAI SDK for proxy compatibility. Tries with
        ``timestamp_granularities[]=word`` first; if the proxy rejects it
        (400), retries without that field (still gets segments).

        Returns an object exposing ``.words`` and ``.segments`` (mirroring the
        SDK response shape consumed by ``create_ass_subtitle_capcut``), or
        raises on failure.
        """
        import requests as _requests
        from types import SimpleNamespace

        base_url = str(self.caption_client.base_url).rstrip("/")
        api_key = self.caption_client.api_key
        url = f"{base_url}/audio/transcriptions"
        headers = {"Authorization": f"Bearer {api_key}"}

        lang = getattr(self, "subtitle_language", None) or "id"

        # Compress WAV → MP3 to reduce upload size (proxy rejects large bodies)
        upload_path = audio_path
        mp3_tmp = None
        if audio_path.lower().endswith(".wav"):
            mp3_tmp = audio_path.rsplit(".", 1)[0] + "_upload.mp3"
            cmd = [
                self.ffmpeg_path, "-y",
                *self.get_hwaccel_args(),
                "-i", audio_path,
                "-acodec", "libmp3lame",
                "-b:a", "64k",
                "-ar", "16000",
                "-ac", "1",
                mp3_tmp
            ]
            result = subprocess.run(cmd, capture_output=True, text=True,
                                    creationflags=SUBPROCESS_FLAGS)
            if result.returncode == 0 and os.path.exists(mp3_tmp):
                upload_path = mp3_tmp
                self.log(f"  [Caption] Compressed WAV→MP3: "
                         f"{os.path.getsize(audio_path)/1024:.0f}KB → "
                         f"{os.path.getsize(mp3_tmp)/1024:.0f}KB")
            else:
                self.log("  [Caption] MP3 compression failed, uploading WAV as-is")
                mp3_tmp = None

        file_size_mb = os.path.getsize(upload_path) / (1024 * 1024)
        mime = "audio/mpeg" if upload_path.endswith(".mp3") else "audio/wav"
        self.log(f"  [Caption] Uploading {file_size_mb:.2f}MB to Whisper ({self.whisper_model})...")

        # Attempt 1: with word-level granularity
        form_data = [
            ("model", self.whisper_model),
            ("response_format", "verbose_json"),
            ("timestamp_granularities[]", "word"),
            ("timestamp_granularities[]", "segment"),
        ]
        if lang and lang != "none":
            form_data.append(("language", lang))

        resp = None
        for attempt in range(2):
            with open(upload_path, "rb") as f:
                files = {"file": (os.path.basename(upload_path), f, mime)}
                resp = _requests.post(url, headers=headers, data=form_data,
                                      files=files, timeout=600)

            if resp.status_code == 200:
                break

            # Log the actual error body for debugging
            self.log(f"  [Caption] Attempt {attempt+1} failed: HTTP {resp.status_code}")
            try:
                self.log(f"  [Caption] Response: {resp.text[:300]}")
            except Exception:
                pass

            if attempt == 0:
                # Retry without timestamp_granularities (proxy may not support it)
                self.log("  [Caption] Retrying without timestamp_granularities...")
                form_data = [
                    ("model", self.whisper_model),
                    ("response_format", "verbose_json"),
                ]
                if lang and lang != "none":
                    form_data.append(("language", lang))
            else:
                # Both attempts failed — clean up and raise
                if mp3_tmp and os.path.exists(mp3_tmp):
                    os.unlink(mp3_tmp)
                raise Exception(
                    f"Whisper API returned HTTP {resp.status_code}: "
                    f"{resp.text[:300]}"
                )

        # Clean up temp mp3
        if mp3_tmp and os.path.exists(mp3_tmp):
            os.unlink(mp3_tmp)

        data = resp.json()
        self.log(f"  [Caption] Whisper OK, text length: {len(data.get('text', ''))}")

        words = [
            SimpleNamespace(
                word=w.get("word", ""),
                start=w.get("start", 0.0),
                end=w.get("end", 0.0),
            )
            for w in (data.get("words") or [])
        ]
        segments = data.get("segments") or []
        self.log(f"  [Caption] Got {len(words)} words, {len(segments)} segments")
        return SimpleNamespace(words=words, segments=segments,
                               text=data.get("text", ""))
    
    @staticmethod
    def _seconds_to_srt_timestamp(seconds: float) -> str:
        """Convert seconds to SRT timestamp format HH:MM:SS,mmm"""
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = seconds % 60
        ms = int((s - int(s)) * 1000)
        return f"{h:02d}:{m:02d}:{int(s):02d},{ms:03d}"
    
    def find_highlights_with_transcription(self, video_path: str, video_info: dict, 
                                            num_clips: int, session_dir: str = None) -> dict:
        """Find highlights by first transcribing the video with Whisper API.
        
        This is the fallback path when no subtitle is available.
        Uses Caption Maker (Whisper) to generate transcript, then feeds it
        to Highlight Finder (GPT) as usual.
        
        Returns:
            dict: Same session_data format as find_highlights_only
        """
        from datetime import datetime
        
        # Use existing session_dir or create new one
        if session_dir:
            session_dir = Path(session_dir)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            session_dir = self.output_dir / "sessions" / timestamp
            session_dir.mkdir(parents=True, exist_ok=True)
        
        # Update temp_dir to session-specific temp
        self.temp_dir = session_dir / "_temp"
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Step 1: Transcribe with Whisper
        self.set_progress("Transcribing video with AI...", 0.3)
        transcript = self.transcribe_full_video(video_path)
        
        if self.is_cancelled():
            return None
        
        # Step 2: Find highlights using the transcript
        self.set_progress("Finding highlights with AI...", 0.6)
        highlights = self.find_highlights(transcript, video_info, num_clips)
        
        if self.is_cancelled():
            return None
        
        if not highlights:
            raise Exception(
                "No valid highlights found!\n\n"
                "Possible causes:\n"
                "1. AI model failed to generate highlights\n"
                "2. Video transcript too short or not suitable\n"
                "3. AI model configuration issue\n\n"
                "Try:\n"
                "- Using a different AI model\n"
                "- Checking AI API settings\n"
                "- Using a longer video with more content"
            )
        
        self.set_progress("Highlights found!", 1.0)
        self.log(f"\n✅ Found {len(highlights)} highlights (via AI transcription)")
        
        # Save session data
        session_data_file = session_dir / "session_data.json"
        session_data = {
            "session_dir": str(session_dir),
            "video_path": video_path,
            "srt_path": None,
            "highlights": highlights,
            "video_info": video_info,
            "created_at": datetime.now().isoformat(),
            "status": "highlights_found",
            "transcription_method": "whisper_api"
        }
        
        with open(session_data_file, "w", encoding="utf-8") as f:
            json.dump(session_data, f, indent=2, ensure_ascii=False)
        
        self.log(f"Session data saved to: {session_data_file}")
        
        return session_data
    
    def find_highlights(self, transcript: str, video_info: dict, num_clips: int) -> list:
        """Find highlights using AI (OpenAI-compatible API)"""
        self.log(f"[2/4] Finding highlights (using {self.model})...")
        
        request_clips = num_clips + 3
        
        video_context = ""
        if video_info:
            video_context = f"""INFO VIDEO:
- Judul: {video_info.get('title', 'Unknown')}
- Channel: {video_info.get('channel', 'Unknown')}
- Deskripsi: {video_info.get('description', '')[:500]}"""
        
        # Replace placeholders safely (avoid .format() which breaks on user's curly braces)
        prompt = self.system_prompt.replace("{num_clips}", str(request_clips))
        prompt = prompt.replace("{video_context}", video_context)
        prompt = prompt.replace("{transcript}", transcript)
        
        # Warn if required placeholders are missing
        if "{transcript}" in self.system_prompt and "{transcript}" in prompt:
            self.log("  ⚠ Warning: {transcript} placeholder not replaced - check your system prompt")
        if "{num_clips}" in self.system_prompt and "{num_clips}" in prompt:
            self.log("  ⚠ Warning: {num_clips} placeholder not replaced - check your system prompt")

        # Use OpenAI-compatible API for all providers
        self.log(f"  Using API: {self.highlight_client.base_url}")
        try:
            response = self.highlight_client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
            )
            
            # Validate response structure
            if not response:
                raise Exception("API returned empty response")
            
            if not hasattr(response, 'choices') or not response.choices:
                # Log response structure for debugging
                self.log(f"  ⚠ Unexpected API response structure: {type(response)}")
                self.log(f"  Response attributes: {dir(response)}")
                raise Exception(
                    "API response missing 'choices' field.\n\n"
                    "This usually happens with custom API providers that don't follow OpenAI format.\n\n"
                    "Please check:\n"
                    "1. API key is valid and has credits\n"
                    "2. Base URL is correct for your provider\n"
                    "3. Model name is supported by your provider\n"
                    "4. Provider follows OpenAI-compatible API format"
                )
            
            if not response.choices[0].message or not response.choices[0].message.content:
                raise Exception(
                    "API returned empty content.\n\n"
                    "Possible causes:\n"
                    "1. Model refused to generate content (content filter)\n"
                    "2. API quota exceeded\n"
                    "3. Model doesn't support this type of request"
                )
            
            # Report token usage (input and output separately)
            if hasattr(response, 'usage') and response.usage:
                self.report_tokens(response.usage.prompt_tokens, response.usage.completion_tokens, 0, 0)
            
            result = response.choices[0].message.content.strip()
            
        except Exception as e:
            # Check if it's our custom exception
            if "API response missing" in str(e) or "API returned empty" in str(e):
                raise
            
            # Otherwise, wrap with more context
            self.log(f"  ❌ API Error: {e}")
            raise Exception(
                f"Failed to get highlights from AI model.\n\n"
                f"Error: {str(e)}\n\n"
                f"Please check:\n"
                f"1. API key is valid: {self.highlight_client.api_key[:20]}...\n"
                f"2. Base URL is correct: {self.highlight_client.base_url}\n"
                f"3. Model exists: {self.model}\n"
                f"4. You have sufficient credits/quota"
            )
        
        # Log raw response for debugging
        self.log(f"  Raw AI response (first 500 chars):\n{result[:500]}")
        
        if result.startswith("```"):
            result = re.sub(r"```json?\n?", "", result)
            result = re.sub(r"```\n?", "", result)
        
        try:
            highlights = json.loads(result)
        except json.JSONDecodeError as e:
            # Log full response on error
            self.log(f"\n❌ JSON Parse Error: {e}")
            self.log(f"\n📄 Full GPT Response:\n{result}")
            self.log(f"\n💡 Error position: line {e.lineno}, column {e.colno}")
            raise Exception(f"Failed to parse GPT response as JSON: {e}\n\nFull response logged above.")
        
        # Filter by duration (min 58s, max 120s)
        valid = []
        for h in highlights:
            # Fallback: convert "reason" to "description" if exists
            if "reason" in h and "description" not in h:
                h["description"] = h.pop("reason")
                self.log(f"  ⚠ Converted 'reason' to 'description' for '{h.get('title', 'Unknown')}'")
            
            duration = self.parse_timestamp(h["end_time"]) - self.parse_timestamp(h["start_time"])
            h["duration_seconds"] = round(duration, 1)
            
            # Ensure virality_score exists (default to 5 if missing)
            if "virality_score" not in h:
                h["virality_score"] = 5
                self.log(f"  ⚠ Missing virality_score for '{h.get('title', 'Unknown')}', defaulting to 5")
            
            # Ensure description exists
            if "description" not in h:
                h["description"] = h.get("title", "No description")
                self.log(f"  ⚠ Missing description for '{h.get('title', 'Unknown')}', using title")
            
            if 58 <= duration <= 120:
                valid.append(h)
                virality = h.get("virality_score", 5)
                self.log(f"  ✓ {h['title']} ({duration:.0f}s) [🔥 {virality}/10]")
            elif duration > 120:
                self.log(f"  ✗ {h['title']} ({duration:.0f}s) - Too long, skipped")
            elif duration < 58:
                self.log(f"  ✗ {h['title']} ({duration:.0f}s) - Too short, skipped")
            
            if len(valid) >= num_clips:
                break
        
        # If we don't have enough valid clips, warn user
        if len(valid) < num_clips:
            self.log(f"\n⚠️ WARNING: Only found {len(valid)} valid clips out of {num_clips} requested!")
            self.log(f"   AI returned many segments that were too short (< 58s).")
            self.log(f"   Consider using a better AI model or adjusting the prompt.")
        
        return valid[:num_clips]

    def _build_landscape_clip_command(self, video_path: str, start: str, end: str, landscape_file: Path, pre_cut: bool):
        """Build the FFmpeg command for the first clip-processing step."""
        encoder_args = self.get_video_encoder_args()
        cmd = [
            self.ffmpeg_path, "-y",
            *self.get_hwaccel_args(),
            "-i", video_path,
        ]

        if not pre_cut:
            cmd.extend(["-ss", start, "-to", end])

        cmd.extend([
            *encoder_args,
            "-c:a", "aac", "-b:a", "192k",
            "-progress", "pipe:1",
            str(landscape_file),
        ])
        return cmd

    def _run_landscape_clip_step(
        self,
        video_path: str,
        start: str,
        end: str,
        landscape_file: Path,
        duration: float,
        pre_cut: bool,
        current_step: int,
        clip_progress,
    ):
        """Cut or re-encode the landscape input while preserving existing logs."""
        if pre_cut:
            step_name = "Re-encoding video..."
            command_name = "Re-encode Pre-cut Section"
            done_message = "  ✓ Re-encoded pre-cut section"
        else:
            step_name = "Cutting video..."
            command_name = "Cut Video"
            done_message = "  ✓ Cut video"

        clip_progress(step_name, current_step, 0)
        cmd = self._build_landscape_clip_command(video_path, start, end, landscape_file, pre_cut)
        self.log_ffmpeg_command(cmd, command_name)
        self.run_ffmpeg_with_progress(
            cmd,
            duration,
            lambda p: clip_progress(step_name, current_step, p),
        )
        self.log(done_message)

    def _run_portrait_clip_step(self, landscape_file: Path, portrait_file: Path, current_step: int, clip_progress):
        clip_progress("Converting to portrait...", current_step, 0)
        self.convert_to_portrait_with_progress(
            str(landscape_file),
            str(portrait_file),
            lambda p: clip_progress("Converting to portrait...", current_step, p),
        )
        self.log("  ✓ Portrait conversion")

    def _run_hook_clip_step(self, current_output: Path, clip_dir: Path, highlight: dict, current_step: int, clip_progress):
        clip_progress("Adding hook...", current_step, 0)
        hooked_file = clip_dir / "temp_hooked.mp4"
        hook_text = highlight.get("hook_text", highlight["title"])
        self.log("  Generating and adding hook...")
        hook_duration = self.add_hook(
            str(current_output),
            hook_text,
            str(hooked_file),
            lambda p: clip_progress("Adding hook...", current_step, p),
        )

        if not hooked_file.exists():
            raise Exception(f"Failed to create hooked video: {hooked_file}")

        self.log(f"  ✓ Added hook ({hook_duration:.1f}s)")
        return hooked_file, hook_duration

    def _run_caption_clip_step(
        self,
        current_output: Path,
        portrait_file: Path,
        final_file: Path,
        clip_dir: Path,
        add_hook: bool,
        hook_duration: float,
        current_step: int,
        clip_progress,
    ):
        clip_progress("Adding captions...", current_step, 0)
        audio_source = str(portrait_file) if add_hook else None

        if self.watermark_settings.get("enabled"):
            temp_captioned = clip_dir / "temp_captioned.mp4"
            output_path = temp_captioned
        else:
            output_path = final_file

        self.add_captions_api_with_progress(
            str(current_output),
            str(output_path),
            audio_source,
            hook_duration,
            lambda p: clip_progress("Adding captions...", current_step, p),
        )

        if not output_path.exists():
            if output_path == final_file:
                raise Exception(f"Failed to create final video: {final_file}")
            raise Exception(f"Failed to create captioned video: {output_path}")

        self.log("  ✓ Added captions")
        return output_path

    def _run_watermark_clip_step(self, current_output: Path, final_file: Path, current_step: int, clip_progress):
        clip_progress("Adding watermark...", current_step, 0)
        self.add_watermark_with_progress(
            str(current_output),
            str(final_file),
            lambda p: clip_progress("Adding watermark...", current_step, p),
        )

        if not final_file.exists():
            raise Exception(f"Failed to create final video with watermark: {final_file}")

        self.log("  ✓ Added watermark")
        return final_file

    def _ensure_final_without_captions_or_watermark(self, current_output: Path, final_file: Path):
        import shutil
        shutil.copy(str(current_output), str(final_file))
        return final_file

    def _run_credit_clip_step(self, current_output: Path, final_file: Path, clip_dir: Path, current_step: int, clip_progress):
        import shutil

        clip_progress("Adding credit...", current_step, 0)

        if str(current_output) == str(final_file):
            temp_credit_input = clip_dir / "temp_before_credit.mp4"
            shutil.copy(str(final_file), str(temp_credit_input))
            current_output = temp_credit_input

        self.add_credit_watermark_with_progress(
            str(current_output),
            str(final_file),
            lambda p: clip_progress("Adding credit...", current_step, p),
        )

        if not final_file.exists():
            raise Exception(f"Failed to create final video with credit: {final_file}")

        self.log(f"  ✓ Added credit: Source: {self.channel_name}")
        self._delete_clip_temp_file(clip_dir / "temp_before_credit.mp4", "temp_before_credit.mp4")
        return final_file

    def process_clip(self, video_path: str, highlight: dict, index: int, total_clips: int = 1, add_captions: bool = True, add_hook: bool = True, pre_cut: bool = False):
        """Process a single clip: cut, portrait, hook (optional), captions (optional)
        
        Args:
            video_path: Path to source video (full video or pre-cut section)
            highlight: Highlight dict with metadata
            index: Clip index (1-based)
            total_clips: Total number of clips being processed
            add_captions: Whether to add captions
            add_hook: Whether to add hook
            pre_cut: If True, video_path is already a pre-cut section (skip cutting step)
        """
        
        # Check cancel before starting
        if self.is_cancelled():
            return
        
        clip_dir = self._create_clip_output_dir(index)
        
        self.log(f"  Output folder: {clip_dir}")
        
        start = highlight["start_time"].replace(",", ".")
        end = highlight["end_time"].replace(",", ".")
        
        self.log(f"\n[Clip {index}] {highlight['title']}")
        
        total_steps = self._get_clip_total_steps(add_captions, add_hook)
        
        # Helper to report sub-progress with percentage
        def clip_progress(step_name: str, step_num: int, sub_progress: float = 0):
            self._report_clip_progress(index, total_clips, total_steps, step_name, step_num, sub_progress)
        
        current_step = 0

        # Step 1: Cut video (skip if pre-cut section from --download-sections)
        if self.is_cancelled():
            return
        
        landscape_file = clip_dir / "temp_landscape.mp4"
        duration = self.parse_timestamp(end) - self.parse_timestamp(start)

        self._run_landscape_clip_step(
            video_path,
            start,
            end,
            landscape_file,
            duration,
            pre_cut,
            current_step,
            clip_progress,
        )
        
        current_step += 1
        
        # Step 2: Convert to portrait with progress
        if self.is_cancelled():
            return
        portrait_file = clip_dir / "temp_portrait.mp4"
        self._run_portrait_clip_step(landscape_file, portrait_file, current_step, clip_progress)
        current_step += 1
        
        # Track which file is the current output
        current_output = portrait_file
        hook_duration = 0
        
        # Step 3: Add hook (optional)
        if add_hook:
            if self.is_cancelled():
                return
            current_output, hook_duration = self._run_hook_clip_step(
                current_output,
                clip_dir,
                highlight,
                current_step,
                clip_progress,
            )
            current_step += 1
        else:
            self.log("  ⊘ Skipped hook (disabled)")
        
        # Step 4: Add captions (optional)
        final_file = clip_dir / "master.mp4"
        if add_captions:
            if self.is_cancelled():
                return
            current_output = self._run_caption_clip_step(
                current_output,
                portrait_file,
                final_file,
                clip_dir,
                add_hook,
                hook_duration,
                current_step,
                clip_progress,
            )
            current_step += 1
        else:
            self.log("  ⊘ Skipped captions (disabled)")
        
        # Step 5: Add watermark (if enabled)
        if self.watermark_settings.get("enabled"):
            if self.is_cancelled():
                return
            
            # Check if we need to add watermark step to progress
            if not add_captions:
                # Watermark is a new step
                total_steps += 1
            
            current_output = self._run_watermark_clip_step(current_output, final_file, current_step, clip_progress)
            current_step += 1
            
            # Cleanup temp captioned file if exists
            if add_captions:
                self._delete_clip_temp_file(clip_dir / "temp_captioned.mp4", "temp_captioned.mp4")
        elif not add_captions:
            # No captions and no watermark, just copy current output to final
            current_output = self._ensure_final_without_captions_or_watermark(current_output, final_file)
        
        # Step 6: Add credit watermark (if enabled)
        if self.credit_watermark_settings.get("enabled") and self.channel_name:
            if self.is_cancelled():
                return
            
            total_steps += 1
            current_output = self._run_credit_clip_step(
                current_output,
                final_file,
                clip_dir,
                current_step,
                clip_progress,
            )
            current_step += 1
        
        # Mark complete
        clip_progress("Done", total_steps, 0)
        
        self._cleanup_clip_temp_files(clip_dir, landscape_file, portrait_file, add_hook)
        self._write_clip_metadata(clip_dir, highlight, add_hook, add_captions)
    
    def convert_to_portrait(self, input_path: str, output_path: str):
        """Convert landscape to 9:16 portrait with speaker tracking (router method)"""
        # Route to the progress version with a dummy callback to reuse logic
        return self.convert_to_portrait_with_progress(input_path, output_path, lambda p: None)
    
    def convert_to_portrait_opencv(self, input_path: str, output_path: str):
        """Convert landscape to 9:16 portrait with speaker tracking (OpenCV Haar Cascade)"""
        
        cap = cv2.VideoCapture(input_path)
        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate crop dimensions
        target_ratio = 9 / 16
        crop_w = int(orig_h * target_ratio)
        crop_h = orig_h
        out_w, out_h = 1080, 1920
        
        # Face detector
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # First pass: analyze frames
        crop_positions = []
        current_target = orig_w / 2
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(50, 50))
            
            if len(faces) > 0:
                # Find largest face
                largest = max(faces, key=lambda f: f[2] * f[3])
                current_target = largest[0] + largest[2] / 2
            
            crop_x = int(current_target - crop_w / 2)
            crop_x = max(0, min(crop_x, orig_w - crop_w))
            crop_positions.append(crop_x)
        
        # Stabilize positions
        crop_positions = self.stabilize_positions(crop_positions)
        
        # Second pass: create video
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        temp_video = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False).name
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_video, fourcc, fps, (out_w, out_h))
        
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            crop_x = crop_positions[frame_idx] if frame_idx < len(crop_positions) else crop_positions[-1]
            cropped = frame[0:crop_h, crop_x:crop_x+crop_w]
            resized = cv2.resize(cropped, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4)
            out.write(resized)
            frame_idx += 1
        
        cap.release()
        out.release()
        
        # Merge with audio using GPU/CPU encoder
        encoder_args = self.get_video_encoder_args()
        cmd = [
            self.ffmpeg_path, "-y",
            *self.get_hwaccel_args(),
            "-i", temp_video,
            "-i", input_path,
            *encoder_args,
            "-c:a", "aac", "-b:a", "192k",
            "-map", "0:v:0", "-map", "1:a:0",
            "-shortest",
            output_path
        ]
        self.log_ffmpeg_command(cmd, "Portrait Merge Audio (OpenCV)")
        self._run_ffmpeg_subprocess(cmd)
        os.unlink(temp_video)
    
    def stabilize_positions(self, positions: list) -> list:
        """Stabilize crop positions - reduce jitter and sudden movements"""
        if not positions:
            return positions
        
        # Use longer window for smoother movement
        window_size = 60  # ~2 seconds at 30fps - longer window = smoother
        stabilized = []
        
        for i in range(len(positions)):
            # Get window around current position
            start = max(0, i - window_size // 2)
            end = min(len(positions), i + window_size // 2)
            window = positions[start:end]
            
            # Use median for stability (resistant to outliers)
            avg = int(np.median(window))
            stabilized.append(avg)
        
        # Second pass: detect shot changes and lock position per shot
        # A shot change is when position jumps significantly
        # Use very high threshold to minimize scene switches
        final = []
        shot_start = 0
        threshold = 250  # pixels - very high threshold = less scene switches
        min_shot_duration = 90  # minimum frames (~3 seconds) before allowing switch
        
        for i in range(len(stabilized)):
            frames_since_last_switch = i - shot_start
            
            # Only allow switch if enough time has passed AND position changed significantly
            if i > 0 and frames_since_last_switch >= min_shot_duration:
                if abs(stabilized[i] - stabilized[shot_start]) > threshold:
                    # Shot change detected - lock previous shot to median
                    shot_positions = stabilized[shot_start:i]
                    if shot_positions:
                        shot_median = int(np.median(shot_positions))
                        final.extend([shot_median] * len(shot_positions))
                    shot_start = i
        
        # Handle last shot
        shot_positions = stabilized[shot_start:]
        if shot_positions:
            shot_median = int(np.median(shot_positions))
            final.extend([shot_median] * len(shot_positions))
        
        return final if final else stabilized
    
    def _init_mediapipe(self):
        """Initialize MediaPipe Face Mesh (lazy loading)"""
        if self.mp_face_mesh is None:
            try:
                import mediapipe as mp
                self.mp_face_mesh = mp.solutions.face_mesh
                self.mp_drawing = mp.solutions.drawing_utils
                self.log("  MediaPipe initialized successfully")
            except ImportError:
                raise Exception("MediaPipe not installed. Run: pip install mediapipe")
    
    def convert_to_portrait_mediapipe(self, input_path: str, output_path: str):
        """Convert landscape to 9:16 portrait with active speaker detection (MediaPipe)"""
        
        # Initialize MediaPipe
        self._init_mediapipe()
        
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise Exception(f"Failed to open video: {input_path}")
        
        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0 or fps == 0:
            cap.release()
            raise Exception(f"Invalid video properties: {total_frames} frames, {fps} fps")
        
        # Calculate crop dimensions
        target_ratio = 9 / 16
        crop_w = int(orig_h * target_ratio)
        crop_h = orig_h
        out_w, out_h = 1080, 1920
        
        # MediaPipe Face Mesh settings
        lip_threshold = self.mediapipe_settings.get("lip_activity_threshold", 0.15)
        switch_threshold = self.mediapipe_settings.get("switch_threshold", 0.3)
        min_shot_duration = self.mediapipe_settings.get("min_shot_duration", 90)
        center_weight = self.mediapipe_settings.get("center_weight", 0.3)
        
        # First pass: analyze frames with MediaPipe
        self.log("  Pass 1: Analyzing lip movements...")
        crop_positions = []
        face_activities = []  # Store activity scores per frame
        
        with self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=3,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as face_mesh:
            
            frame_count = 0
            prev_lip_distances = {}  # Track previous lip distances per face
            
            while True:
                if self.is_cancelled():
                    cap.release()
                    raise Exception("Cancelled by user")
                
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convert to RGB for MediaPipe
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(rgb_frame)
                
                best_face_x = orig_w / 2  # Default to center
                max_activity = 0
                
                if results.multi_face_landmarks:
                    faces_data = []
                    
                    for face_id, face_landmarks in enumerate(results.multi_face_landmarks):
                        # Calculate lip activity
                        activity = self._calculate_lip_activity(
                            face_landmarks, 
                            orig_w, 
                            orig_h,
                            prev_lip_distances.get(face_id, None)
                        )
                        
                        # Get face center position
                        face_x = face_landmarks.landmark[1].x * orig_w  # Nose tip
                        
                        # Calculate combined score (activity + center position)
                        center_score = 1.0 - abs(face_x - orig_w / 2) / (orig_w / 2)
                        combined_score = (activity * (1 - center_weight)) + (center_score * center_weight)
                        
                        faces_data.append({
                            'x': face_x,
                            'activity': activity,
                            'combined_score': combined_score
                        })
                        
                        # Update previous lip distance
                        upper_lip = face_landmarks.landmark[13]  # Upper lip center
                        lower_lip = face_landmarks.landmark[14]  # Lower lip center
                        lip_distance = abs(upper_lip.y - lower_lip.y)
                        prev_lip_distances[face_id] = lip_distance
                    
                    # Select face with highest combined score
                    if faces_data:
                        best_face = max(faces_data, key=lambda f: f['combined_score'])
                        best_face_x = best_face['x']
                        max_activity = best_face['activity']
                
                # Calculate crop position
                crop_x = int(best_face_x - crop_w / 2)
                crop_x = max(0, min(crop_x, orig_w - crop_w))
                crop_positions.append(crop_x)
                face_activities.append(max_activity)
                
                frame_count += 1
                
                if frame_count % 30 == 0:
                    self.log(f"    Analyzed {frame_count}/{total_frames} frames...")
        
        self.log(f"  Analyzed {frame_count} frames with MediaPipe")
        
        # Stabilize positions with shot-based switching
        crop_positions = self._stabilize_positions_with_activity(
            crop_positions, 
            face_activities,
            min_shot_duration,
            switch_threshold
        )
        
        # Second pass: create video
        self.log("  Pass 2: Creating portrait video...")
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        temp_video = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False).name
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_video, fourcc, fps, (out_w, out_h))
        
        if not out.isOpened():
            cap.release()
            raise Exception(f"Failed to create VideoWriter: {temp_video}")
        
        frame_idx = 0
        while True:
            if self.is_cancelled():
                cap.release()
                out.release()
                try:
                    os.unlink(temp_video)
                except:
                    pass
                raise Exception("Cancelled by user")
            
            ret, frame = cap.read()
            if not ret:
                break
            
            crop_x = crop_positions[frame_idx] if frame_idx < len(crop_positions) else crop_positions[-1]
            cropped = frame[0:crop_h, crop_x:crop_x+crop_w]
            resized = cv2.resize(cropped, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4)
            out.write(resized)
            
            frame_idx += 1
            
            if frame_idx % 30 == 0:
                self.log(f"    Created {frame_idx}/{total_frames} frames...")
        
        cap.release()
        out.release()
        
        # Verify temp video was created
        if not os.path.exists(temp_video) or os.path.getsize(temp_video) < 1000:
            raise Exception(f"Failed to create temp video: {temp_video}")
        
        # Merge with audio using GPU/CPU encoder
        self.log("  Pass 3: Merging audio...")
        encoder_args = self.get_video_encoder_args()
        cmd = [
            self.ffmpeg_path, "-y",
            *self.get_hwaccel_args(),
            "-i", temp_video,
            "-i", input_path,
            *encoder_args,
            "-c:a", "aac", "-b:a", "192k",
            "-map", "0:v:0", "-map", "1:a:0",
            "-shortest",
            output_path
        ]
        self.log_ffmpeg_command(cmd, "Portrait Merge Audio (MediaPipe)")
        self._run_ffmpeg_subprocess(cmd)
        
        # Cleanup
        try:
            os.unlink(temp_video)
        except:
            pass
    
    def _calculate_lip_activity(self, face_landmarks, frame_width, frame_height, prev_lip_distance=None):
        """Calculate lip movement activity score"""
        
        # Key lip landmarks (MediaPipe Face Mesh indices)
        # Upper lip: 13, Lower lip: 14
        upper_lip = face_landmarks.landmark[13]
        lower_lip = face_landmarks.landmark[14]
        
        # Mouth corners: 61 (left), 291 (right)
        mouth_left = face_landmarks.landmark[61]
        mouth_right = face_landmarks.landmark[291]
        
        # Calculate mouth openness (vertical distance)
        mouth_height = abs(upper_lip.y - lower_lip.y)
        
        # Calculate mouth width (horizontal distance)
        mouth_width = abs(mouth_left.x - mouth_right.x)
        
        # Aspect ratio (height/width) - higher when mouth is open
        if mouth_width > 0:
            aspect_ratio = mouth_height / mouth_width
        else:
            aspect_ratio = 0
        
        # Calculate movement delta (change from previous frame)
        delta = 0
        if prev_lip_distance is not None:
            delta = abs(mouth_height - prev_lip_distance)
        
        # Activity score: combination of openness and movement
        # Weight movement more heavily (0.6) than static openness (0.4)
        activity_score = (aspect_ratio * 0.4) + (delta * 0.6)
        
        return activity_score
    
    def _stabilize_positions_with_activity(self, positions, activities, min_shot_duration, switch_threshold):
        """Stabilize crop positions based on activity scores"""
        if not positions:
            return positions
        
        # First pass: smooth positions with moving median
        window_size = 30
        smoothed = []
        
        for i in range(len(positions)):
            start = max(0, i - window_size // 2)
            end = min(len(positions), i + window_size // 2)
            window = positions[start:end]
            smoothed.append(int(np.median(window)))
        
        # Second pass: lock positions per shot based on activity
        final = []
        shot_start = 0
        current_position = smoothed[0] if smoothed else 0
        
        for i in range(len(smoothed)):
            frames_since_switch = i - shot_start
            
            # Only allow switch if:
            # 1. Minimum shot duration has passed
            # 2. Position changed significantly
            # 3. Activity is high enough (speaker is talking)
            if frames_since_switch >= min_shot_duration:
                position_diff = abs(smoothed[i] - current_position)
                activity = activities[i] if i < len(activities) else 0
                
                # Switch if position changed significantly AND there's activity
                if position_diff > 200 and activity > switch_threshold:
                    # Lock previous shot
                    shot_positions = smoothed[shot_start:i]
                    if shot_positions:
                        shot_median = int(np.median(shot_positions))
                        final.extend([shot_median] * len(shot_positions))
                    
                    shot_start = i
                    current_position = smoothed[i]
        
        # Handle last shot
        shot_positions = smoothed[shot_start:]
        if shot_positions:
            shot_median = int(np.median(shot_positions))
            final.extend([shot_median] * len(shot_positions))
        
        return final if final else smoothed
    

    
    def add_captions_api(self, input_path: str, output_path: str, audio_source: str = None, time_offset: float = 0):
        """Add CapCut-style captions using OpenAI Whisper API
        
        Args:
            input_path: Video to burn captions into (with hook)
            output_path: Output video path
            audio_source: Video to extract audio from for transcription (without hook)
            time_offset: Offset to add to all timestamps (hook duration)
        """
        return self.add_captions_api_with_progress(
            input_path,
            output_path,
            audio_source=audio_source,
            time_offset=time_offset,
            progress_callback=lambda _progress: None,
        )
    
    def create_ass_subtitle_capcut(self, transcript, output_path: str, time_offset: float = 0):
        """Create ASS subtitle file with CapCut-style word-by-word highlighting"""
        
        # ASS header - CapCut style: white text, yellow highlight, black outline
        ass_content = """[Script Info]
Title: Auto-generated captions
ScriptType: v4.00+
WrapStyle: 0
PlayResX: 1080
PlayResY: 1920
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial Black,65,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,4,2,2,50,50,400,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
        
        events = []
        
        # Check if we have word-level timestamps
        if hasattr(transcript, 'words') and transcript.words:
            words = transcript.words
            
            # Group words into chunks (3-4 words per line for readability)
            chunk_size = 4
            
            for i in range(0, len(words), chunk_size):
                chunk = words[i:i + chunk_size]
                if not chunk:
                    continue
                
                # For each word in the chunk, create a subtitle event with that word highlighted
                for j, current_word in enumerate(chunk):
                    # Add time_offset to account for hook duration
                    word_start = current_word.start + time_offset
                    word_end = current_word.end + time_offset
                    
                    # Build text with current word highlighted in yellow
                    text_parts = []
                    for k, w in enumerate(chunk):
                        word_text = w.word.strip().upper()
                        if k == j:
                            # Highlight current word (yellow: &H00FFFF in BGR)
                            text_parts.append(f"{{\\c&H00FFFF&}}{word_text}{{\\c&HFFFFFF&}}")
                        else:
                            text_parts.append(word_text)
                    
                    text = " ".join(text_parts)
                    
                    events.append({
                        'start': self.format_time(word_start),
                        'end': self.format_time(word_end),
                        'text': text
                    })
        
        # Fallback: use segment-level timestamps if no word timestamps
        elif hasattr(transcript, 'segments') and transcript.segments:
            for segment in transcript.segments:
                start = segment.get('start', 0) + time_offset
                end = segment.get('end', 0) + time_offset
                text = segment.get('text', '').strip().upper()
                
                if text:
                    events.append({
                        'start': self.format_time(start),
                        'end': self.format_time(end),
                        'text': text
                    })
        
        # Write events to ASS file
        for event in events:
            ass_content += f"Dialogue: 0,{event['start']},{event['end']},Default,,0,0,0,,{event['text']}\n"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(ass_content)
    
    def format_time(self, seconds: float) -> str:
        """Convert seconds to ASS time format"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        centisecs = int((seconds % 1) * 100)
        return f"{hours}:{minutes:02d}:{secs:02d}.{centisecs:02d}"
    
    def parse_timestamp(self, ts: str) -> float:
        """Convert timestamp to seconds"""
        ts = ts.replace(",", ".")
        parts = ts.split(":")
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])

    def _create_clip_output_dir(self, index: int) -> Path:
        """Create the timestamped output directory for a processed clip."""
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S") + f"-{index:02d}"
        clip_dir = self.output_dir / timestamp
        clip_dir.mkdir(parents=True, exist_ok=True)
        return clip_dir

    def _get_clip_total_steps(self, add_captions: bool, add_hook: bool) -> int:
        """Return the initial progress step count for process_clip."""
        total_steps = 2  # Cut/re-encode and portrait conversion are always run.
        if add_hook:
            total_steps += 1
        if add_captions:
            total_steps += 1
        return total_steps

    def _report_clip_progress(
        self,
        index: int,
        total_clips: int,
        total_steps: int,
        step_name: str,
        step_num: int,
        sub_progress: float = 0,
    ):
        """Report per-clip progress using the existing overall progress scale."""
        clip_base = 0.3 + (0.6 * (index - 1) / total_clips)
        clip_portion = 0.6 / total_clips
        step_progress = clip_portion * ((step_num + sub_progress) / total_steps)
        overall = clip_base + step_progress

        percent = int(sub_progress * 100)
        if percent > 0:
            status = f"Clip {index}/{total_clips}: {step_name} ({percent}%)"
        else:
            status = f"Clip {index}/{total_clips}: {step_name}"

        print(f"[DEBUG] clip_progress: {status} (overall: {overall*100:.1f}%)")
        self.set_progress(status, overall)

    def _delete_clip_temp_file(self, path: Path, display_name: str = None):
        """Delete one temporary clip file, logging the same warning style as before."""
        try:
            if path.exists():
                path.unlink()
        except Exception as e:
            self.log(f"  Warning: Could not delete {display_name or path.name}: {e}")

    def _cleanup_clip_temp_files(self, clip_dir: Path, landscape_file: Path, portrait_file: Path, add_hook: bool):
        """Clean up temporary files created by process_clip."""
        self._delete_clip_temp_file(landscape_file)
        self._delete_clip_temp_file(portrait_file)
        if add_hook:
            self._delete_clip_temp_file(clip_dir / "temp_hooked.mp4", "temp_hooked.mp4")

    def _write_clip_metadata(self, clip_dir: Path, highlight: dict, add_hook: bool, add_captions: bool):
        """Write the clip metadata file produced by process_clip."""
        metadata = {
            "title": highlight["title"],
            "hook_text": highlight.get("hook_text", highlight["title"]),
            "start_time": highlight["start_time"],
            "end_time": highlight["end_time"],
            "duration_seconds": highlight["duration_seconds"],
            "has_hook": add_hook,
            "has_captions": add_captions,
            "has_watermark": self.watermark_settings.get("enabled", False),
            "has_credit": self.credit_watermark_settings.get("enabled", False),
            "channel_name": self.channel_name,
        }

        with open(clip_dir / "data.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    def cleanup(self):
        """Clean up temp files"""
        import shutil
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def run_ffmpeg_with_progress(self, cmd: list, duration: float, progress_callback):
        """Run ffmpeg command and parse progress"""
        print(f"[DEBUG] Running ffmpeg command: {' '.join(cmd[:5])}...")
        print(f"[DEBUG] Expected duration: {duration}s")
        
        # Just run ffmpeg normally without progress parsing for now
        # Progress parsing from ffmpeg is complex due to carriage returns
        # _run_ffmpeg_subprocess auto-falls-back to libx264 if a GPU encoder
        # error is detected (e.g. invalid preset on h264_qsv).
        result = self._run_ffmpeg_subprocess(cmd)
        
        # Set to 100% when done
        progress_callback(1.0)
        print(f"[DEBUG] FFmpeg completed with return code: {result.returncode}")
        
        if result.returncode != 0:
            error_msg = result.stderr if result.stderr else "Unknown FFmpeg error"
            
            # Extract the actual error (usually at the end)
            error_lines = error_msg.split('\n')
            relevant_errors = [line for line in error_lines if any(keyword in line.lower() for keyword in 
                ['error', 'invalid', 'failed', 'cannot', 'unable', 'not found', 'does not exist'])]
            
            # Get last 10 lines which usually contain the actual error
            last_lines = '\n'.join(error_lines[-10:])
            
            print(f"[FFMPEG ERROR] Full stderr:\n{error_msg}")
            self.log(f"FFmpeg command failed: {' '.join(cmd)}")
            self.log(f"FFmpeg full error output:\n{error_msg}")
            
            # Show relevant error or last lines
            if relevant_errors:
                error_summary = '\n'.join(relevant_errors[-5:])
            else:
                error_summary = last_lines
            
            raise Exception(f"FFmpeg process failed:\n{error_summary}")
    
    def convert_to_portrait_with_progress(self, input_path: str, output_path: str, progress_callback):
        """Convert landscape to 9:16 portrait with active speaker tracking and progress"""
        try:
            mode = getattr(self, "detection_engine", None) or getattr(self, "face_tracking_mode", "hybrid_auto")
            
            if mode == "mediapipe":
                self.log("  Using legacy MediaPipe (Active Speaker Detection)")
                return self.convert_to_portrait_mediapipe_with_progress(input_path, output_path, progress_callback)
            else:
                self.log(f"  Using Portrait Engine Manager (Mode: {mode})")
                return self.convert_to_portrait_opencv_with_progress(input_path, output_path, progress_callback)
                
        except Exception as e:
            # Fallback to OpenCV if anything fails
            self.log(f"  ⚠ Tracking engine failed: {e}")
            self.log("  Falling back to OpenCV Fast mode...")
            
            # Temporarily force opencv_fast for fallback
            self.face_tracking_mode = "opencv_fast"
            self.detection_engine = "opencv_fast"
            return self.convert_to_portrait_opencv_with_progress(input_path, output_path, progress_callback)
    
    def convert_to_portrait_opencv_with_progress(self, input_path: str, output_path: str, progress_callback):
        """Convert landscape to 9:16 portrait with speaker tracking and progress (OpenCV)"""
        
        self.log("[DEBUG] Starting portrait conversion...")
        print("[DEBUG] Starting portrait conversion...")
        print(f"[DEBUG] Input: {input_path}")
        print(f"[DEBUG] Output: {output_path}")
        sys.stdout.flush()
        
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise Exception(f"Failed to open video: {input_path}")
        
        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        self.log(f"[DEBUG] Video: {orig_w}x{orig_h}, {fps}fps, {total_frames} frames")
        print(f"[DEBUG] Video: {orig_w}x{orig_h}, {fps}fps, {total_frames} frames")
        sys.stdout.flush()
        
        if total_frames == 0 or fps == 0:
            cap.release()
            raise Exception(f"Invalid video properties: {total_frames} frames, {fps} fps")
        
        # Calculate crop dimensions
        target_ratio = 9 / 16
        crop_w = int(orig_h * target_ratio)
        crop_h = orig_h
        out_w, out_h = 1080, 1920
        
        # First pass: analyze frames using EngineManager (0-40%)
        from services.portrait_service import EngineManager
        engine_mode = getattr(self, "detection_engine", None) or getattr(self, "face_tracking_mode", "opencv_fast")
        engine_manager = EngineManager(mode=engine_mode, settings=self.performance_settings)
        # Ensure we respect the config if it exists, otherwise default to balanced
        profile = getattr(self, "performance_profile", "balanced")
        engine_name = engine_manager.setup_engine(profile=profile)
        interval = engine_manager.get_detection_interval()
        self.log(f"  Detection engine: {engine_name}")
        self.log(f"  Detection profile: {profile}, interval: every {interval} frames")
        
        crop_positions = engine_manager.process_pass_1(
            cap, total_frames, orig_w, orig_h, crop_w,
            is_cancelled_callback=self.is_cancelled,
            progress_callback=progress_callback
        )
        
        # Stabilize positions
        crop_positions = self.stabilize_positions(crop_positions)
        progress_callback(0.45)
        
        # Second pass: create video (45-85%)
        print("[DEBUG] Pass 2: Creating portrait video...")
        sys.stdout.flush()  # Force output
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        temp_video = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False).name
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_video, fourcc, fps, (out_w, out_h))
        
        if not out.isOpened():
            cap.release()
            raise Exception(f"Failed to create VideoWriter: {temp_video}")
        
        frame_idx = 0
        last_log_time = 0
        import time
        last_frame_time = time.time()
        
        while True:
            # Check for cancellation
            if self.is_cancelled():
                cap.release()
                out.release()
                try:
                    os.unlink(temp_video)
                except:
                    pass
                raise Exception("Cancelled by user")
            
            # Watchdog: check if we're stuck (no frame processed in 30 seconds)
            current_time = time.time()
            if current_time - last_frame_time > 30:
                cap.release()
                out.release()
                raise Exception(f"Portrait conversion timeout: stuck at frame {frame_idx}/{total_frames}")
            
            ret, frame = cap.read()
            if not ret:
                break
            
            last_frame_time = current_time  # Update watchdog timer
            
            crop_x = crop_positions[frame_idx] if frame_idx < len(crop_positions) else crop_positions[-1]
            cropped = frame[0:crop_h, crop_x:crop_x+crop_w]
            resized = cv2.resize(cropped, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4)
            
            # Write frame
            out.write(resized)
            
            frame_idx += 1
            
            # Update progress more frequently and with time-based logging
            if frame_idx % 30 == 0 or (current_time - last_log_time) > 2:  # Every 30 frames or 2 seconds
                progress = 0.45 + (frame_idx / total_frames) * 0.4  # 45-85%
                print(f"[DEBUG] Pass 2 progress: {progress*100:.1f}% ({frame_idx}/{total_frames} frames)")
                sys.stdout.flush()
                progress_callback(progress)
                last_log_time = current_time
        
        print(f"[DEBUG] Created {frame_idx} frames")
        sys.stdout.flush()
        
        cap.release()
        print("[DEBUG] Released VideoCapture")
        sys.stdout.flush()
        
        out.release()
        print("[DEBUG] Released VideoWriter")
        sys.stdout.flush()
        
        # Verify temp video was created
        if not os.path.exists(temp_video) or os.path.getsize(temp_video) < 1000:
            raise Exception(f"Failed to create temp video: {temp_video}")
        
        print(f"[DEBUG] Temp video size: {os.path.getsize(temp_video)} bytes")
        sys.stdout.flush()
        
        progress_callback(0.85)
        
        # Merge with audio (85-100%) using GPU/CPU encoder
        print("[DEBUG] Pass 3: Merging audio...")
        sys.stdout.flush()
        
        duration = total_frames / fps if fps > 0 else 60
        encoder_args = self.get_video_encoder_args()
        cmd = [
            self.ffmpeg_path, "-y",
            *self.get_hwaccel_args(),
            "-i", temp_video,
            "-i", input_path,
            *encoder_args,
            "-c:a", "aac", "-b:a", "192k",
            "-map", "0:v:0", "-map", "1:a:0",
            "-shortest",
            output_path
        ]
        
        # Run without progress parsing for audio merge (quick operation)
        print(f"[DEBUG] Running audio merge command...")
        sys.stdout.flush()
        
        self.log_ffmpeg_command(cmd, "Portrait Merge Audio (with progress)")
        result = self._run_ffmpeg_subprocess(cmd)
        
        if result.returncode != 0:
            print(f"[FFMPEG ERROR] {result.stderr}")
            sys.stdout.flush()
            raise Exception("Audio merge failed")
        
        print("[DEBUG] Audio merge complete")
        sys.stdout.flush()
        
        progress_callback(1.0)
        print("[DEBUG] Portrait conversion complete")
        sys.stdout.flush()
        
        # Cleanup temp video
        try:
            os.unlink(temp_video)
            print("[DEBUG] Cleaned up temp video")
            sys.stdout.flush()
        except Exception as e:
            print(f"[WARNING] Failed to cleanup temp video: {e}")
            sys.stdout.flush()
    
    def convert_to_portrait_mediapipe_with_progress(self, input_path: str, output_path: str, progress_callback):
        """Convert landscape to 9:16 portrait with active speaker detection and progress (MediaPipe)"""
        
        # Initialize MediaPipe
        self._init_mediapipe()
        
        self.log("[DEBUG] Starting MediaPipe portrait conversion...")
        print("[DEBUG] Starting MediaPipe portrait conversion...")
        sys.stdout.flush()
        
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise Exception(f"Failed to open video: {input_path}")
        
        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        self.log(f"[DEBUG] Video: {orig_w}x{orig_h}, {fps}fps, {total_frames} frames")
        print(f"[DEBUG] Video: {orig_w}x{orig_h}, {fps}fps, {total_frames} frames")
        sys.stdout.flush()
        
        if total_frames == 0 or fps == 0:
            cap.release()
            raise Exception(f"Invalid video properties: {total_frames} frames, {fps} fps")
        
        # Calculate crop dimensions
        target_ratio = 9 / 16
        crop_w = int(orig_h * target_ratio)
        crop_h = orig_h
        out_w, out_h = 1080, 1920
        
        # MediaPipe settings
        lip_threshold = self.mediapipe_settings.get("lip_activity_threshold", 0.15)
        switch_threshold = self.mediapipe_settings.get("switch_threshold", 0.3)
        min_shot_duration = self.mediapipe_settings.get("min_shot_duration", 90)
        center_weight = self.mediapipe_settings.get("center_weight", 0.3)
        
        # First pass: analyze frames with MediaPipe (0-40%)
        print("[DEBUG] Pass 1: Analyzing lip movements with MediaPipe...")
        sys.stdout.flush()
        
        crop_positions = []
        face_activities = []
        frame_count = 0
        detected_frames = 0
        skipped_frames = 0
        detection_interval = self.performance_settings.get("detection_interval", 10)
        if not isinstance(detection_interval, int) or detection_interval <= 0:
            detection_interval = {"quality": 5, "balanced": 10, "fast": 30}.get(self.performance_profile, 10)
        last_face_x = orig_w / 2
        last_activity = 0
        last_log_time = 0
        import time
        self.log(f"  Detection engine: legacy_mediapipe")
        self.log(f"  Detection profile: {self.performance_profile}, interval: every {detection_interval} frames")
        
        with self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=3,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as face_mesh:
            
            prev_lip_distances = {}
            
            while True:
                if self.is_cancelled():
                    cap.release()
                    raise Exception("Cancelled by user")
                
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % detection_interval == 0:
                    detected_frames += 1
                    # Convert to RGB for MediaPipe
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = face_mesh.process(rgb_frame)
                    
                    best_face_x = last_face_x
                    max_activity = last_activity
                    
                    if results.multi_face_landmarks:
                        faces_data = []
                        
                        for face_id, face_landmarks in enumerate(results.multi_face_landmarks):
                            # Calculate lip activity
                            activity = self._calculate_lip_activity(
                                face_landmarks,
                                orig_w,
                                orig_h,
                                prev_lip_distances.get(face_id, None)
                            )
                            
                            # Get face center position
                            face_x = face_landmarks.landmark[1].x * orig_w
                            
                            # Combined score
                            center_score = 1.0 - abs(face_x - orig_w / 2) / (orig_w / 2)
                            combined_score = (activity * (1 - center_weight)) + (center_score * center_weight)
                            
                            faces_data.append({
                                'x': face_x,
                                'activity': activity,
                                'combined_score': combined_score
                            })
                            
                            # Update previous lip distance
                            upper_lip = face_landmarks.landmark[13]
                            lower_lip = face_landmarks.landmark[14]
                            lip_distance = abs(upper_lip.y - lower_lip.y)
                            prev_lip_distances[face_id] = lip_distance
                        
                        if faces_data:
                            best_face = max(faces_data, key=lambda f: f['combined_score'])
                            best_face_x = best_face['x']
                            max_activity = best_face['activity']

                    last_face_x = best_face_x
                    last_activity = max_activity
                else:
                    skipped_frames += 1
                    best_face_x = last_face_x
                    max_activity = last_activity
                
                crop_x = int(best_face_x - crop_w / 2)
                crop_x = max(0, min(crop_x, orig_w - crop_w))
                crop_positions.append(crop_x)
                face_activities.append(max_activity)
                
                frame_count += 1
                
                current_time = time.time()
                if frame_count % 30 == 0 or (current_time - last_log_time) > 2:
                    progress = (frame_count / total_frames) * 0.4
                    print(f"[DEBUG] Pass 1 progress: {progress*100:.1f}% ({frame_count}/{total_frames} frames)")
                    sys.stdout.flush()
                    progress_callback(progress)
                    last_log_time = current_time
        
        print(f"[DEBUG] Analyzed {frame_count} frames with MediaPipe")
        sys.stdout.flush()
        self.log(f"  MediaPipe analyzed {frame_count} frames; detected {detected_frames}, skipped {skipped_frames}")
        
        # Stabilize positions (40-45%)
        progress_callback(0.4)
        crop_positions = self._stabilize_positions_with_activity(
            crop_positions,
            face_activities,
            min_shot_duration,
            switch_threshold
        )
        progress_callback(0.45)
        
        # Second pass: create video (45-85%)
        print("[DEBUG] Pass 2: Creating portrait video...")
        sys.stdout.flush()
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        temp_video = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False).name
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_video, fourcc, fps, (out_w, out_h))
        
        if not out.isOpened():
            cap.release()
            raise Exception(f"Failed to create VideoWriter: {temp_video}")
        
        frame_idx = 0
        last_log_time = 0
        last_frame_time = time.time()
        
        while True:
            if self.is_cancelled():
                cap.release()
                out.release()
                try:
                    os.unlink(temp_video)
                except:
                    pass
                raise Exception("Cancelled by user")
            
            current_time = time.time()
            if current_time - last_frame_time > 30:
                cap.release()
                out.release()
                raise Exception(f"Portrait conversion timeout: stuck at frame {frame_idx}/{total_frames}")
            
            ret, frame = cap.read()
            if not ret:
                break
            
            last_frame_time = current_time
            
            crop_x = crop_positions[frame_idx] if frame_idx < len(crop_positions) else crop_positions[-1]
            cropped = frame[0:crop_h, crop_x:crop_x+crop_w]
            resized = cv2.resize(cropped, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4)
            
            out.write(resized)
            
            frame_idx += 1
            
            if frame_idx % 30 == 0 or (current_time - last_log_time) > 2:
                progress = 0.45 + (frame_idx / total_frames) * 0.4
                print(f"[DEBUG] Pass 2 progress: {progress*100:.1f}% ({frame_idx}/{total_frames} frames)")
                sys.stdout.flush()
                progress_callback(progress)
                last_log_time = current_time
        
        print(f"[DEBUG] Created {frame_idx} frames")
        sys.stdout.flush()
        
        cap.release()
        out.release()
        
        if not os.path.exists(temp_video) or os.path.getsize(temp_video) < 1000:
            raise Exception(f"Failed to create temp video: {temp_video}")
        
        print(f"[DEBUG] Temp video size: {os.path.getsize(temp_video)} bytes")
        sys.stdout.flush()
        
        progress_callback(0.85)
        
        # Merge with audio (85-100%) using GPU/CPU encoder
        print("[DEBUG] Pass 3: Merging audio...")
        sys.stdout.flush()
        
        encoder_args = self.get_video_encoder_args()
        cmd = [
            self.ffmpeg_path, "-y",
            *self.get_hwaccel_args(),
            "-i", temp_video,
            "-i", input_path,
            *encoder_args,
            "-c:a", "aac", "-b:a", "192k",
            "-map", "0:v:0", "-map", "1:a:0",
            "-shortest",
            output_path
        ]
        
        self.log_ffmpeg_command(cmd, "MediaPipe Portrait Merge Audio")
        result = self._run_ffmpeg_subprocess(cmd)
        
        if result.returncode != 0:
            print(f"[FFMPEG ERROR] {result.stderr}")
            sys.stdout.flush()
            raise Exception("Audio merge failed")
        
        print("[DEBUG] Audio merge complete")
        sys.stdout.flush()
        
        progress_callback(1.0)
        print("[DEBUG] MediaPipe portrait conversion complete")
        sys.stdout.flush()
        
        # Cleanup
        try:
            os.unlink(temp_video)
            print("[DEBUG] Cleaned up temp video")
            sys.stdout.flush()
        except Exception as e:
            print(f"[WARNING] Failed to cleanup temp video: {e}")
            sys.stdout.flush()
    
    def add_hook(self, input_path: str, hook_text: str, output_path: str, progress_callback=None) -> float:
        """Add hook scene at the beginning with progress tracking"""
        
        if progress_callback is None:
            progress_callback = lambda p: None
        
        # Report TTS character usage
        self.report_tokens(0, 0, 0, len(hook_text))
        
        # Generate TTS audio (10% progress)
        progress_callback(0.1)
        try:
            tts_response = self.tts_client.audio.speech.create(
                model=self.tts_model,
                voice="nova",
                input=hook_text,
                speed=1.0
            )
        except APIConnectionError as e:
            self.log(f"  ❌ TTS API Connection Error: Could not connect to {self.tts_client.base_url}")
            raise Exception(f"TTS API connection failed!\n\nCould not connect to: {self.tts_client.base_url}\nError: {e}")
        except RateLimitError as e:
            self.log(f"  ❌ TTS API Rate Limit: {e}")
            raise Exception(f"TTS API rate limit exceeded!\n\nPlease wait a moment and try again.\nDetails: {e}")
        except APIStatusError as e:
            self.log(f"  ❌ TTS API Error (HTTP {e.status_code}): {e.message}")
            self.log(f"     Model: {self.tts_model}, Base URL: {self.tts_client.base_url}")
            raise Exception(
                f"TTS (Hook) API Error!\n\n"
                f"Status: {e.status_code}\n"
                f"Message: {e.message}\n"
                f"Model: {self.tts_model}\n"
                f"Base URL: {self.tts_client.base_url}\n\n"
                f"Check your Hook Maker API settings."
            )
        except Exception as e:
            self.log(f"  ❌ TTS API Unexpected Error: {type(e).__name__}: {e}")
            raise Exception(f"TTS (Hook) generation failed!\n\nError: {type(e).__name__}: {e}\nModel: {self.tts_model}")
        
        tts_file = tempfile.NamedTemporaryFile(suffix='.mp3', delete=False).name
        with open(tts_file, 'wb') as f:
            f.write(tts_response.content)
        
        progress_callback(0.2)
        
        # Get TTS duration using ffprobe
        probe_cmd = [
            self.ffmpeg_path, "-i", tts_file,
            "-f", "null", "-"
        ]
        result = subprocess.run(probe_cmd, capture_output=True, text=True, creationflags=SUBPROCESS_FLAGS)
        duration_match = re.search(r"Duration: (\d+):(\d+):(\d+\.\d+)", result.stderr)
        
        if duration_match:
            h, m, s = duration_match.groups()
            hook_duration = int(h) * 3600 + int(m) * 60 + float(s) + 0.5
        else:
            hook_duration = 3.0
        
        # Format hook text
        hook_upper = hook_text.upper()
        words = hook_upper.split()
        
        lines = []
        current_line = []
        for word in words:
            current_line.append(word)
            if len(current_line) >= 3:
                lines.append(' '.join(current_line))
                current_line = []
        if current_line:
            lines.append(' '.join(current_line))
        
        # Get input video info
        probe_cmd = [self.ffmpeg_path, "-i", input_path]
        result = subprocess.run(probe_cmd, capture_output=True, text=True, creationflags=SUBPROCESS_FLAGS)
        
        fps_match = re.search(r'(\d+(?:\.\d+)?)\s*fps', result.stderr)
        fps = float(fps_match.group(1)) if fps_match else 30
        
        res_match = re.search(r'(\d{3,4})x(\d{3,4})', result.stderr)
        if res_match:
            width, height = int(res_match.group(1)), int(res_match.group(2))
        else:
            width, height = 1080, 1920
        
        progress_callback(0.3)
        
        # Create hook video in our temp directory
        hook_video = str(self.temp_dir / f"hook_{int(time.time() * 1000)}.mp4")
        
        # Use a simpler approach: create static image with text, then combine with audio
        # This avoids complex FFmpeg filter escaping issues
        
        # First, create a simple background video from first frame using GPU/CPU encoder
        bg_video = str(self.temp_dir / f"hook_bg_{int(time.time() * 1000)}.mp4")
        
        encoder_args = self.get_video_encoder_args()
        bg_cmd = [
            self.ffmpeg_path, "-y",
            *self.get_hwaccel_args(),
            "-i", input_path,
            "-vf", f"trim=0:0.04,loop=loop=-1:size=1:start=0,setpts=N/{fps}/TB",
            "-t", str(hook_duration),
            *encoder_args,
            "-r", str(fps),
            "-s", f"{width}x{height}",
            "-pix_fmt", "yuv420p",
            "-an",
            bg_video
        ]
        
        self.log_ffmpeg_command(bg_cmd, "Create Hook Background")
        result = self._run_ffmpeg_subprocess(bg_cmd)
        if result.returncode != 0:
            self.log(f"Failed to create background video: {result.stderr}")
            raise Exception("Failed to create background video")
        
        # Verify background video was created successfully
        if not os.path.exists(bg_video) or os.path.getsize(bg_video) < 1000:
            raise Exception("Background video was not created properly")
        
        # === Render hook overlay using PIL (supports user-customized font, colors, corners) ===
        from PIL import Image, ImageDraw, ImageFont

        style = self.hook_style_settings or {}
        font_size_frac = float(style.get("font_size", 0.054))
        font_color_hex = style.get("font_color", "#FFD700")
        bg_color_hex = style.get("bg_color", "#FFFFFF")
        corner_radius = int(style.get("corner_radius", 0))
        pos_x = float(style.get("position_x", 0.5))
        pos_y = float(style.get("position_y", 0.333))
        user_font_path = style.get("font_path") or ""

        # Resolve font path with sensible fallbacks
        font_candidates = [user_font_path, self._find_system_font_bold()]
        pil_font = None
        font_px = max(20, int(font_size_frac * width))
        for candidate in font_candidates:
            if not candidate or not os.path.exists(candidate):
                continue
            try:
                pil_font = ImageFont.truetype(candidate, font_px)
                self.log(f"  Hook font: {candidate} @ {font_px}px")
                break
            except Exception as e:
                self.log(f"  ⚠ Failed to load font {candidate}: {e}")
        if pil_font is None:
            self.log("  ⚠ No usable TTF font found, using PIL default (will look basic)")
            pil_font = ImageFont.load_default()

        font_color_rgb = hex_to_rgb(font_color_hex)
        bg_color_rgb = hex_to_rgb(bg_color_hex)

        # Per-line geometry
        padding = max(10, int(font_px * 0.22))
        line_spacing = max(6, int(font_px * 0.25))

        line_metrics = []
        for line in lines:
            try:
                bbox = pil_font.getbbox(line)
            except AttributeError:
                w, h = pil_font.getsize(line)
                bbox = (0, 0, w, h)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
            line_metrics.append({
                "text": line,
                "bbox": bbox,
                "box_w": text_w + padding * 2,
                "box_h": text_h + padding * 2,
            })

        total_h = sum(m["box_h"] for m in line_metrics)
        if len(line_metrics) > 1:
            total_h += line_spacing * (len(line_metrics) - 1)

        center_x = int(pos_x * width)
        center_y = int(pos_y * height)
        block_top = center_y - total_h // 2

        # Compose the static overlay (transparent everywhere except the hook boxes)
        overlay_img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay_img)

        cur_y = block_top
        for m in line_metrics:
            box_w = m["box_w"]
            box_h = m["box_h"]
            box_x1 = center_x - box_w // 2
            box_y1 = cur_y
            box_x2 = box_x1 + box_w
            box_y2 = box_y1 + box_h

            if corner_radius > 0 and hasattr(draw, "rounded_rectangle"):
                # Clamp radius so it never exceeds half the smaller dimension
                r = min(corner_radius, box_w // 2, box_h // 2)
                draw.rounded_rectangle(
                    [box_x1, box_y1, box_x2, box_y2],
                    radius=r,
                    fill=(*bg_color_rgb, 255),
                )
            else:
                draw.rectangle(
                    [box_x1, box_y1, box_x2, box_y2],
                    fill=(*bg_color_rgb, 255),
                )

            # PIL draws text at the top-left of the glyph bounding box;
            # subtract bbox[0]/[1] so the glyphs sit cleanly inside the padding.
            text_x = box_x1 + padding - m["bbox"][0]
            text_y = box_y1 + padding - m["bbox"][1]
            draw.text(
                (text_x, text_y),
                m["text"],
                font=pil_font,
                fill=(*font_color_rgb, 255),
            )

            cur_y = box_y2 + line_spacing

        overlay_png = str(self.temp_dir / f"hook_overlay_{int(time.time() * 1000)}.png")
        overlay_img.save(overlay_png, "PNG")
        progress_callback(0.4)

        # Composite overlay on the (frozen) background video in one FFmpeg pass
        overlay_video = str(self.temp_dir / f"hook_overlay_video_{int(time.time() * 1000)}.mp4")
        encoder_args = self.get_video_encoder_args()
        overlay_cmd = [
            self.ffmpeg_path, "-y",
            *self.get_hwaccel_args(),
            "-i", bg_video,
            "-i", overlay_png,
            "-filter_complex", "[0:v][1:v]overlay=0:0[v]",
            "-map", "[v]",
            *encoder_args,
            "-pix_fmt", "yuv420p",
            "-an",
            overlay_video,
        ]
        self.log_ffmpeg_command(overlay_cmd, "Composite Hook Overlay (PIL)")
        result = self._run_ffmpeg_subprocess(overlay_cmd)
        if result.returncode != 0:
            self.log(f"Failed to composite hook overlay: {result.stderr}")
            raise Exception("Failed to composite hook overlay video")

        if not os.path.exists(overlay_video) or os.path.getsize(overlay_video) < 1000:
            raise Exception("Hook overlay video was not created properly")

        progress_callback(0.55)

        # Both names point at the same file so the rest of the pipeline (audio mux,
        # cleanup) keeps working without further changes.
        current_video = overlay_video
        reencoded_video = overlay_video

        
        # Finally, add audio to re-encoded video
        cmd = [
            self.ffmpeg_path, "-y",
            *self.get_hwaccel_args(),
            "-i", reencoded_video,
            "-i", tts_file,
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "192k",
            "-ar", "44100",
            "-ac", "2",
            "-shortest",
            hook_video
        ]
        
        # Hook creation is 30-60%
        self.run_ffmpeg_with_progress(cmd, hook_duration, 
            lambda p: progress_callback(0.3 + p * 0.3))
        
        # Re-encode main video (60-80%) using GPU/CPU encoder
        progress_callback(0.6)
        main_reencoded = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False).name
        
        # Get main video duration
        probe_cmd = [self.ffmpeg_path, "-i", input_path, "-f", "null", "-"]
        result = subprocess.run(probe_cmd, capture_output=True, text=True, creationflags=SUBPROCESS_FLAGS)
        duration_match = re.search(r"Duration: (\d+):(\d+):(\d+\.\d+)", result.stderr)
        main_duration = 60
        if duration_match:
            h, m, s = duration_match.groups()
            main_duration = int(h) * 3600 + int(m) * 60 + float(s)
        
        encoder_args = self.get_video_encoder_args()
        cmd = [
            self.ffmpeg_path, "-y",
            *self.get_hwaccel_args(),
            "-i", input_path,
            *encoder_args,
            "-r", str(fps),
            "-s", f"{width}x{height}",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-b:a", "192k",
            "-ar", "44100",
            "-ac", "2",
            "-progress", "pipe:1",
            main_reencoded
        ]
        
        self.log_ffmpeg_command(cmd, "Re-encode Main Video for Hook Concat")
        self.run_ffmpeg_with_progress(cmd, main_duration,
            lambda p: progress_callback(0.6 + p * 0.2))
        
        # Concatenate (80-100%)
        progress_callback(0.8)
        concat_list = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False).name
        with open(concat_list, 'w') as f:
            f.write(f"file '{hook_video.replace(chr(92), '/')}'\n")
            f.write(f"file '{main_reencoded.replace(chr(92), '/')}'\n")
        
        cmd = [
            self.ffmpeg_path, "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", concat_list,
            "-c", "copy",
            output_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, creationflags=SUBPROCESS_FLAGS)
        
        if result.returncode != 0:
            # Fallback to filter_complex using GPU/CPU encoder
            encoder_args = self.get_video_encoder_args()
            cmd = [
                self.ffmpeg_path, "-y",
                *self.get_hwaccel_args(),
                "-i", hook_video,
                "-i", main_reencoded,
                "-filter_complex",
                "[0:v:0][0:a:0][1:v:0][1:a:0]concat=n=2:v=1:a=1[outv][outa]",
                "-map", "[outv]",
                "-map", "[outa]",
                *encoder_args,
                "-c:a", "aac",
                "-b:a", "192k",
                "-progress", "pipe:1",
                output_path
            ]
            self.log_ffmpeg_command(cmd, "Concat Hook (filter_complex fallback - old)")
            total_duration = hook_duration + main_duration
            self.run_ffmpeg_with_progress(cmd, total_duration,
                lambda p: progress_callback(0.8 + p * 0.2))
        else:
            progress_callback(1.0)
        
        # Cleanup
        for path in (tts_file, hook_video, main_reencoded, concat_list,
                     bg_video, overlay_video, overlay_png):
            try:
                if path and os.path.exists(path):
                    os.unlink(path)
            except Exception:
                pass
        
        return hook_duration
    
    def add_captions_api_with_progress(self, input_path: str, output_path: str, audio_source: str = None, time_offset: float = 0, progress_callback=None):
        """Add CapCut-style captions using OpenAI Whisper API with progress"""
        
        if progress_callback:
            progress_callback(0.1)
        
        # Use audio_source if provided, otherwise use input_path
        transcribe_source = audio_source if audio_source else input_path
        
        # Extract audio from video
        audio_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False).name
        cmd = [
            self.ffmpeg_path, "-y",
            *self.get_hwaccel_args(),
            "-i", transcribe_source,
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
            audio_file
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, creationflags=SUBPROCESS_FLAGS)
        
        if result.returncode != 0:
            self.log(f"  Warning: Audio extraction failed")
            import shutil
            shutil.copy(input_path, output_path)
            return
        
        if progress_callback:
            progress_callback(0.2)
        
        # Check if audio file exists
        if not os.path.exists(audio_file) or os.path.getsize(audio_file) < 1000:
            self.log(f"  Warning: Audio file too small or missing")
            import shutil
            shutil.copy(input_path, output_path)
            if os.path.exists(audio_file):
                os.unlink(audio_file)
            return
        
        # Get audio duration for token reporting
        probe_cmd = [self.ffmpeg_path, "-i", audio_file, "-f", "null", "-"]
        result = subprocess.run(probe_cmd, capture_output=True, text=True, creationflags=SUBPROCESS_FLAGS)
        duration_match = re.search(r"Duration: (\d+):(\d+):(\d+\.\d+)", result.stderr)
        audio_duration = 0
        if duration_match:
            h, m, s = duration_match.groups()
            audio_duration = int(h) * 3600 + int(m) * 60 + float(s)
            self.report_tokens(0, 0, audio_duration, 0)
        
        if progress_callback:
            progress_callback(0.3)
        
        # Transcribe using Whisper API (raw HTTP for proxy compatibility)
        try:
            transcript = self._whisper_transcribe_words_api(audio_file)
        except Exception as e:
            self.log(f"  Warning: Whisper API error: {e}")
            import shutil
            shutil.copy(input_path, output_path)
            os.unlink(audio_file)
            return
        
        os.unlink(audio_file)
        
        if progress_callback:
            progress_callback(0.5)
        
        # Create ASS subtitle file
        ass_file = tempfile.NamedTemporaryFile(mode='w', suffix='.ass', delete=False, encoding='utf-8').name
        self.create_ass_subtitle_capcut(transcript, ass_file, time_offset)
        
        if progress_callback:
            progress_callback(0.6)
        
        # Burn subtitles into video using GPU/CPU encoder
        ass_path_escaped = ass_file.replace('\\', '/').replace(':', '\\:')
        
        # Get video duration for progress
        probe_cmd = [self.ffmpeg_path, "-i", input_path, "-f", "null", "-"]
        result = subprocess.run(probe_cmd, capture_output=True, text=True, creationflags=SUBPROCESS_FLAGS)
        duration_match = re.search(r"Duration: (\d+):(\d+):(\d+\.\d+)", result.stderr)
        video_duration = 60
        if duration_match:
            h, m, s = duration_match.groups()
            video_duration = int(h) * 3600 + int(m) * 60 + float(s)
        
        encoder_args = self.get_video_encoder_args()
        cmd = [
            self.ffmpeg_path, "-y",
            *self.get_hwaccel_args(),
            "-i", input_path,
            "-vf", f"ass='{ass_path_escaped}'",
            *encoder_args,
            "-c:a", "copy",
            "-progress", "pipe:1",
            output_path
        ]
        
        self.log_ffmpeg_command(cmd, "Burn Captions (old function)")
        
        # Caption burn is 60-100%
        self.run_ffmpeg_with_progress(cmd, video_duration,
            lambda p: progress_callback(0.6 + p * 0.4) if progress_callback else None)
        
        os.unlink(ass_file)

    def add_watermark_with_progress(self, input_path: str, output_path: str, progress_callback):
        """Add watermark overlay to video with progress tracking"""
        
        watermark_path = self.watermark_settings.get("image_path", "")
        if not watermark_path or not Path(watermark_path).exists():
            self.log("  Warning: Watermark image not found, skipping")
            import shutil
            shutil.copy(input_path, output_path)
            return
        
        progress_callback(0.1)
        
        # Get video dimensions
        probe_cmd = [self.ffmpeg_path, "-i", input_path]
        result = subprocess.run(probe_cmd, capture_output=True, text=True, creationflags=SUBPROCESS_FLAGS)
        
        res_match = re.search(r'(\d{3,4})x(\d{3,4})', result.stderr)
        if res_match:
            video_width, video_height = int(res_match.group(1)), int(res_match.group(2))
        else:
            video_width, video_height = 1080, 1920
        
        progress_callback(0.2)
        
        # Calculate watermark size and position
        scale = self.watermark_settings.get("scale", 0.15)
        pos_x = self.watermark_settings.get("position_x", 0.85)
        pos_y = self.watermark_settings.get("position_y", 0.05)
        opacity = self.watermark_settings.get("opacity", 0.8)
        
        # Calculate watermark width in pixels
        watermark_width = int(video_width * scale)
        
        # Calculate position in pixels
        x_pixels = int(pos_x * video_width)
        y_pixels = int(pos_y * video_height)
        
        # Escape watermark path for FFmpeg (Windows paths)
        watermark_escaped = watermark_path.replace('\\', '/').replace(':', '\\:')
        
        # Build FFmpeg overlay filter with proper opacity control
        # Scale watermark, apply opacity via colorchannelmixer, then overlay
        filter_complex = (
            f"[1:v]scale={watermark_width}:-1,format=rgba,"
            f"colorchannelmixer=aa={opacity}[wm];"
            f"[0:v][wm]overlay={x_pixels}:{y_pixels}"
        )
        
        progress_callback(0.3)
        
        # Get video duration for progress
        duration_match = re.search(r"Duration: (\d+):(\d+):(\d+\.\d+)", result.stderr)
        video_duration = 60
        if duration_match:
            h, m, s = duration_match.groups()
            video_duration = int(h) * 3600 + int(m) * 60 + float(s)
        
        # Apply watermark using GPU/CPU encoder
        encoder_args = self.get_video_encoder_args()
        cmd = [
            self.ffmpeg_path, "-y",
            *self.get_hwaccel_args(),
            "-i", input_path,
            "-i", watermark_path,
            "-filter_complex", filter_complex,
            *encoder_args,
            "-pix_fmt", "yuv420p",  # Ensure compatibility
            "-c:a", "copy",
            "-movflags", "+faststart",  # Enable streaming
            "-progress", "pipe:1",
            output_path
        ]
        
        self.log_ffmpeg_command(cmd, "Apply Watermark")
        
        # Watermark application is 30-100%
        self.run_ffmpeg_with_progress(cmd, video_duration,
            lambda p: progress_callback(0.3 + p * 0.7))
        
        if not Path(output_path).exists():
            raise Exception("Failed to apply watermark")

    def add_credit_watermark_with_progress(self, input_path: str, output_path: str, progress_callback):
        """Add credit text watermark (channel name) to video with progress tracking"""
        
        if not self.channel_name:
            self.log("  Warning: No channel name available, skipping credit")
            import shutil
            shutil.copy(input_path, output_path)
            return
        
        progress_callback(0.1)
        
        # Get video dimensions
        probe_cmd = [self.ffmpeg_path, "-i", input_path]
        result = subprocess.run(probe_cmd, capture_output=True, text=True, creationflags=SUBPROCESS_FLAGS)
        
        res_match = re.search(r'(\d{3,4})x(\d{3,4})', result.stderr)
        if res_match:
            video_width, video_height = int(res_match.group(1)), int(res_match.group(2))
        else:
            video_width, video_height = 1080, 1920
        
        progress_callback(0.2)
        
        # Get credit watermark settings
        size = self.credit_watermark_settings.get("size", 0.03)
        pos_x = self.credit_watermark_settings.get("position_x", 0.5)
        pos_y = self.credit_watermark_settings.get("position_y", 0.95)
        opacity = self.credit_watermark_settings.get("opacity", 0.7)
        
        # Calculate font size in pixels (based on video height)
        font_size = int(video_height * size)
        
        # Calculate position in pixels
        x_pixels = int(pos_x * video_width)
        y_pixels = int(pos_y * video_height)
        
        # Prepare credit text
        credit_text = f"Source: {self.channel_name}"
        # Escape special characters for FFmpeg drawtext
        credit_text_escaped = credit_text.replace("'", "'\\''").replace(":", "\\:")
        
        # Build FFmpeg drawtext filter
        # Use fontfile for portable FFmpeg (avoids fontconfig dependency)
        # Try to find a system font, fallback to built-in if not available
        font_file = None
        if sys.platform == "win32":
            # Windows fonts directory
            windows_fonts = [
                "C:/Windows/Fonts/arial.ttf",
                "C:/Windows/Fonts/segoeui.ttf",
                "C:/Windows/Fonts/tahoma.ttf",
            ]
            for font in windows_fonts:
                if Path(font).exists():
                    font_file = font.replace("\\", "/").replace(":", "\\:")
                    break
        
        # Build filter string
        if font_file:
            filter_str = (
                f"drawtext=fontfile='{font_file}':"
                f"text='{credit_text_escaped}':"
                f"fontsize={font_size}:"
                f"fontcolor=white@{opacity}:"
                f"borderw=2:"
                f"bordercolor=black@{opacity}:"
                f"x={x_pixels}-(text_w/2):"
                f"y={y_pixels}-(text_h/2)"
            )
        else:
            # Fallback without fontfile (may cause fontconfig warning but should still work)
            filter_str = (
                f"drawtext=text='{credit_text_escaped}':"
                f"fontsize={font_size}:"
                f"fontcolor=white@{opacity}:"
                f"borderw=2:"
                f"bordercolor=black@{opacity}:"
                f"x={x_pixels}-(text_w/2):"
                f"y={y_pixels}-(text_h/2)"
            )
        
        progress_callback(0.3)
        
        # Get video duration for progress
        duration_match = re.search(r"Duration: (\d+):(\d+):(\d+\.\d+)", result.stderr)
        video_duration = 60
        if duration_match:
            h, m, s = duration_match.groups()
            video_duration = int(h) * 3600 + int(m) * 60 + float(s)
        
        # Apply credit text using GPU/CPU encoder
        encoder_args = self.get_video_encoder_args()
        cmd = [
            self.ffmpeg_path, "-y",
            *self.get_hwaccel_args(),
            "-i", input_path,
            "-vf", filter_str,
            *encoder_args,
            "-c:a", "copy",
            "-movflags", "+faststart",
            "-progress", "pipe:1",
            output_path
        ]
        
        self.log_ffmpeg_command(cmd, "Apply Credit Watermark")
        
        # Credit application is 30-100%
        self.run_ffmpeg_with_progress(cmd, video_duration,
            lambda p: progress_callback(0.3 + p * 0.7))
        
        if not Path(output_path).exists():
            raise Exception("Failed to apply credit watermark")

    def find_highlights_only(self, url: str, num_clips: int = 5) -> dict:
        """Phase 1: Download subtitle only and find highlights (no video download)
        
        Returns:
            dict with keys:
                - 'session_dir': Path to session directory
                - 'url': YouTube video URL (for later section download)
                - 'srt_path': Path to subtitle file
                - 'highlights': List of highlight dicts with metadata + transcript
                - 'video_info': Video metadata (title, channel, etc.)
        """
        # Create session directory with timestamp
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = self.output_dir / "sessions" / timestamp
        session_dir.mkdir(parents=True, exist_ok=True)
        
        # Update temp_dir to session-specific temp
        self.temp_dir = session_dir / "_temp"
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        self.log(f"Session directory: {session_dir}")
        
        # Step 1: Download subtitle only (no video!)
        self.set_progress("Downloading subtitle...", 0.1)
        srt_path, video_info = self.download_service.download_subtitle_only(url)
        
        # Store channel name for credit watermark
        self.channel_name = video_info.get("channel", "") if video_info else ""
        
        if self.is_cancelled():
            return None
        
        if not srt_path:
            raise SubtitleNotFoundError(
                f"No subtitle available for language: {self.subtitle_language.upper()}",
                video_path=None,
                video_info=video_info,
                session_dir=str(session_dir)
            )
        
        # Step 2: Find highlights
        self.set_progress("Finding highlights with AI...", 0.5)
        transcript = self.download_service.parse_srt(srt_path)
        highlights = self.find_highlights(transcript, video_info, num_clips)
        
        if self.is_cancelled():
            return None
        
        if not highlights:
            raise Exception(
                "❌ No valid highlights found!\n\n"
                "Possible causes:\n"
                "1. AI model failed to generate highlights\n"
                "2. Video transcript too short or not suitable\n"
                "3. AI model configuration issue\n\n"
                "Try:\n"
                "- Using a different AI model (GPT-4, Gemini, etc.)\n"
                "- Checking AI API settings\n"
                "- Using a longer video with more content"
            )
        
        # Extract transcript text for each highlight
        for h in highlights:
            h["transcript_text"] = self.download_service.extract_transcript_for_highlight(srt_path, h)
        
        self.set_progress("Highlights found!", 1.0)
        self.log(f"\n✅ Found {len(highlights)} highlights")
        
        # Save session data to JSON for resume capability
        session_data_file = session_dir / "session_data.json"
        session_data = {
            "session_dir": str(session_dir),
            "url": url,
            "srt_path": srt_path,
            "highlights": highlights,
            "video_info": video_info,
            "created_at": datetime.now().isoformat(),
            "status": "highlights_found"
        }
        
        with open(session_data_file, "w", encoding="utf-8") as f:
            json.dump(session_data, f, indent=2, ensure_ascii=False)
        
        self.log(f"Session data saved to: {session_data_file}")
        
        return session_data
    
    def process_selected_highlights(self, url: str, selected_highlights: list, 
                                   session_dir: Path, add_captions: bool = True, 
                                   add_hook: bool = True):
        """Phase 2: Download video sections and process selected highlights
        
        Args:
            url: YouTube video URL (for downloading sections)
            selected_highlights: List of highlight dicts to process
            session_dir: Session directory for output
            add_captions: Whether to add captions
            add_hook: Whether to add hook
        """
        if not selected_highlights:
            raise Exception("No highlights selected for processing")
        
        self.log(f"\n[Processing {len(selected_highlights)} selected clips]")
        
        # Ensure session_dir is Path object
        if isinstance(session_dir, str):
            session_dir = Path(session_dir)
        
        # Update output_dir to session clips folder
        clips_dir = session_dir / "clips"
        clips_dir.mkdir(parents=True, exist_ok=True)
        
        # Update temp_dir to session-specific temp
        self.temp_dir = session_dir / "_temp"
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Process each selected clip
        total_clips = len(selected_highlights)
        for i, highlight in enumerate(selected_highlights, 1):
            if self.is_cancelled():
                return
            
            # Step A: Download video section for this clip
            self.set_progress(f"Clip {i}/{total_clips}: Downloading video section...", 
                            0.05 + (0.9 * (i - 1) / total_clips))
            self.log(f"\n[Clip {i}/{total_clips}] Downloading: {highlight.get('title', 'Untitled')}")
            
            section_filename = f"section_{i:03d}.mp4"
            section_path = str(self.temp_dir / section_filename)
            
            try:
                video_path = self.download_service.download_video_section(
                    url, 
                    highlight["start_time"], 
                    highlight["end_time"],
                    section_path
                )
            except Exception as e:
                self.log(f"  ✗ Failed to download section: {e}")
                raise Exception(
                    f"Failed to download video section for clip {i}!\n\n"
                    f"Title: {highlight.get('title', 'Untitled')}\n"
                    f"Time: {highlight['start_time']} → {highlight['end_time']}\n\n"
                    f"Error: {str(e)}"
                )
            
            # Step B: Process the downloaded section
            # Create clip-specific folder
            clip_folder = clips_dir / f"clip_{i:03d}"
            clip_folder.mkdir(parents=True, exist_ok=True)
            
            # Temporarily override output_dir for this clip
            original_output_dir = self.output_dir
            self.output_dir = clip_folder.parent
            
            try:
                # Pass pre_cut=True since we downloaded the section already
                self.process_clip(video_path, highlight, i, total_clips, 
                                add_captions=add_captions, add_hook=add_hook,
                                pre_cut=True)
            finally:
                # Restore original output_dir
                self.output_dir = original_output_dir
            
            # Clean up section file after processing
            try:
                if Path(video_path).exists():
                    os.remove(video_path)
            except Exception:
                pass
        
        # Cleanup temp files
        self.set_progress("Cleaning up...", 0.95)
        self.cleanup()
        
        # Update session status to completed
        session_data_file = session_dir / "session_data.json"
        if session_data_file.exists():
            with open(session_data_file, "r", encoding="utf-8") as f:
                session_data = json.load(f)
            
            session_data["status"] = "completed"
            session_data["completed_at"] = datetime.now().isoformat()
            session_data["clips_processed"] = total_clips
            
            with open(session_data_file, "w", encoding="utf-8") as f:
                json.dump(session_data, f, indent=2, ensure_ascii=False)
        
        self.set_progress("Complete!", 1.0)
        self.log(f"\n✅ Created {total_clips} clips in: {clips_dir}")
