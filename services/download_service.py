import os
import json
import subprocess
import re
import shlex
from utils.logger import debug_log
from pathlib import Path
from typing import Optional
from utils.helpers import ensure_binaries_in_path, get_app_dir, get_ffmpeg_path, get_deno_path, parse_timestamp

try:
    import yt_dlp
    YTDLP_MODULE_AVAILABLE = True
except ImportError:
    YTDLP_MODULE_AVAILABLE = False

SUBPROCESS_FLAGS = 0x08000000 if os.name == "nt" else 0


class _SectionYtdlpLogger:
    """Filter yt-dlp verbose output down to FFmpeg command diagnostics."""
    def debug(self, msg):
        text = str(msg)
        lowered = text.lower()
        if "ffmpeg" in lowered and ("command" in lowered or "execut" in lowered or "ffmpeg_i" in lowered):
            debug_log(f"  yt-dlp: {text}")

    def warning(self, msg):
        debug_log(f"  yt-dlp warning: {msg}")

    def error(self, msg):
        debug_log(f"  yt-dlp error: {msg}")


class DownloadService:
    def __init__(self, temp_dir: Path, output_dir: Path, cookies_file: str, subtitle_language: str, ytdlp_path: str, log_callback, progress_callback, is_cancelled_callback, youtube_api_key: str = "", performance_settings: dict = None):
        self.temp_dir = temp_dir
        self.output_dir = output_dir
        self.cookies_file = cookies_file
        self.subtitle_language = subtitle_language
        self.ytdlp_path = ytdlp_path
        self.log_callback = log_callback
        self.progress_callback = progress_callback
        self.is_cancelled_callback = is_cancelled_callback
        self.youtube_api_key = youtube_api_key
        self.performance_settings = dict(performance_settings or {})

    def should_use_gpu(self) -> bool:
        return bool(self.performance_settings.get("prefer_gpu", False))

    def _format_cmd(self, cmd: list) -> str:
        """Format a command for logs without executing through a shell."""
        return " ".join(shlex.quote(str(part)) for part in cmd)

    def _use_ytdlp_module(self) -> bool:
        return YTDLP_MODULE_AVAILABLE and self.ytdlp_path == "yt_dlp_module"

    def _format_selector(self) -> str:
        return "bestvideo[height>=720][height<=2160]+bestaudio/best[height>=720][height<=2160]/bestvideo+bestaudio/best"

    def _find_cookies_path(self, log_checks: bool = False) -> Optional[Path]:
        for loc in [Path("cookies.txt"), get_app_dir() / "cookies.txt"]:
            if log_checks:
                debug_log(f"  Checking cookies at: {loc} - exists: {loc.exists()}")
            if loc.exists():
                return loc
        return None

    def _require_cookies_path(self, log_checks: bool = False) -> Path:
        cookies_path = self._find_cookies_path(log_checks=log_checks)
        if not cookies_path:
            raise Exception("cookies.txt not found!\n\nPlease upload cookies.txt file from home page.")
        return cookies_path

    def _apply_js_runtime_options(self, ydl_opts: dict, deno_path: str, log_status: bool = False):
        if deno_path and Path(deno_path).exists():
            ydl_opts['js_runtimes'] = {'deno': {'path': deno_path}}
            ydl_opts['remote_components'] = ['ejs:github']
            if log_status:
                debug_log(f"  JS runtime: deno at {deno_path}")
        elif log_status:
            debug_log("  WARNING: Deno not found - some formats may be missing!")

    def _apply_ffmpeg_location_options(self, ydl_opts: dict, ffmpeg_path: str, include_subtitle_postprocessor: bool):
        if ffmpeg_path and Path(ffmpeg_path).exists():
            ydl_opts['ffmpeg_location'] = str(Path(ffmpeg_path).parent)
            debug_log(f"  FFmpeg location: {ydl_opts['ffmpeg_location']}")
            if include_subtitle_postprocessor:
                ydl_opts['postprocessors'] = [{
                    'key': 'FFmpegSubtitlesConvertor',
                    'format': 'srt',
                }]
        else:
            debug_log("  WARNING: FFmpeg not found - subtitle conversion disabled")

    def _find_downloaded_srt(self) -> Optional[Path]:
        srt_path = self.temp_dir / f"source.{self.subtitle_language}.srt"
        if srt_path.exists():
            return srt_path

        available_subs = list(self.temp_dir.glob("source.*.srt"))
        if available_subs:
            srt_path = available_subs[0]
            detected_lang = srt_path.stem.split('.')[-1]
            debug_log(f"  ⚠ {self.subtitle_language} subtitle not found, using {detected_lang} instead")
            return srt_path

        debug_log(f"  ✗ No subtitle found for language: {self.subtitle_language}")
        return None

    def _extract_video_encoder(self, args: list) -> str:
        for idx, token in enumerate(args[:-1]):
            if token == "-c:v":
                return args[idx + 1]
        return ""

    def _section_filter_status(self) -> tuple[bool, list]:
        """Section download is a simple trim/re-encode path with no explicit filters here."""
        filters = []
        return False, filters

    def _get_section_ffmpeg_args(self, ffmpeg_path: str) -> dict:
        """Build and log the FFmpeg decode/encode plan used by yt-dlp section downloads."""
        plan = {
            "decode_args": [],
            "encoder_args": [],
            "encoder": "",
            "decoder": "none",
            "gpu_decode": False,
            "gpu_encode": False,
            "cpu_filters": False,
            "filters": [],
            "fallback_reason": "",
        }

        if not self.should_use_gpu():
            plan["fallback_reason"] = "GPU disabled in performance settings"
            debug_log("  Section download: CPU fallback active (GPU disabled)")
            return plan

        from utils.gpu_detector import GPUDetector
        detector = GPUDetector(ffmpeg_path)
        gpu_info = detector.detect_gpu()
        encoder_args = detector.get_encoder_args(
            use_gpu=True,
            preferred_codec=self.performance_settings.get("codec", "h264"),
            encoder=self.performance_settings.get("encoder", "auto")
        )
        encoder = self._extract_video_encoder(encoder_args)

        if encoder in ("libx264", "libx265", ""):
            plan["fallback_reason"] = "No compatible GPU encoder available"
            debug_log(f"  Section download: CPU fallback active ({plan['fallback_reason']})")
            return plan

        cpu_filters, filters = self._section_filter_status()
        decode_enabled = self.performance_settings.get("decode_enabled", True)
        # Do not force CUDA frame output for yt-dlp section downloads. The
        # generated command merges video+audio and has no explicit GPU filter
        # graph; keeping frames in CUDA surfaces can fail even though NVENC is
        # available. Plain "-hwaccel cuda" still enables NVIDIA hardware decode.
        output_format = False
        decode_args = detector.get_decode_args(use_gpu=decode_enabled, output_format=output_format)

        plan.update({
            "decode_args": decode_args,
            "encoder_args": encoder_args,
            "encoder": encoder,
            "decoder": " ".join(decode_args) if decode_args else "none",
            "gpu_decode": bool(decode_args),
            "gpu_encode": encoder not in ("libx264", "libx265"),
            "cpu_filters": cpu_filters,
            "filters": filters,
        })

        if plan["gpu_decode"] and plan["gpu_encode"]:
            mode = "GPU decode+encode active"
        elif plan["gpu_encode"]:
            mode = "GPU encode active"
        elif plan["gpu_decode"]:
            mode = "GPU decode active"
        else:
            mode = "CPU fallback active"

        if cpu_filters and plan["gpu_encode"]:
            mode = "GPU encode active, CPU filters active"

        debug_log(f"  Section download GPU plan: {mode}")
        debug_log(f"    encoder selected: {encoder}")
        debug_log(f"    decoder hwaccel selected: {plan['decoder']}")
        debug_log(f"    filter chain GPU-compatible: {not cpu_filters}")
        debug_log(f"    CPU filters active: {cpu_filters} ({', '.join(filters) if filters else 'none'})")
        if gpu_info.get("type") == "nvidia" and decode_args and not output_format:
            debug_log("    fallback reason: CUDA output surfaces disabled for yt-dlp multi-input section command")
        debug_log(f"    ffmpeg_i args: {decode_args if decode_args else 'none'}")
        debug_log(f"    ffmpeg_o args: {encoder_args if encoder_args else 'none'}")
        return plan

    def log(self, msg: str):
        if self.log_callback:
            self.log_callback(msg)

    def set_progress(self, msg: str, val: float):
        if self.progress_callback:
            self.progress_callback(msg, val)

    def is_cancelled(self) -> bool:
        if self.is_cancelled_callback:
            return self.is_cancelled_callback()
        return False

    def download_video(self, url: str) -> tuple:
        """Download video and subtitle with progress using yt-dlp module or executable"""
        debug_log("[1/4] Downloading video & subtitle...")
        
        # Check if using yt-dlp module
        if self._use_ytdlp_module():
            return self._download_video_module(url)
        return self._download_video_subprocess(url)
    
    def _download_video_module(self, url: str) -> tuple:
        """Download video using yt-dlp Python module API"""
        debug_log(f"  Using yt-dlp module v{yt_dlp.version.__version__}")
        
        video_info = {}
        
        # Get FFmpeg and Deno paths
        ffmpeg_path = get_ffmpeg_path()
        deno_path = get_deno_path()
        
        debug_log(f"  FFmpeg path: {ffmpeg_path}")
        debug_log(f"  Deno path: {deno_path}")
        
        # Setup environment with Deno and FFmpeg in PATH
        paths = ensure_binaries_in_path()
        if not paths["deno_path"]:
            debug_log("  WARNING: Deno not found!")
        
        # Progress hook for yt-dlp
        def progress_hook(d):
            if self.is_cancelled():
                raise Exception("Cancelled by user")
            
            if d['status'] == 'downloading':
                percent_str = d.get('_percent_str', '0%').strip()
                # Extract numeric percent
                match = re.search(r'(\d+\.?\d*)%', percent_str)
                if match:
                    percent = float(match.group(1))
                    self.set_progress(f"Downloading video... {percent:.1f}%", 0.05 + percent / 100 * 0.2)
            elif d['status'] == 'finished':
                debug_log("  Download finished, processing...")
                self.set_progress("Processing downloaded file...", 0.25)
        
        # High-quality format selector
        format_selector = self._format_selector()
        
        # Base yt-dlp options
        ydl_opts = {
            'format': format_selector,
            'format_sort': ['res', 'br'],
            'merge_output_format': 'mp4',
            'outtmpl': str(self.temp_dir / 'source.%(ext)s'),
            'progress_hooks': [progress_hook],
            'quiet': True,
            'no_warnings': False,
            'extract_flat': False,
        }
        
        # Only request subtitles if a real language is selected (skip for AI transcription mode)
        if self.subtitle_language and self.subtitle_language != "none":
            ydl_opts['writesubtitles'] = True
            ydl_opts['writeautomaticsub'] = True
            ydl_opts['subtitleslangs'] = [self.subtitle_language]
            ydl_opts['subtitlesformat'] = 'srt'
        else:
            debug_log("  Skipping subtitle download (AI transcription mode)")
        
        # Add Deno JS runtime if available
        self._apply_js_runtime_options(ydl_opts, deno_path, log_status=True)
        
        # Add FFmpeg location if available
        self._apply_ffmpeg_location_options(
            ydl_opts,
            ffmpeg_path,
            include_subtitle_postprocessor=bool(self.subtitle_language and self.subtitle_language != "none"),
        )
        
        cookies_path = self._require_cookies_path(log_checks=True)
        ydl_opts['cookiefile'] = str(cookies_path)
        debug_log(f"  Using cookies from: {cookies_path}")
        
        # Single download attempt (no browser cookies fallback)
        last_error = None
        try:
            debug_log(f"  Starting download...")
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                # First get video info
                debug_log("  Fetching video info...")
                info = ydl.extract_info(url, download=False)
                
                if info:
                    video_info = {
                        "title": info.get("title", ""),
                        "description": (info.get("description", "") or "")[:2000],
                        "channel": info.get("channel", ""),
                    }
                    debug_log(f"  Title: {video_info['title'][:50]}...")
                
                # Now download
                if self.subtitle_language and self.subtitle_language != "none":
                    debug_log(f"  Downloading video with {self.subtitle_language} subtitle...")
                else:
                    debug_log(f"  Downloading video (no subtitle, AI transcription mode)...")
                ydl.download([url])
            
            debug_log(f"  ✓ Download successful!")
                
        except Exception as e:
            last_error = str(e)
            debug_log(f"  ✗ Failed: {last_error[:100]}")
            
            # Provide helpful error message for common issues
            if "403" in last_error or "Forbidden" in last_error:
                raise Exception(
                    "❌ ERROR: YouTube menolak akses (HTTP 403 Forbidden)\n\n"
                    "PENYEBAB:\n"
                    "• Cookies sudah EXPIRED (biasanya 1-2 minggu)\n"
                    "• Cookies tidak lengkap atau tidak valid\n"
                    "• Browser tidak login ke YouTube saat export cookies\n\n"
                    "SOLUSI:\n"
                    "1. Buka youtube.com di browser\n"
                    "2. PASTIKAN sudah LOGIN ke akun YouTube/Google\n"
                    "3. Export cookies BARU menggunakan extension:\n"
                    "   - Chrome/Edge: 'Get cookies.txt LOCALLY'\n"
                    "   - Firefox: 'cookies.txt'\n"
                    "4. Upload cookies.txt yang baru di halaman Home\n\n"
                    "📖 Lihat COOKIES.md untuk panduan lengkap"
                )
            elif "downloaded file is empty" in last_error.lower() or "file is empty" in last_error.lower():
                raise Exception(
                    "❌ ERROR: File video kosong (0 bytes)\n\n"
                    "PENYEBAB:\n"
                    "• YouTube mendeteksi aktivitas BOT\n"
                    "• Cookies tidak cukup kuat untuk akses video content\n"
                    "• Video mungkin memiliki proteksi khusus\n\n"
                    "SOLUSI:\n"
                    "1. Buka browser INCOGNITO/PRIVATE mode\n"
                    "2. Buka youtube.com dan LOGIN ke akun Google\n"
                    "3. Tonton 2-3 video LENGKAP (bukan skip)\n"
                    "4. Buka video yang ingin di-download, tonton sebentar\n"
                    "5. Export cookies BARU dengan extension:\n"
                    "   - Chrome/Edge: 'Get cookies.txt LOCALLY'\n"
                    "   - Firefox: 'cookies.txt'\n"
                    "6. Upload cookies.txt yang baru\n\n"
                    "💡 TIP: Gunakan akun yang aktif menonton YouTube\n"
                    "📖 Lihat COOKIES.md untuk panduan lengkap"
                )
            elif "Sign in to confirm" in last_error or "bot" in last_error.lower():
                raise Exception(
                    "❌ ERROR: YouTube meminta verifikasi bot\n\n"
                    "PENYEBAB:\n"
                    "• Cookies sudah tidak valid\n"
                    "• YouTube mendeteksi aktivitas mencurigakan\n\n"
                    "SOLUSI:\n"
                    "1. Buka youtube.com di browser INCOGNITO/PRIVATE\n"
                    "2. Login ke akun YouTube/Google\n"
                    "3. Tonton 1-2 video untuk 'warm up' akun\n"
                    "4. Export cookies baru\n"
                    "5. Upload cookies.txt yang baru\n\n"
                    "📖 Lihat COOKIES.md untuk panduan lengkap"
                )
            else:
                raise Exception(f"Download failed!\n\n{last_error}")
        
        video_path = self.temp_dir / "source.mp4"
        srt_path = self._find_downloaded_srt()
        
        return str(video_path), str(srt_path) if srt_path else None, video_info
    
    def _download_video_subprocess(self, url: str) -> tuple:
        """Download video using yt-dlp subprocess (fallback)"""
        # Validate yt-dlp is available
        try:
            version_check = subprocess.run(
                [self.ytdlp_path, "--version"],
                capture_output=True,
                text=True,
                creationflags=SUBPROCESS_FLAGS,
                timeout=5
            )
            if version_check.returncode != 0:
                raise Exception(f"yt-dlp not working properly. Path: {self.ytdlp_path}")
            debug_log(f"  Using yt-dlp version: {version_check.stdout.strip()}")
        except FileNotFoundError:
            raise Exception(f"yt-dlp not found at: {self.ytdlp_path}\n\nPlease install yt-dlp or check the path in settings.")
        except subprocess.TimeoutExpired:
            raise Exception(f"yt-dlp not responding. Path: {self.ytdlp_path}")
        except Exception as e:
            raise Exception(f"Failed to validate yt-dlp: {str(e)}")
        
        base_args = []
        try:
            help_result = subprocess.run(
                [self.ytdlp_path, "--help"],
                capture_output=True,
                text=True,
                creationflags=SUBPROCESS_FLAGS,
                timeout=5
            )
            if help_result.returncode == 0:
                help_text = help_result.stdout
                if "--no-impersonate" in help_text:
                    base_args.append("--no-impersonate")
        except Exception:
            pass
        
        # Get video metadata
        debug_log("  Fetching video info...")
        meta_cmd = [self.ytdlp_path, "--dump-json", "--no-download", *base_args, url]
        
        result = subprocess.run(
            meta_cmd, 
            capture_output=True, 
            text=True,
            creationflags=SUBPROCESS_FLAGS
        )
        video_info = {}
        
        if result.returncode == 0:
            try:
                yt_data = json.loads(result.stdout)
                video_info = {
                    "title": yt_data.get("title", ""),
                    "description": yt_data.get("description", "")[:2000],
                    "channel": yt_data.get("channel", ""),
                }
                debug_log(f"  Title: {video_info['title'][:50]}...")
            except json.JSONDecodeError:
                debug_log("  Warning: Could not parse metadata")
        
        # Download video + subtitle with progress
        if self.subtitle_language and self.subtitle_language != "none":
            debug_log(f"  Downloading video with {self.subtitle_language} subtitle...")
        else:
            debug_log(f"  Downloading video (no subtitle, AI transcription mode)...")
        
        # Try multiple download strategies (fallback on failure)
        download_strategies = [
            {
                "name": "Browser cookies (Chrome)",
                "extra_args": ["--cookies-from-browser", "chrome"]
            },
            {
                "name": "Browser cookies (Edge)",
                "extra_args": ["--cookies-from-browser", "edge"]
            },
            {
                "name": "Simple format (no auth)",
                "extra_args": []
            }
        ]
        
        # High-quality format selector (prioritize 720p+ with fallback)
        format_selector = self._format_selector()
        
        last_error = None
        for strategy in download_strategies:
            if self.is_cancelled():
                raise Exception("Cancelled by user")
            
            debug_log(f"  Trying: {strategy['name']}...")
            
            cmd = [
                self.ytdlp_path,
                "-f", format_selector,
                "--format-sort", "res,br",
                *base_args,
                *strategy["extra_args"],
            ]
            
            # Only request subtitles if a real language is selected
            if self.subtitle_language and self.subtitle_language != "none":
                cmd.extend([
                    "--write-sub", "--write-auto-sub",
                    "--sub-lang", self.subtitle_language,
                    "--convert-subs", "srt",
                ])
            
            cmd.extend([
                "--merge-output-format", "mp4",
                "--newline",
                "-o", str(self.temp_dir / "source.%(ext)s"),
                url
            ])
            
            # Run with realtime progress output
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                creationflags=SUBPROCESS_FLAGS
            )
            
            last_progress = ""
            output_lines = []
            
            while True:
                if self.is_cancelled():
                    process.terminate()
                    process.wait()
                    raise Exception("Cancelled by user")
                
                line = process.stdout.readline()
                if not line and process.poll() is not None:
                    break
                
                line = line.strip()
                output_lines.append(line)
                
                if not line:
                    continue
                    
                # Parse download progress
                if "[download]" in line and "%" in line:
                    match = re.search(r'(\d+\.?\d*)%', line)
                    if match:
                        percent = match.group(1)
                        progress_text = f"  Downloading: {percent}%"
                        if progress_text != last_progress:
                            self.set_progress(f"Downloading video... {percent}%", 0.05 + float(percent) / 100 * 0.2)
                            last_progress = progress_text
                elif "[Merger]" in line or "Merging" in line:
                    debug_log("  Merging video & audio...")
                    self.set_progress("Merging video & audio...", 0.25)
            
            # Check if successful
            if process.returncode == 0:
                debug_log(f"  ✓ Download successful using: {strategy['name']}")
                break
            else:
                # Capture error for logging
                stderr_output = process.stderr.read() if process.stderr else ""
                error_lines = []
                
                for line in output_lines + stderr_output.split('\n'):
                    line = line.strip()
                    if line and ('ERROR' in line or 'error' in line):
                        error_lines.append(line)
                
                last_error = '\n'.join(error_lines[-5:]) if error_lines else f"Return code {process.returncode}"
                debug_log(f"  ✗ Failed: {last_error.split(chr(10))[0][:80]}")  # First line only
                
                # Continue to next strategy
                continue
        else:
            # All strategies failed - provide helpful error message
            if last_error and ("403" in last_error or "Forbidden" in last_error):
                raise Exception(
                    "❌ ERROR: YouTube menolak akses (HTTP 403 Forbidden)\n\n"
                    "PENYEBAB:\n"
                    "• Cookies sudah EXPIRED (biasanya 1-2 minggu)\n"
                    "• Cookies tidak lengkap atau tidak valid\n"
                    "• Browser tidak login ke YouTube saat export cookies\n\n"
                    "SOLUSI:\n"
                    "1. Buka youtube.com di browser\n"
                    "2. PASTIKAN sudah LOGIN ke akun YouTube/Google\n"
                    "3. Export cookies BARU menggunakan extension:\n"
                    "   - Chrome/Edge: 'Get cookies.txt LOCALLY'\n"
                    "   - Firefox: 'cookies.txt'\n"
                    "4. Upload cookies.txt yang baru di halaman Home\n\n"
                    "📖 Lihat COOKIES.md untuk panduan lengkap\n\n"
                    f"Detail error:\n{last_error}"
                )
            elif last_error and ("downloaded file is empty" in last_error.lower() or "file is empty" in last_error.lower()):
                raise Exception(
                    "❌ ERROR: File video kosong (0 bytes)\n\n"
                    "PENYEBAB:\n"
                    "• YouTube mendeteksi aktivitas BOT\n"
                    "• Cookies tidak cukup kuat untuk akses video content\n"
                    "• Video mungkin memiliki proteksi khusus\n\n"
                    "SOLUSI:\n"
                    "1. Buka browser INCOGNITO/PRIVATE mode\n"
                    "2. Buka youtube.com dan LOGIN ke akun Google\n"
                    "3. Tonton 2-3 video LENGKAP (bukan skip)\n"
                    "4. Buka video yang ingin di-download, tonton sebentar\n"
                    "5. Export cookies BARU dengan extension:\n"
                    "   - Chrome/Edge: 'Get cookies.txt LOCALLY'\n"
                    "   - Firefox: 'cookies.txt'\n"
                    "6. Upload cookies.txt yang baru\n\n"
                    "💡 TIP: Gunakan akun yang aktif menonton YouTube\n"
                    "📖 Lihat COOKIES.md untuk panduan lengkap\n\n"
                    f"Detail error:\n{last_error}"
                )
            elif last_error and ("Sign in to confirm" in last_error or "bot" in last_error.lower()):
                raise Exception(
                    "❌ ERROR: YouTube meminta verifikasi bot\n\n"
                    "PENYEBAB:\n"
                    "• Cookies sudah tidak valid\n"
                    "• YouTube mendeteksi aktivitas mencurigakan\n\n"
                    "SOLUSI:\n"
                    "1. Buka youtube.com di browser INCOGNITO/PRIVATE\n"
                    "2. Login ke akun YouTube/Google\n"
                    "3. Tonton 1-2 video untuk 'warm up' akun\n"
                    "4. Export cookies baru\n"
                    "5. Upload cookies.txt yang baru\n\n"
                    "📖 Lihat COOKIES.md untuk panduan lengkap\n\n"
                    f"Detail error:\n{last_error}"
                )
            else:
                raise Exception(f"Download failed after trying all methods!\n\nLast error:\n{last_error}")
        
        video_path = self.temp_dir / "source.mp4"
        srt_path = self._find_downloaded_srt()
        
        return str(video_path), str(srt_path) if srt_path else None, video_info

    @staticmethod
    def get_available_subtitles(url: str, ytdlp_path: str = "yt-dlp", cookies_path: str = None) -> dict:
        """Get list of available subtitles for a YouTube video
        
        Args:
            url: YouTube video URL
            ytdlp_path: Path to yt-dlp executable or "yt_dlp_module" for module
            cookies_path: Path to cookies.txt file (required)
        
        Returns:
            dict with keys:
                - 'subtitles': list of manual subtitle languages
                - 'automatic_captions': list of auto-generated subtitle languages
                - 'error': error message if failed
        """
        # Language name mapping (common ones)
        lang_names = {
            "en": "English",
            "id": "Indonesian",
            "es": "Spanish",
            "fr": "French",
            "de": "German",
            "pt": "Portuguese",
            "ru": "Russian",
            "ja": "Japanese",
            "ko": "Korean",
            "zh": "Chinese",
            "ar": "Arabic",
            "hi": "Hindi",
            "it": "Italian",
            "nl": "Dutch",
            "pl": "Polish",
            "tr": "Turkish",
            "vi": "Vietnamese",
            "th": "Thai",
        }
        
        # Check if using yt-dlp module
        use_module = YTDLP_MODULE_AVAILABLE and ytdlp_path == "yt_dlp_module"
        
        if use_module:
            return DownloadService._get_subtitles_module(url, cookies_path, lang_names)
        else:
            return DownloadService._get_subtitles_subprocess(url, ytdlp_path, cookies_path, lang_names)
    
    @staticmethod
    def _get_subtitles_module(url: str, cookies_path: str, lang_names: dict) -> dict:
        """Get subtitles using yt-dlp Python module API"""
        try:
            # Check if cookies.txt exists
            if not cookies_path or not Path(cookies_path).exists():
                return {
                    "error": "cookies.txt not found. Please upload cookies.txt file.",
                    "subtitles": [],
                    "automatic_captions": []
                }
            
            # Validate cookies file has YouTube auth cookies
            # Check both plain cookies (SID, HSID, etc.) and __Secure- prefixed variants
            # Modern browsers/extensions often export only __Secure- versions
            required_cookies = ['SID', 'HSID', 'SSID', 'APISID', 'SAPISID', 'LOGIN_INFO']
            secure_prefixes = ['__Secure-1P', '__Secure-3P']
            found_cookies = []
            try:
                with open(cookies_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    for cookie in required_cookies:
                        # Check plain cookie name (tab-separated format)
                        if f"\t{cookie}\t" in content or content.endswith(f"\t{cookie}"):
                            found_cookies.append(cookie)
                        else:
                            # Check __Secure- prefixed variants (e.g. __Secure-3PSID)
                            for prefix in secure_prefixes:
                                secure_name = f"{prefix}{cookie}"
                                if f"\t{secure_name}\t" in content or content.endswith(f"\t{secure_name}"):
                                    found_cookies.append(secure_name)
                                    break
                
                if not found_cookies:
                    debug_log(f"Cookies file missing required auth cookies. Found: {found_cookies}")
                    return {
                        "error": "Invalid cookies.txt - missing YouTube authentication cookies.\n\n"
                                 "Please export fresh cookies from your browser while logged into YouTube.\n\n"
                                 "Required cookies: SID, HSID, SSID, APISID, SAPISID, LOGIN_INFO\n\n"
                                 "Use a browser extension like 'Get cookies.txt LOCALLY' to export.",
                        "subtitles": [],
                        "automatic_captions": []
                    }
                debug_log(f"Found auth cookies: {found_cookies}")
            except Exception as e:
                debug_log(f"Error reading cookies file: {e}")
            
            debug_log(f"Using yt-dlp module v{yt_dlp.version.__version__}")
            debug_log(f"Cookies path: {cookies_path} (exists: {Path(cookies_path).exists()})")
            
            # Setup Deno and FFmpeg in PATH
            ensure_binaries_in_path()
            deno_path = get_deno_path()
            ffmpeg_path = get_ffmpeg_path()
            
            # yt-dlp options for fetching info only
            # NOTE: Don't use player_client=android with cookies - it bypasses cookie auth
            ydl_opts = {
                'skip_download': True,
                'quiet': False,  # Show warnings for debugging
                'no_warnings': False,
                'cookiefile': str(cookies_path),  # Ensure string path
            }
            
            # Add Deno JS runtime if available
            if deno_path and Path(deno_path).exists():
                ydl_opts['js_runtimes'] = {'deno': {'path': deno_path}}
                ydl_opts['remote_components'] = ['ejs:github']
                debug_log(f"JS runtime: deno at {deno_path}")
            
            # Add FFmpeg location if available
            if ffmpeg_path and Path(ffmpeg_path).exists():
                ydl_opts['ffmpeg_location'] = str(Path(ffmpeg_path).parent)
                debug_log(f"FFmpeg location: {ydl_opts['ffmpeg_location']}")
            
            debug_log(f"yt-dlp opts: cookiefile={ydl_opts['cookiefile']}")
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                video_data = ydl.extract_info(url, download=False)
            
            if not video_data:
                return {"error": "Failed to fetch video info", "subtitles": [], "automatic_captions": []}
            
            # Extract subtitles (exclude live_chat)
            subtitles = []
            auto_captions = []
            
            # Get manual subtitles
            if "subtitles" in video_data and video_data["subtitles"]:
                for lang_code in video_data["subtitles"].keys():
                    if "live_chat" in lang_code:
                        continue
                    lang_name = lang_names.get(lang_code, lang_code.upper())
                    subtitles.append({"code": lang_code, "name": lang_name})
            
            # Get automatic captions
            if "automatic_captions" in video_data and video_data["automatic_captions"]:
                for lang_code in video_data["automatic_captions"].keys():
                    if "live_chat" in lang_code:
                        continue
                    lang_name = lang_names.get(lang_code, lang_code.upper())
                    auto_captions.append({"code": lang_code, "name": lang_name})
            
            return {
                "subtitles": subtitles,
                "automatic_captions": auto_captions,
                "error": None
            }
            
        except Exception as e:
            debug_log(f"yt-dlp module error: {e}")
            return {"error": str(e), "subtitles": [], "automatic_captions": []}
    
    @staticmethod
    def _get_subtitles_subprocess(url: str, ytdlp_path: str, cookies_path: str, lang_names: dict) -> dict:
        """Get subtitles using yt-dlp subprocess (fallback)"""
        try:
            # Check if cookies.txt exists
            if not cookies_path or not Path(cookies_path).exists():
                return {
                    "error": "cookies.txt not found. Please upload cookies.txt file.",
                    "subtitles": [],
                    "automatic_captions": []
                }
            
            # Setup environment with Deno and FFmpeg in PATH
            paths = ensure_binaries_in_path()
            deno_path = paths["deno_path"]
            if not deno_path:
                debug_log("Deno not found - remote-components may not work")
            
            # Use --dump-json to get structured data
            # NOTE: Don't use player_client=android with cookies - it bypasses cookie auth
            cmd = [ytdlp_path, "--dump-json", "--skip-download", 
                   "--cookies", cookies_path]
            
            # Check for remote-components support (requires Deno)
            try:
                help_result = subprocess.run(
                    [ytdlp_path, "--help"],
                    capture_output=True,
                    text=True,
                    creationflags=SUBPROCESS_FLAGS,
                    timeout=5
                )
                if help_result.returncode == 0:
                    help_text = help_result.stdout
                    
                    # Add remote-components if supported AND Deno is available
                    if "--remote-components" in help_text and deno_path:
                        cmd.extend(["--remote-components", "ejs:github"])
                        debug_log("Added --remote-components ejs:github")
                    
                    # Add no-impersonate if supported
                    if "--no-impersonate" in help_text:
                        cmd.append("--no-impersonate")
                        debug_log("Added --no-impersonate flag")
            except Exception as e:
                debug_log(f"Error checking yt-dlp features: {e}")
            
            # Add URL at the end
            cmd.append(url)
            
            # Log command for debugging
            debug_log(f"Running command: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                creationflags=SUBPROCESS_FLAGS,
                env=os.environ.copy(),  # Use modified environment with Deno path
                timeout=30  # Add timeout to prevent hanging
            )
            
            if result.returncode != 0:
                # Log stderr for debugging
                error_msg = result.stderr.strip() if result.stderr else "Unknown error"
                debug_log(f"yt-dlp stderr: {error_msg}")
                return {"error": f"Failed to fetch video info: {error_msg[:100]}", "subtitles": [], "automatic_captions": []}
            
            # Parse JSON output
            video_data = json.loads(result.stdout)
            
            # Extract subtitles (exclude live_chat)
            subtitles = []
            auto_captions = []
            
            # Get manual subtitles
            if "subtitles" in video_data and video_data["subtitles"]:
                for lang_code in video_data["subtitles"].keys():
                    if "live_chat" in lang_code:
                        continue
                    lang_name = lang_names.get(lang_code, lang_code.upper())
                    subtitles.append({"code": lang_code, "name": lang_name})
            
            # Get automatic captions
            if "automatic_captions" in video_data and video_data["automatic_captions"]:
                for lang_code in video_data["automatic_captions"].keys():
                    if "live_chat" in lang_code:
                        continue
                    lang_name = lang_names.get(lang_code, lang_code.upper())
                    auto_captions.append({"code": lang_code, "name": lang_name})
            
            return {
                "subtitles": subtitles,
                "automatic_captions": auto_captions,
                "error": None
            }
            
        except subprocess.TimeoutExpired:
            return {"error": "Timeout fetching subtitles", "subtitles": [], "automatic_captions": []}
        except json.JSONDecodeError:
            return {"error": "Failed to parse video data", "subtitles": [], "automatic_captions": []}
        except Exception as e:
            return {"error": str(e), "subtitles": [], "automatic_captions": []}
    
    def parse_srt(self, srt_path: str) -> str:
        """Parse SRT to text with timestamps"""
        with open(srt_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        pattern = r"(\d+)\n(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\n(.*?)(?=\n\n|\Z)"
        matches = re.findall(pattern, content, re.DOTALL)
        
        lines = []
        for idx, start, end, text in matches:
            clean_text = text.replace("\n", " ").strip()
            lines.append(f"[{start} - {end}] {clean_text}")
        
        return "\n".join(lines)
    
    def extract_transcript_for_highlight(self, srt_path: str, highlight: dict) -> str:
        """Extract subtitle text within a highlight's time range.
        
        Args:
            srt_path: Path to SRT file
            highlight: Dict with start_time and end_time keys
            
        Returns:
            str: Concatenated subtitle text within the time range
        """
        with open(srt_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        pattern = r"(\d+)\n(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\n(.*?)(?=\n\n|\Z)"
        matches = re.findall(pattern, content, re.DOTALL)
        
        start_sec = parse_timestamp(highlight["start_time"])
        end_sec = parse_timestamp(highlight["end_time"])
        
        lines = []
        for idx, start, end, text in matches:
            sub_start = parse_timestamp(start)
            sub_end = parse_timestamp(end)
            
            # Include subtitle if it overlaps with highlight range
            if sub_end >= start_sec and sub_start <= end_sec:
                clean_text = text.replace("\n", " ").strip()
                if clean_text:
                    lines.append(clean_text)
        
        return " ".join(lines)
    
    def download_subtitle_only(self, url: str) -> tuple:
        """Download only subtitle (no video) using yt-dlp.
        
        Returns:
            tuple: (srt_path, video_info) where srt_path is str or None
        """
        debug_log("[1/2] Downloading subtitle only...")
        
        if self._use_ytdlp_module():
            return self._download_subtitle_only_module(url)
        return self._download_subtitle_only_subprocess(url)
    
    def _download_subtitle_only_module(self, url: str) -> tuple:
        """Download subtitle only using yt-dlp Python module API"""
        debug_log(f"  Using yt-dlp module v{yt_dlp.version.__version__}")
        
        video_info = {}
        
        # Get Deno path
        deno_path = get_deno_path()
        ffmpeg_path = get_ffmpeg_path()
        
        # Setup environment with Deno and FFmpeg in PATH
        ensure_binaries_in_path()
        
        # yt-dlp options: skip video download, only get subtitle
        ydl_opts = {
            'skip_download': True,
            'writesubtitles': True,
            'writeautomaticsub': True,
            'subtitleslangs': [self.subtitle_language],
            'subtitlesformat': 'srt',
            'outtmpl': str(self.temp_dir / 'source.%(ext)s'),
            'quiet': True,
            'no_warnings': False,
        }
        
        # Add Deno JS runtime if available
        self._apply_js_runtime_options(ydl_opts, deno_path)
        
        # Add FFmpeg location for subtitle conversion
        self._apply_ffmpeg_location_options(
            ydl_opts,
            ffmpeg_path,
            include_subtitle_postprocessor=True,
        )
        
        cookies_path = self._require_cookies_path()
        ydl_opts['cookiefile'] = str(cookies_path)
        
        try:
            debug_log(f"  Downloading {self.subtitle_language} subtitle...")
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                # Get video info + download subtitle
                info = ydl.extract_info(url, download=True)
                
                if info:
                    video_info = {
                        "title": info.get("title", ""),
                        "description": (info.get("description", "") or "")[:2000],
                        "channel": info.get("channel", ""),
                    }
                    debug_log(f"  Title: {video_info['title'][:50]}...")
            
            debug_log(f"  ✓ Subtitle download complete!")
            
        except Exception as e:
            last_error = str(e)
            debug_log(f"  ✗ Failed: {last_error[:100]}")
            
            if "403" in last_error or "Forbidden" in last_error:
                raise Exception(
                    "❌ ERROR: YouTube menolak akses (HTTP 403 Forbidden)\n\n"
                    "Cookies sudah EXPIRED. Silakan export cookies baru.\n\n"
                    "📖 Lihat COOKIES.md untuk panduan lengkap"
                )
            else:
                raise Exception(f"Subtitle download failed!\n\n{last_error}")
        
        # Find downloaded subtitle file
        srt_path = self._find_downloaded_srt()
        
        return str(srt_path) if srt_path else None, video_info
    
    def _download_subtitle_only_subprocess(self, url: str) -> tuple:
        """Download subtitle only using yt-dlp subprocess (fallback)"""
        # Validate yt-dlp
        try:
            version_check = subprocess.run(
                [self.ytdlp_path, "--version"],
                capture_output=True, text=True,
                creationflags=SUBPROCESS_FLAGS, timeout=5
            )
            if version_check.returncode != 0:
                raise Exception(f"yt-dlp not working properly. Path: {self.ytdlp_path}")
            debug_log(f"  Using yt-dlp version: {version_check.stdout.strip()}")
        except FileNotFoundError:
            raise Exception(f"yt-dlp not found at: {self.ytdlp_path}")
        
        # Get video metadata first
        debug_log("  Fetching video info...")
        meta_cmd = [self.ytdlp_path, "--dump-json", "--no-download", url]
        
        cookies_path = self._find_cookies_path()
        
        if cookies_path:
            meta_cmd.extend(["--cookies", str(cookies_path)])
        
        result = subprocess.run(
            meta_cmd, capture_output=True, text=True,
            creationflags=SUBPROCESS_FLAGS, timeout=30
        )
        
        video_info = {}
        if result.returncode == 0:
            try:
                yt_data = json.loads(result.stdout)
                video_info = {
                    "title": yt_data.get("title", ""),
                    "description": yt_data.get("description", "")[:2000],
                    "channel": yt_data.get("channel", ""),
                }
                debug_log(f"  Title: {video_info['title'][:50]}...")
            except json.JSONDecodeError:
                debug_log("  Warning: Could not parse metadata")
        
        # Download subtitle only
        debug_log(f"  Downloading {self.subtitle_language} subtitle...")
        cmd = [
            self.ytdlp_path,
            "--skip-download",
            "--write-sub", "--write-auto-sub",
            "--sub-lang", self.subtitle_language,
            "--convert-subs", "srt",
            "-o", str(self.temp_dir / "source.%(ext)s"),
        ]
        
        if cookies_path:
            cmd.extend(["--cookies", str(cookies_path)])
        
        cmd.append(url)
        
        result = subprocess.run(
            cmd, capture_output=True, text=True,
            creationflags=SUBPROCESS_FLAGS, timeout=30
        )
        
        if result.returncode != 0:
            error_msg = result.stderr.strip() if result.stderr else "Unknown error"
            debug_log(f"  ✗ Failed: {error_msg[:100]}")
            raise Exception(f"Subtitle download failed!\n\n{error_msg}")
        
        debug_log(f"  ✓ Subtitle download complete!")
        
        srt_path = self._find_downloaded_srt()
        
        return str(srt_path) if srt_path else None, video_info
    
    def download_video_section(self, url: str, start_time: str, end_time: str, output_path: str) -> str:
        """Download a specific section of a video using yt-dlp --download-sections.
        
        Args:
            url: YouTube video URL
            start_time: Start timestamp (HH:MM:SS,mmm or HH:MM:SS.mmm)
            end_time: End timestamp (HH:MM:SS,mmm or HH:MM:SS.mmm)
            output_path: Path to save the downloaded section
            
        Returns:
            str: Path to downloaded video file
        """
        # Normalize timestamps (replace comma with dot for yt-dlp)
        start_clean = start_time.replace(",", ".")
        end_clean = end_time.replace(",", ".")
        
        if self._use_ytdlp_module():
            return self._download_section_module(url, start_clean, end_clean, output_path)
        return self._download_section_subprocess(url, start_clean, end_clean, output_path)
    
    def _download_section_module(self, url: str, start_time: str, end_time: str, output_path: str) -> str:
        """Download video section using yt-dlp Python module"""
        debug_log(f"  Downloading section {start_time} → {end_time}...")
        
        # Get paths
        ffmpeg_path = get_ffmpeg_path()
        deno_path = get_deno_path()
        
        # Setup Deno and FFmpeg in PATH
        ensure_binaries_in_path()
        
        # Progress hook
        def progress_hook(d):
            if self.is_cancelled():
                raise Exception("Cancelled by user")
            
            if d['status'] == 'downloading':
                percent_str = d.get('_percent_str', '0%').strip()
                match = re.search(r'(\d+\.?\d*)%', percent_str)
                if match:
                    percent = float(match.group(1))
                    self.set_progress(f"Downloading video section... {percent:.1f}%", 0)
            elif d['status'] == 'finished':
                debug_log("  Section download finished, processing...")
        
        # Format selector
        format_selector = self._format_selector()
        
        ydl_opts = {
            'format': format_selector,
            'format_sort': ['res', 'br'],
            'merge_output_format': 'mp4',
            'outtmpl': output_path,
            'progress_hooks': [progress_hook],
            'quiet': True,
            'no_warnings': False,
            'verbose': True,
            'logger': _SectionYtdlpLogger(),
            'download_ranges': yt_dlp.utils.download_range_func(None, [(
                parse_timestamp(start_time),
                parse_timestamp(end_time)
            )]),
            'force_keyframes_at_cuts': True,
        }
        
        self._apply_js_runtime_options(ydl_opts, deno_path)
        
        # Add FFmpeg location and GPU args
        if ffmpeg_path and Path(ffmpeg_path).exists():
            ydl_opts['ffmpeg_location'] = str(Path(ffmpeg_path).parent)
            
            plan = self._get_section_ffmpeg_args(ffmpeg_path)
            if plan["encoder_args"]:
                ydl_opts['external_downloader_args'] = {
                    'ffmpeg_i1': plan["decode_args"],
                    'ffmpeg_o': plan["encoder_args"]
                }
            debug_log("  yt-dlp section params:")
            debug_log(f"    download section: {start_time} -> {end_time}")
            debug_log(f"    ffmpeg_location: {ydl_opts.get('ffmpeg_location')}")
            debug_log(f"    external_downloader_args: {ydl_opts.get('external_downloader_args', 'none')}")
        
        cookies_path = self._require_cookies_path()
        ydl_opts['cookiefile'] = str(cookies_path)
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            
            debug_log(f"  ✓ Section downloaded!")
            
        except Exception as e:
            last_error = str(e)
            debug_log(f"  ✗ Section download failed: {last_error[:100]}")
            raise Exception(f"Failed to download video section!\n\n{last_error}")
        
        # Find the actual output file (yt-dlp may add extension)
        output_dir = Path(output_path).parent
        output_stem = Path(output_path).stem
        
        # Check for exact match first
        if Path(output_path).exists():
            return output_path
        
        # Check for .mp4 variant
        mp4_path = output_dir / f"{output_stem}.mp4"
        if mp4_path.exists():
            return str(mp4_path)
        
        # Search for any file with the stem
        candidates = list(output_dir.glob(f"{output_stem}.*"))
        video_candidates = [c for c in candidates if c.suffix in ('.mp4', '.mkv', '.webm')]
        if video_candidates:
            return str(video_candidates[0])
        
        raise Exception(f"Downloaded section file not found at: {output_path}")
    
    def _download_section_subprocess(self, url: str, start_time: str, end_time: str, output_path: str) -> str:
        """Download video section using yt-dlp subprocess (fallback)"""
        debug_log(f"  Downloading section {start_time} → {end_time}...")
        
        # Build section string for yt-dlp
        section_str = f"*{start_time}-{end_time}"
        
        format_selector = self._format_selector()
        
        cmd = [
            self.ytdlp_path,
            "-f", format_selector,
            "--format-sort", "res,br",
            "--download-sections", section_str,
            "--force-keyframes-at-cuts",
            "--merge-output-format", "mp4"
        ]
        
        # Check for GPU acceleration
        ffmpeg_path = get_ffmpeg_path() or "ffmpeg"
        plan = self._get_section_ffmpeg_args(ffmpeg_path)
        if plan["encoder_args"]:
            decode_str = " ".join(plan["decode_args"])
            encode_str = " ".join(plan["encoder_args"])
            if decode_str:
                cmd.extend(["--downloader-args", f"ffmpeg_i1:{decode_str}"])
            cmd.extend(["--downloader-args", f"ffmpeg_o:{encode_str}"])
            
        cmd.extend(["-o", output_path])
        
        cookies_path = self._find_cookies_path()
        if cookies_path:
            cmd.extend(["--cookies", str(cookies_path)])
        
        cmd.append(url)
        
        debug_log(f"  Full yt-dlp section command: {self._format_cmd(cmd)}")
        
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            creationflags=SUBPROCESS_FLAGS
        )
        
        while True:
            if self.is_cancelled():
                process.terminate()
                process.wait()
                raise Exception("Cancelled by user")
            
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break
            
            line = line.strip()
            if "[download]" in line and "%" in line:
                match = re.search(r'(\d+\.?\d*)%', line)
                if match:
                    percent = match.group(1)
                    self.set_progress(f"Downloading video section... {percent}%", 0)
        
        if process.returncode != 0:
            stderr_output = process.stderr.read() if process.stderr else ""
            debug_log(f"  ✗ Section download failed: {stderr_output[:200]}")
            raise Exception(f"Failed to download video section!\n\n{stderr_output[:200]}")
        
        debug_log(f"  ✓ Section downloaded!")
        
        # Find the actual output file
        output_dir = Path(output_path).parent
        output_stem = Path(output_path).stem
        
        if Path(output_path).exists():
            return output_path
        
        mp4_path = output_dir / f"{output_stem}.mp4"
        if mp4_path.exists():
            return str(mp4_path)
        
        candidates = list(output_dir.glob(f"{output_stem}.*"))
        video_candidates = [c for c in candidates if c.suffix in ('.mp4', '.mkv', '.webm')]
        if video_candidates:
            return str(video_candidates[0])
        
        raise Exception(f"Downloaded section file not found at: {output_path}")
    
