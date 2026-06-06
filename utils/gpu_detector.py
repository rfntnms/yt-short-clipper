"""GPU Detection and FFmpeg Hardware Acceleration Support.

v2 simplified: focuses on CUDA detection and FFmpeg encoder availability.
Fallback to CPU if detection fails.
"""

import subprocess
import shutil
from utils.logger import logger

def detect_cuda() -> dict:
    """Detect NVIDIA CUDA GPU via nvidia-smi.

    Returns:
        dict: {
            'available': bool,
            'name': str or None,
            'h264_nvenc_available': bool
        }
    """
    gpu_info = {
        'available': False,
        'name': None,
        'h264_nvenc_available': False
    }

    # 1. Check nvidia-smi
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0 and result.stdout.strip():
            gpu_name = result.stdout.strip().split('\n')[0]
            gpu_info['available'] = True
            gpu_info['name'] = gpu_name
            logger.info(f"CUDA GPU detected: {gpu_name}")
        else:
            logger.info("nvidia-smi command executed but no GPU returned.")
    except FileNotFoundError:
        logger.info("nvidia-smi not found, falling back to CPU.")
    except subprocess.TimeoutExpired:
        logger.warning("nvidia-smi command timed out.")
    except Exception as e:
        logger.warning(f"Unexpected error detecting NVIDIA GPU: {e}")

    # 2. If GPU available, check if h264_nvenc encoder exists in FFmpeg
    if gpu_info['available']:
        ffmpeg_path = shutil.which('ffmpeg')
        if not ffmpeg_path:
            logger.warning("FFmpeg not found in PATH, cannot check for nvenc support.")
            return gpu_info
            
        try:
            result = subprocess.run(
                [ffmpeg_path, '-encoders'],
                capture_output=True,
                text=True,
                timeout=10
            )
            # FFmpeg encoders output goes to stderr or stdout depending on version
            output = result.stdout + result.stderr
            if 'h264_nvenc' in output:
                gpu_info['h264_nvenc_available'] = True
                logger.info("h264_nvenc encoder is available in FFmpeg.")
            else:
                logger.warning("CUDA GPU detected, but h264_nvenc not found in FFmpeg encoders.")
        except (FileNotFoundError, subprocess.TimeoutExpired) as e:
            logger.warning(f"Failed to check FFmpeg encoders: {e}")

    return gpu_info


def get_gpu_flags(gpu_info: dict) -> dict:
    """Convert GPU detection info into FFmpeg command-line flags.

    Args:
        gpu_info: Output from detect_cuda()

    Returns:
        dict: {
            'encoder': str,           # 'h264_nvenc' or 'libx264'
            'hwaccel': list[str],     # ['-hwaccel', 'cuda'] or []
            'description': str        # Human-readable flag info
        }
    """
    if gpu_info.get('available') and gpu_info.get('h264_nvenc_available'):
        return {
            'encoder': 'h264_nvenc',
            'hwaccel': [],
            'description': f"Using GPU: {gpu_info.get('name')} (NVENC encoding only, CPU filters)",
        }
    elif gpu_info.get('available') and not gpu_info.get('h264_nvenc_available'):
        return {
            'encoder': 'libx264',
            'hwaccel': [],
            'description': f"GPU detected ({gpu_info.get('name')}) but nvenc encoder missing, falling back to CPU",
        }
    else:
        return {
            'encoder': 'libx264',
            'hwaccel': [],
            'description': "No GPU detected, using CPU",
        }

__all__ = ['detect_cuda', 'get_gpu_flags']
