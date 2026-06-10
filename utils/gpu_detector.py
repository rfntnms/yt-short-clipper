"""GPU Detection and FFmpeg Hardware Acceleration Support.

v2 simplified: focuses on CUDA detection and FFmpeg encoder availability.
Fallback to CPU if detection fails.

Usage:
    from utils.gpu_detector import detect_cuda, get_gpu_flags

    gpu = detect_cuda()
    flags = get_gpu_flags(gpu)
    # flags['hwaccel'] → ['-hwaccel', 'cuda', '-hwaccel_output_format', 'cuda'] or []
    # flags['encoder'] → 'h264_nvenc' or 'libx264'

    # Or use the convenience wrapper for caption burn-in:
    from utils.gpu_detector import get_ffmpeg_hwaccel_flags
    hf = get_ffmpeg_hwaccel_flags()
    # hf['vcodec'], hf['preset']
"""

from __future__ import annotations

import subprocess
import shutil
from typing import Any

from utils.logger import logger


def detect_cuda() -> dict[str, Any]:
    """Detect NVIDIA CUDA GPU via nvidia-smi.

    Returns:
        dict: {
            'available': bool,
            'name': str or None,
            'h264_nvenc_available': bool
        }
    """
    gpu_info: dict[str, Any] = {
        'available': False,
        'name': None,
        'h264_nvenc_available': False,
    }

    # 1. Check nvidia-smi
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
            capture_output=True,
            text=True,
            timeout=5,
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
        if not _ffmpeg_has_nvenc():
            logger.warning("CUDA GPU detected, but h264_nvenc not found in FFmpeg encoders.")
        else:
            gpu_info['h264_nvenc_available'] = True
            logger.info("h264_nvenc encoder is available in FFmpeg.")

    return gpu_info


def _ffmpeg_has_nvenc() -> bool:
    """Check if FFmpeg has h264_nvenc encoder compiled in."""
    ffmpeg_path = shutil.which('ffmpeg')
    if not ffmpeg_path:
        logger.warning("FFmpeg not found in PATH, cannot check for nvenc support.")
        return False

    try:
        result = subprocess.run(
            [ffmpeg_path, '-encoders'],
            capture_output=True,
            text=True,
            timeout=10,
        )
        output = result.stdout + result.stderr
        return 'h264_nvenc' in output
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        logger.warning(f"Failed to check FFmpeg encoders: {e}")
        return False


def get_gpu_flags(gpu_info: dict[str, Any]) -> dict[str, Any]:
    """Convert GPU detection info into FFmpeg command-line flags.

    When CUDA + nvenc are both available, returns hardware decode flags
    so frames stay on the GPU through decode → filter → encode:

        -hwaccel cuda -hwaccel_output_format cuda

    When GPU is unavailable or nvenc is missing, returns CPU defaults.

    Args:
        gpu_info: Output from detect_cuda().

    Returns:
        dict: {
            'encoder': str,           # 'h264_nvenc' or 'libx264'
            'hwaccel': list[str],     # ['-hwaccel', 'cuda', ...] or []
            'description': str        # Human-readable flag info
        }
    """
    if gpu_info.get('available') and gpu_info.get('h264_nvenc_available'):
        return {
            'encoder': 'h264_nvenc',
            'hwaccel': ['-hwaccel', 'cuda', '-hwaccel_output_format', 'cuda'],
            'description': f"Using GPU: {gpu_info.get('name')} (CUDA decode + NVENC encode)",
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


def get_ffmpeg_hwaccel_flags() -> dict[str, str]:
    """Convenience wrapper for quick FFmpeg flag access.

    Designed for modules (like caption_generator) that just need the
    encoder name and preset without handling raw GPU info dicts.

    Returns:
        dict: {
            'vcodec': str,      # 'h264_nvenc' or 'libx264'
            'preset': str       # 'fast' for GPU, 'medium' for CPU
        }
    """
    gpu = detect_cuda()
    flags = get_gpu_flags(gpu)

    is_gpu = flags['encoder'] == 'h264_nvenc'
    return {
        'vcodec': flags['encoder'],
        'preset': 'fast' if is_gpu else 'medium',
    }


__all__ = [
    'detect_cuda',
    'get_gpu_flags',
    'get_ffmpeg_hwaccel_flags',
]
