"""
GPU Detection and FFmpeg Hardware Acceleration Support
"""

import subprocess
import re
import sys
from pathlib import Path


# Hide console window on Windows
SUBPROCESS_FLAGS = 0
if sys.platform == "win32":
    SUBPROCESS_FLAGS = subprocess.CREATE_NO_WINDOW


class GPUDetector:
    """Detect available GPU and FFmpeg hardware encoder support"""
    
    def __init__(self, ffmpeg_path: str = "ffmpeg"):
        self.ffmpeg_path = ffmpeg_path
        self._gpu_info = None
        self._ffmpeg_encoders = None
        self._tested_encoders = {}
    
    def detect_gpu(self) -> dict:
        """
        Detect available GPU hardware
        
        Returns:
            dict with keys:
                - 'type': 'nvidia', 'amd', 'intel', or None
                - 'name': GPU name string
                - 'available': bool
        """
        if self._gpu_info is not None:
            return self._gpu_info
        
        gpu_info = {
            'type': None,
            'name': 'No GPU detected',
            'available': False
        }
        
        # Try NVIDIA first (most common for encoding)
        nvidia = self._detect_nvidia()
        if nvidia['available']:
            self._gpu_info = nvidia
            return nvidia
        
        # Try AMD
        amd = self._detect_amd()
        if amd['available']:
            self._gpu_info = amd
            return amd
        
        # Try Intel
        intel = self._detect_intel()
        if intel['available']:
            self._gpu_info = intel
            return intel
        
        # macOS: Try Apple GPU (Apple Silicon or integrated)
        if sys.platform == "darwin":
            apple = self._detect_apple()
            if apple['available']:
                self._gpu_info = apple
                return apple
        
        self._gpu_info = gpu_info
        return gpu_info
    
    def _detect_nvidia(self) -> dict:
        """Detect NVIDIA GPU using nvidia-smi"""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                capture_output=True,
                text=True,
                timeout=5,
                creationflags=SUBPROCESS_FLAGS
            )
            
            if result.returncode == 0 and result.stdout.strip():
                gpu_name = result.stdout.strip().split('\n')[0]
                return {
                    'type': 'nvidia',
                    'name': gpu_name,
                    'available': True
                }
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        
        # Fallback: Try wmic on Windows
        if sys.platform == "win32":
            try:
                result = subprocess.run(
                    ['wmic', 'path', 'win32_VideoController', 'get', 'name'],
                    capture_output=True,
                    text=True,
                    timeout=5,
                    creationflags=SUBPROCESS_FLAGS
                )
                
                if result.returncode == 0:
                    lines = result.stdout.strip().split('\n')
                    for line in lines[1:]:  # Skip header
                        line = line.strip()
                        if 'NVIDIA' in line or 'GeForce' in line or 'Quadro' in line or 'RTX' in line or 'GTX' in line:
                            return {
                                'type': 'nvidia',
                                'name': line,
                                'available': True
                            }
            except (FileNotFoundError, subprocess.TimeoutExpired):
                pass
        
        return {'type': None, 'name': '', 'available': False}
    
    def _detect_amd(self) -> dict:
        """Detect AMD GPU"""
        # Windows: Use PowerShell (wmic deprecated in Windows 11)
        if sys.platform == "win32":
            try:
                # Try PowerShell Get-WmiObject first
                result = subprocess.run(
                    ['powershell', '-Command', 
                     'Get-WmiObject Win32_VideoController | Select-Object -ExpandProperty Name'],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    creationflags=SUBPROCESS_FLAGS
                )
                
                if result.returncode == 0:
                    lines = result.stdout.strip().split('\n')
                    for line in lines:
                        line = line.strip()
                        if 'AMD' in line or 'Radeon' in line:
                            return {
                                'type': 'amd',
                                'name': line,
                                'available': True
                            }
            except (FileNotFoundError, subprocess.TimeoutExpired):
                pass
            
            # Fallback: Try wmic (older Windows)
            try:
                result = subprocess.run(
                    ['wmic', 'path', 'win32_VideoController', 'get', 'name'],
                    capture_output=True,
                    text=True,
                    timeout=5,
                    creationflags=SUBPROCESS_FLAGS
                )
                
                if result.returncode == 0:
                    lines = result.stdout.strip().split('\n')
                    for line in lines[1:]:  # Skip header
                        line = line.strip()
                        if 'AMD' in line or 'Radeon' in line:
                            return {
                                'type': 'amd',
                                'name': line,
                                'available': True
                            }
            except (FileNotFoundError, subprocess.TimeoutExpired):
                pass
        
        # Linux: Try lspci
        elif sys.platform.startswith('linux'):
            try:
                result = subprocess.run(
                    ['lspci'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                
                if result.returncode == 0:
                    for line in result.stdout.split('\n'):
                        if 'VGA' in line and ('AMD' in line or 'Radeon' in line):
                            # Extract GPU name
                            match = re.search(r':\s*(.+)$', line)
                            if match:
                                return {
                                    'type': 'amd',
                                    'name': match.group(1).strip(),
                                    'available': True
                                }
            except (FileNotFoundError, subprocess.TimeoutExpired):
                pass
        
        # macOS: Try system_profiler
        elif sys.platform == "darwin":
            try:
                result = subprocess.run(
                    ['system_profiler', 'SPDisplaysDataType'],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                
                if result.returncode == 0:
                    for line in result.stdout.split('\n'):
                        line = line.strip()
                        if ('AMD' in line or 'Radeon' in line) and ':' in line:
                            name = line.split(':', 1)[-1].strip() if ':' in line else line
                            return {
                                'type': 'amd',
                                'name': name,
                                'available': True
                            }
            except (FileNotFoundError, subprocess.TimeoutExpired):
                pass
        
        return {'type': None, 'name': '', 'available': False}
    
    def _detect_intel(self) -> dict:
        """Detect Intel GPU"""
        # Windows: Use PowerShell (wmic deprecated in Windows 11)
        if sys.platform == "win32":
            try:
                # Try PowerShell Get-WmiObject first
                result = subprocess.run(
                    ['powershell', '-Command', 
                     'Get-WmiObject Win32_VideoController | Select-Object -ExpandProperty Name'],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    creationflags=SUBPROCESS_FLAGS
                )
                
                if result.returncode == 0:
                    lines = result.stdout.strip().split('\n')
                    for line in lines:
                        line = line.strip()
                        if 'Intel' in line and ('HD' in line or 'UHD' in line or 'Iris' in line or 'Arc' in line):
                            return {
                                'type': 'intel',
                                'name': line,
                                'available': True
                            }
            except (FileNotFoundError, subprocess.TimeoutExpired):
                pass
            
            # Fallback: Try wmic (older Windows)
            try:
                result = subprocess.run(
                    ['wmic', 'path', 'win32_VideoController', 'get', 'name'],
                    capture_output=True,
                    text=True,
                    timeout=5,
                    creationflags=SUBPROCESS_FLAGS
                )
                
                if result.returncode == 0:
                    lines = result.stdout.strip().split('\n')
                    for line in lines[1:]:  # Skip header
                        line = line.strip()
                        if 'Intel' in line and ('HD' in line or 'UHD' in line or 'Iris' in line or 'Arc' in line):
                            return {
                                'type': 'intel',
                                'name': line,
                                'available': True
                            }
            except (FileNotFoundError, subprocess.TimeoutExpired):
                pass
        
        # Linux: Try lspci
        elif sys.platform.startswith('linux'):
            try:
                result = subprocess.run(
                    ['lspci'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                
                if result.returncode == 0:
                    for line in result.stdout.split('\n'):
                        if 'VGA' in line and 'Intel' in line:
                            match = re.search(r':\s*(.+)$', line)
                            if match:
                                return {
                                    'type': 'intel',
                                    'name': match.group(1).strip(),
                                    'available': True
                                }
            except (FileNotFoundError, subprocess.TimeoutExpired):
                pass
        
        # macOS: Try system_profiler
        elif sys.platform == "darwin":
            try:
                result = subprocess.run(
                    ['system_profiler', 'SPDisplaysDataType'],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                
                if result.returncode == 0:
                    for line in result.stdout.split('\n'):
                        line = line.strip()
                        if 'Intel' in line and ('HD' in line or 'UHD' in line or 'Iris' in line):
                            name = line.split(':', 1)[-1].strip() if ':' in line else line
                            return {
                                'type': 'intel',
                                'name': name,
                                'available': True
                            }
            except (FileNotFoundError, subprocess.TimeoutExpired):
                pass
        
        return {'type': None, 'name': '', 'available': False}
    
    def _detect_apple(self) -> dict:
        """Detect Apple GPU (Apple Silicon or integrated) on macOS"""
        try:
            result = subprocess.run(
                ['system_profiler', 'SPDisplaysDataType'],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                output = result.stdout
                # Look for Apple Silicon GPU (M1, M2, M3, M4, etc.)
                for line in output.split('\n'):
                    line = line.strip()
                    if 'Chipset Model' in line or 'Chip' in line:
                        name = line.split(':', 1)[-1].strip()
                        if 'Apple' in name or name.startswith('M'):
                            return {
                                'type': 'apple',
                                'name': name,
                                'available': True
                            }
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        
        return {'type': None, 'name': '', 'available': False}
    
    def get_available_encoders(self) -> list:
        """
        Get list of available hardware encoders in FFmpeg
        
        Returns:
            list of encoder names (e.g., ['h264_nvenc', 'hevc_nvenc'])
        """
        if self._ffmpeg_encoders is not None:
            return self._ffmpeg_encoders
        
        try:
            result = subprocess.run(
                [self.ffmpeg_path, '-encoders'],
                capture_output=True,
                text=True,
                timeout=10,
                creationflags=SUBPROCESS_FLAGS
            )
            
            if result.returncode == 0 or result.stderr:
                # FFmpeg outputs to stderr, not stdout
                output = result.stdout + result.stderr
                encoders = []
                
                # Parse encoder list - look for hardware encoders
                for line in output.split('\n'):
                    line = line.strip()
                    # Look for hardware encoder lines
                    hardware_encoders = [
                        'h264_nvenc', 'h264_amf', 'h264_qsv', 'h264_mf', 'h264_videotoolbox',
                        'hevc_nvenc', 'hevc_amf', 'hevc_qsv', 'hevc_mf', 'hevc_videotoolbox'
                    ]
                    if any(enc in line for enc in hardware_encoders):
                        # Extract encoder name (format: " V....D h264_nvenc ...")
                        parts = line.split()
                        if len(parts) >= 2:
                            encoder_name = parts[1]
                            if encoder_name.startswith('h264_') or encoder_name.startswith('hevc_'):
                                encoders.append(encoder_name)
                
                self._ffmpeg_encoders = encoders
                return encoders
        
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        
        self._ffmpeg_encoders = []
        return []

    def get_decode_args(self, use_gpu: bool = True) -> list:
        """
        Get FFmpeg hardware decode arguments based on the detected GPU.
        
        Args:
            use_gpu: If False, returns empty list.
            
        Returns:
            list of FFmpeg arguments (e.g., ['-hwaccel', 'cuda'])
        """
        if not use_gpu:
            return []
            
        gpu = self.detect_gpu()
        gpu_type = gpu.get('type')
        
        if gpu_type == 'nvidia':
            return ['-hwaccel', 'cuda']
        elif gpu_type == 'intel':
            return ['-hwaccel', 'qsv']
        elif gpu_type == 'amd':
            if sys.platform == "win32":
                return ['-hwaccel', 'd3d11va']
            return ['-hwaccel', 'auto']
        elif gpu_type == 'apple':
            return ['-hwaccel', 'videotoolbox']
            
        return ['-hwaccel', 'auto']
    
    def get_recommended_encoder(self, preferred_codec: str = "h264") -> dict:
        """
        Get recommended encoder based on detected GPU
        
        Returns:
            dict with keys:
                - 'encoder': encoder name (e.g., 'h264_nvenc') or None
                - 'preset': preset value (e.g., 'p4')
                - 'available': bool
                - 'reason': explanation string
        """
        gpu = self.detect_gpu()
        encoders = self.get_available_encoders()
        
        if not gpu['available']:
            return {
                'encoder': None,
                'preset': None,
                'available': False,
                'reason': 'No GPU detected'
            }
        
        codec = preferred_codec if preferred_codec in ("h264", "hevc") else "h264"

        # Map GPU type to encoder. HEVC is optional and only selected when
        # explicitly requested because H.264 is the safest shorts/export default.
        encoder_map = {
            'h264': {
                'nvidia': 'h264_nvenc',
                'amd': 'h264_amf',
                'intel': 'h264_qsv',
                'apple': 'h264_videotoolbox'
            },
            'hevc': {
                'nvidia': 'hevc_nvenc',
                'amd': 'hevc_amf',
                'intel': 'hevc_qsv',
                'apple': 'hevc_videotoolbox'
            }
        }
        
        preset_map = {
            'nvidia': 'p4',       # p1-p7, p4 is balanced
            'amd': 'balanced',    # h264_amf -quality: speed | balanced | quality
            'intel': 'faster',    # h264_qsv -preset: veryfast..veryslow (NOT 'balanced')
            'apple': None         # VideoToolbox doesn't use presets
        }
        
        recommended_encoder = encoder_map[codec].get(gpu['type'])
        
        if recommended_encoder in encoders:
            return {
                'encoder': recommended_encoder,
                'preset': preset_map.get(gpu['type']),
                'available': True,
                'reason': f"Using {gpu['name']}"
            }
        else:
            return {
                'encoder': None,
                'preset': None,
                'available': False,
                'reason': f"GPU detected ({gpu['name']}) but FFmpeg doesn't support {recommended_encoder}"
            }
    
    def test_encoder(self, encoder: str, timeout: int = 10) -> dict:
        """Run a tiny synthetic encode to verify an FFmpeg encoder works."""
        if not encoder:
            return {'available': False, 'reason': 'No encoder specified'}

        if encoder in self._tested_encoders:
            return self._tested_encoders[encoder]

        try:
            cmd = [
                self.ffmpeg_path, '-hide_banner', '-y',
                '-f', 'lavfi',
                '-i', 'testsrc2=size=256x256:rate=1:duration=1',
                '-frames:v', '1',
                '-c:v', encoder,
                '-f', 'null',
                '-'
            ]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                creationflags=SUBPROCESS_FLAGS
            )
            if result.returncode == 0:
                tested = {'available': True, 'reason': 'Encoder test passed'}
            else:
                stderr = (result.stderr or '').strip()
                reason = next(
                    (line.strip() for line in stderr.splitlines()
                     if 'error' in line.lower() or 'failed' in line.lower()),
                    stderr.splitlines()[-1].strip() if stderr else 'Encoder test failed'
                )
                tested = {'available': False, 'reason': reason[:200]}
        except Exception as e:
            tested = {'available': False, 'reason': str(e)[:200]}

        self._tested_encoders[encoder] = tested
        return tested

    def get_encoder_args(self, use_gpu: bool = True, preferred_codec: str = "h264", encoder: str = "auto") -> list:
        """
        Get FFmpeg encoder arguments
        
        Args:
            use_gpu: Whether to use GPU acceleration
        
        Returns:
            list of FFmpeg arguments for video encoding
        """
        if not use_gpu:
            # CPU encoding (default)
            if preferred_codec == "hevc":
                return ['-c:v', 'libx265', '-preset', 'fast', '-crf', '22']
            return ['-c:v', 'libx264', '-preset', 'fast', '-crf', '18']
        
        recommendation = self.get_recommended_encoder(preferred_codec=preferred_codec)
        
        if not recommendation['available']:
            # Fallback to CPU
            return self.get_encoder_args(use_gpu=False, preferred_codec=preferred_codec)
        
        encoder = encoder if encoder and encoder != "auto" else recommendation['encoder']
        preset = recommendation['preset']
        
        # Build encoder-specific arguments
        if encoder == 'h264_nvenc':
            # NVIDIA NVENC
            # -pix_fmt yuv420p required for compatibility with various source formats
            return [
                '-c:v', 'h264_nvenc',
                '-preset', preset,
                '-rc', 'vbr',
                '-cq', '19',  # Similar quality to CRF 18
                '-b:v', '0',  # Variable bitrate
                '-pix_fmt', 'yuv420p'
            ]
        
        elif encoder == 'h264_amf':
            # AMD AMF
            # -pix_fmt yuv420p required for compatibility with various source formats
            return [
                '-c:v', 'h264_amf',
                '-quality', preset,
                '-rc', 'vbr_latency',
                '-qp_i', '18',
                '-qp_p', '19',
                '-pix_fmt', 'yuv420p'
            ]
        
        elif encoder == 'h264_qsv':
            # Intel QSV
            # -pix_fmt yuv420p required for compatibility with various source formats
            return [
                '-c:v', 'h264_qsv',
                '-preset', preset,
                '-global_quality', '19',
                '-pix_fmt', 'yuv420p'
            ]
        
        elif encoder == 'h264_videotoolbox':
            # Apple VideoToolbox (macOS)
            return [
                '-c:v', 'h264_videotoolbox',
                '-q:v', '65',  # Quality 1-100, 65 is good balance
                '-pix_fmt', 'yuv420p'
            ]
        elif encoder == 'hevc_nvenc':
            return [
                '-c:v', 'hevc_nvenc',
                '-preset', preset,
                '-rc', 'vbr',
                '-cq', '23',
                '-b:v', '0',
                '-pix_fmt', 'yuv420p'
            ]
        elif encoder == 'hevc_amf':
            return [
                '-c:v', 'hevc_amf',
                '-quality', preset,
                '-rc', 'vbr_latency',
                '-qp_i', '22',
                '-qp_p', '23',
                '-pix_fmt', 'yuv420p'
            ]
        elif encoder == 'hevc_qsv':
            return [
                '-c:v', 'hevc_qsv',
                '-preset', preset,
                '-global_quality', '23',
                '-pix_fmt', 'yuv420p'
            ]
        elif encoder == 'hevc_videotoolbox':
            return [
                '-c:v', 'hevc_videotoolbox',
                '-q:v', '60',
                '-pix_fmt', 'yuv420p'
            ]
        
        # Fallback to CPU
        return self.get_encoder_args(use_gpu=False, preferred_codec=preferred_codec)
