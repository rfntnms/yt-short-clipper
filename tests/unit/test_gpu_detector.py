import subprocess
import pytest

from utils.gpu_detector import detect_cuda, get_gpu_flags, get_ffmpeg_hwaccel_flags


def test_detect_cuda_returns_dict():
    """detect_cuda should return a dict with expected keys."""
    result = detect_cuda()
    assert isinstance(result, dict)
    assert 'available' in result
    assert 'name' in result
    assert 'h264_nvenc_available' in result
    assert isinstance(result['available'], bool)
    assert isinstance(result['h264_nvenc_available'], bool)


def test_detect_cuda_handles_nvidia_smi_missing(monkeypatch):
    """detect_cuda should gracefully return available=False if nvidia-smi not found."""
    def mock_run(*args, **kwargs):
        raise FileNotFoundError("nvidia-smi not found")
    
    monkeypatch.setattr(subprocess, "run", mock_run)
    
    result = detect_cuda()
    assert result['available'] is False
    assert result['name'] is None
    assert result['h264_nvenc_available'] is False


def test_detect_cuda_handles_nvidia_smi_timeout(monkeypatch):
    """detect_cuda should gracefully handle nvidia-smi timeout."""
    def mock_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="nvidia-smi", timeout=5)
    
    monkeypatch.setattr(subprocess, "run", mock_run)
    
    result = detect_cuda()
    assert result['available'] is False
    assert result['name'] is None


def test_detect_cuda_handles_nvidia_smi_failure(monkeypatch):
    """detect_cuda should handle nvidia-smi returning non-zero exit."""
    class MockResult:
        returncode = 1
        stdout = ""
        stderr = ""
    
    def mock_run(*args, **kwargs):
        return MockResult()
    
    monkeypatch.setattr(subprocess, "run", mock_run)
    
    result = detect_cuda()
    assert result['available'] is False


def test_detect_cuda_succeeds_with_valid_nvidia_smi(monkeypatch):
    """detect_cuda should detect GPU when nvidia-smi returns valid output."""
    class MockResult:
        returncode = 0
        stdout = "NVIDIA GeForce RTX 3090\n"
        stderr = ""
    
    def mock_run(*args, **kwargs):
        return MockResult()
    
    monkeypatch.setattr(subprocess, "run", mock_run)
    
    result = detect_cuda()
    assert result['available'] is True
    assert "RTX 3090" in result['name']
    # nvenc check will fail here (no ffmpeg mock), so h264_nvenc_available stays False
    assert result['h264_nvenc_available'] is False


# ── get_gpu_flags ──────────────────────────────────────────────────────────


def test_get_gpu_flags_returns_correct_shape():
    """get_gpu_flags should return dict with encoder, hwaccel, description."""
    flags = get_gpu_flags({'available': False, 'name': None, 'h264_nvenc_available': False})
    assert 'encoder' in flags
    assert 'hwaccel' in flags
    assert 'description' in flags
    assert flags['encoder'] in ('h264_nvenc', 'libx264')


def test_get_gpu_flags_falls_back_to_cpu():
    """When no GPU, get_gpu_flags should return libx264 with no hwaccel."""
    flags = get_gpu_flags({
        'available': False, 
        'name': None, 
        'h264_nvenc_available': False
    })
    assert flags['encoder'] == 'libx264'
    assert flags['hwaccel'] == []
    assert 'CPU' in flags['description']


def test_get_gpu_flags_uses_nvenc_and_cuda_hwaccel():
    """When CUDA + nvenc available, should return h264_nvenc with CUDA hwaccel."""
    flags = get_gpu_flags({
        'available': True, 
        'name': 'RTX 3090', 
        'h264_nvenc_available': True
    })
    assert flags["encoder"] == "h264_nvenc"
    assert flags["hwaccel"] == ['-hwaccel', 'cuda', '-hwaccel_output_format', 'cuda']
    assert 'CUDA decode + NVENC encode' in flags["description"]


def test_get_gpu_flags_falls_back_when_nvenc_missing():
    """When GPU detected but nvenc not in FFmpeg, fall back to CPU."""
    flags = get_gpu_flags({
        'available': True, 
        'name': 'RTX 3090', 
        'h264_nvenc_available': False
    })
    assert flags['encoder'] == 'libx264'
    assert flags['hwaccel'] == []
    assert 'falling back to CPU' in flags['description']


# ── get_ffmpeg_hwaccel_flags ──────────────────────────────────────────────


def test_get_ffmpeg_hwaccel_flags_returns_correct_keys():
    """get_ffmpeg_hwaccel_flags should return dict with vcodec and preset."""
    flags = get_ffmpeg_hwaccel_flags()
    assert 'vcodec' in flags
    assert 'preset' in flags
    assert flags['vcodec'] in ('h264_nvenc', 'libx264')
    assert flags['preset'] in ('fast', 'medium')


def test_get_ffmpeg_hwaccel_flags_returns_cpu_defaults(monkeypatch):
    """When CUDA unavailable, should return libx264 with medium preset."""
    def mock_no_gpu(*args, **kwargs):
        raise FileNotFoundError("nvidia-smi not found")

    monkeypatch.setattr(subprocess, "run", mock_no_gpu)

    flags = get_ffmpeg_hwaccel_flags()
    assert flags['vcodec'] == 'libx264'
    assert flags['preset'] == 'medium'


def test_get_ffmpeg_hwaccel_flags_uses_nvenc_when_gpu_available(monkeypatch):
    """When CUDA + nvenc available, should return fast preset."""
    nvidia_call_count = 0

    class _MockNvencResult:
        returncode = 0
        stdout = " h264_nvenc "
        stderr = ""

    def mock_run_nvidia(*args, **kwargs):
        nonlocal nvidia_call_count
        nvidia_call_count += 1
        cmd = args[0] if args else kwargs.get('args', [])
        if 'nvidia-smi' in str(cmd):
            class _MockNvidiaResult:
                returncode = 0
                stdout = "RTX 3090\n"
                stderr = ""
            return _MockNvidiaResult()
        return _MockNvencResult()

    monkeypatch.setattr(subprocess, "run", mock_run_nvidia)

    flags = get_ffmpeg_hwaccel_flags()
    assert flags['vcodec'] == 'h264_nvenc'
    assert flags['preset'] == 'fast'


# ── Regression: detect_cuda nvenc check with ffmpeg mock ──────────────────


def test_detect_cuda_checks_nvenc_after_gpu_detected(monkeypatch):
    """detect_cuda should probe ffmpeg encoders when GPU is found."""
    calls = []

    class _MockNvencDetectionResult:
        returncode = 0
        stdout = " h264_nvenc "
        stderr = ""

    def mock_run(*args, **kwargs):
        cmd = args[0] if args else kwargs.get('args', [])
        calls.append(cmd)
        if 'nvidia-smi' in str(cmd):
            class _MockNvidiaDetectionResult:
                returncode = 0
                stdout = "RTX 4090\n"
                stderr = ""
            return _MockNvidiaDetectionResult()
        return _MockNvencDetectionResult()

    monkeypatch.setattr(subprocess, "run", mock_run)

    result = detect_cuda()
    assert result['available'] is True
    assert result['h264_nvenc_available'] is True
    # Should have called nvidia-smi AND ffmpeg
    assert len(calls) == 2
