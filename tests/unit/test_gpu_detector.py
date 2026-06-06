import subprocess
import pytest

from utils.gpu_detector import detect_cuda, get_gpu_flags


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


def test_get_gpu_flags_uses_nvenc_when_available():
    """When CUDA + nvenc available, get_gpu_flags should return h264_nvenc."""
    flags = get_gpu_flags({
        'available': True, 
        'name': 'RTX 3090', 
        'h264_nvenc_available': True
    })
    assert flags['encoder'] == 'h264_nvenc'
    assert flags['hwaccel'] == ['-hwaccel', 'cuda']
    assert 'RTX 3090' in flags['description']


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
