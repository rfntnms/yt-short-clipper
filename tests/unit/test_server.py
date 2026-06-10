import pytest
import uuid
from unittest.mock import patch, MagicMock
from server import _get_config, _tail_log, _ui_to_config, _config_to_ui, _batch_add

def test_tail_log_handles_missing_file(tmp_path):
    with patch("server.LOG_PATH", tmp_path / "missing.log"):
        assert "no log yet" in _tail_log()

def test_config_roundtrip():
    ui_vals = {
        "llm_base_url": "http://test",
        "llm_model": "test-model",
        "llm_api_key": "test-key",
        "tr_base_url": "http://test-tr",
        "tr_model": "test-tr-model",
        "tr_api_key": "test-tr-key",
        "face_backend": "dlib",
        "split_enabled": False,
        "caption_font": "Roboto",
        "caption_fontsize": 48,
        "caption_outline": 2,
    }
    cfg = _ui_to_config(ui_vals)
    
    assert cfg["llm"]["base_url"] == "http://test"
    assert cfg["llm"]["api_key"] == "test-key"
    assert cfg["transcription"]["model"] == "test-tr-model"
    assert cfg["portrait"]["face_backend"] == "dlib"
    assert cfg["caption"]["fontsize"] == 48
    
    ui_vals2 = _config_to_ui(cfg)
    assert ui_vals2["llm_base_url"] == "http://test"
    assert ui_vals2["caption_fontsize"] == 48

@patch("server._job_queue")
def test_batch_add(mock_queue):
    urls = "https://youtube.com/watch?v=123\n\nhttps://youtube.com/watch?v=456"
    mock_cfg = {"llm": {"api_key": "x"}}
    
    gen = _batch_add(urls, mock_cfg)
    result = next(gen)
    
    assert "2 job(s) added" in result[0]
    assert mock_queue.put.call_count == 2
