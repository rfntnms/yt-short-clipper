# 🧪 TEST_PLAN.md — Strategi Testing YT-Short-Clipper v2

> **Prinsip:** Setiap modul `pipeline/` dan `providers/` harus bisa di-test secara independen.
> Gunakan mock untuk semua external dependency (API, ffmpeg subprocess, file I/O besar).

---

## 📁 Struktur Direktori Test

```
tests/
├── conftest.py                  # Shared fixtures (sample SRT, mock config, mock LLM response)
├── fixtures/
│   ├── sample_transcript.srt    # Sample SRT file untuk testing
│   ├── sample_transcript.json   # Word-level JSON dari Whisper
│   ├── sample_highlight_response.json  # Mock LLM response (valid JSON)
│   ├── sample_highlight_malformed.txt  # Mock LLM response (malformed, untuk test retry)
│   ├── test_clip_short.mp4      # Clip pendek 5 detik untuk unit test (commit ke repo)
│   └── mock_config.json         # Config lengkap untuk testing
│
├── unit/
│   ├── test_config_manager.py
│   ├── test_logger.py
│   ├── test_gpu_detector.py
│   ├── test_highlight_detector.py
│   ├── test_speaker_layout.py
│   ├── test_video_processor.py
│   ├── test_caption_generator.py
│   ├── test_job_queue.py
│   └── test_batch_runner.py
│
└── integration/
    ├── test_pipeline_full.py    # Full pipeline dengan video pendek real
    └── test_orchestrator.py     # Orchestrator dengan mocked pipeline steps
```

---

## ⚙️ Setup & Tools

```
pytest
pytest-mock
pytest-cov
unittest.mock (built-in)
```

Jalankan semua test:

```bash
pytest tests/ -v --cov=pipeline --cov=providers --cov=batch --cov-report=term-missing
```

Jalankan hanya unit test (tanpa integration):

```bash
pytest tests/unit/ -v
```

---

## 📋 Fixtures Shared (`conftest.py`)

```python
# tests/conftest.py

import pytest
import json

@pytest.fixture
def mock_config():
    return {
        "llm": {
            "base_url": "http://localhost:11434/v1",
            "model": "llama3",
            "api_key": "test"
        },
        "transcription": {
            "base_url": "https://api.openai.com/v1",
            "model": "whisper-1",
            "api_key": "sk-test"
        },
        "portrait": {
            "face_backend": "opencv",
            "split_enabled": True,
            "split_active_threshold": 0.15,
            "split_window_ratio": 0.6,
            "split_hysteresis_sec": 3.0,
            "body_head_pad_ratio": 0.30,
            "body_lower_pad_ratio": 1.20
        }
    }

@pytest.fixture
def sample_srt_text():
    return """1
00:00:01,000 --> 00:00:03,000
Hello, welcome to this tutorial.

2
00:00:03,500 --> 00:00:06,000
Today we'll cover Python basics."""

@pytest.fixture
def valid_highlight_response():
    return json.dumps([
        {"start": 10.0, "end": 45.0, "hook_text": "The secret to fast Python", "score": 0.92},
        {"start": 120.0, "end": 160.0, "hook_text": "Why you're doing loops wrong", "score": 0.85}
    ])

@pytest.fixture
def malformed_highlight_response():
    return "Here are the highlights: [{start: 10, end: 45}]"  # Invalid JSON
```

---

## 🧪 Unit Tests Per Modul

### `providers/config_manager.py`

**File:** `tests/unit/test_config_manager.py`

| Test                             | Deskripsi                                  |
| -------------------------------- | ------------------------------------------ |
| `test_load_valid_config`         | Load config valid → return dict lengkap    |
| `test_load_missing_file`         | File tidak ada → raise `FileNotFoundError` |
| `test_load_missing_required_key` | Key wajib hilang → raise `ValueError`      |
| `test_save_and_reload`           | Save config lalu load → identical dict     |
| `test_api_key_not_in_repr`       | API key tidak muncul di `str(config)`      |

---

### `pipeline/highlight_detector.py`

**File:** `tests/unit/test_highlight_detector.py`

| Test                                    | Deskripsi                                                 |
| --------------------------------------- | --------------------------------------------------------- |
| `test_find_highlights_valid_response`   | LLM return JSON valid → `List[Highlight]` sorted by score |
| `test_retry_on_malformed_json`          | LLM return malformed → retry 2x → raise setelah exhaust   |
| `test_retry_succeeds_on_second_attempt` | Attempt 1 malformed, attempt 2 valid → return results     |
| `test_highlights_sorted_by_score`       | Hasil selalu sorted descending by score                   |
| `test_empty_highlights`                 | LLM return `[]` → return empty list, tidak raise          |
| `test_raw_llm_response_logged`          | Setiap LLM response di-log ke logger                      |

```python
# Contoh test retry
def test_retry_on_malformed_json(mock_config, malformed_highlight_response, valid_highlight_response, mocker):
    call_count = 0

    def mock_completion(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            return MockResponse(malformed_highlight_response)
        return MockResponse(valid_highlight_response)

    mocker.patch("providers.ai_client.get_client").return_value.chat.completions.create = mock_completion
    result = find_highlights("sample srt", mock_config)
    assert len(result) == 2
    assert call_count == 3  # initial + 2 retries
```

---

### `pipeline/speaker_layout.py`

**File:** `tests/unit/test_speaker_layout.py`

| Test                                   | Deskripsi                                                  |
| -------------------------------------- | ---------------------------------------------------------- |
| `test_single_face_returns_single_mode` | 1 wajah terdeteksi → semua segment SINGLE                  |
| `test_dual_active_returns_split_mode`  | 2 wajah aktif > threshold → segment SPLIT                  |
| `test_hysteresis_prevents_quick_flip`  | Mode flip hanya setelah 3 consecutive windows              |
| `test_no_face_falls_back_to_center`    | Tidak ada wajah → mode CENTER_FALLBACK                     |
| `test_body_safe_crop_center_face`      | Wajah di tengah → crop tidak melebihi frame                |
| `test_body_safe_crop_top_edge`         | Wajah di atas frame → crop_top = 0, terima partial padding |
| `test_body_safe_crop_bottom_edge`      | Wajah di bawah frame → crop_bottom = frame_height          |
| `test_body_safe_crop_left_edge`        | Wajah di kiri → crop_x = 0                                 |
| `test_head_pad_ratio_minimum`          | HEAD_PAD_RATIO tidak boleh < 0.2                           |
| `test_body_pad_ratio_minimum`          | BODY_PAD_RATIO tidak boleh < 1.0                           |
| `test_three_faces_uses_top_two`        | 3 wajah aktif → ambil 2 tertinggi score-nya                |

```python
# Test body-safe crop di edge cases
def test_body_safe_crop_top_edge():
    """Wajah sangat dekat atas frame — crop_top harus 0, bukan negatif."""
    face_bbox = (100, 5, 80, 60)  # (x, y, w, h) — y=5, sangat dekat atas
    src_height, src_width = 720, 1280
    crop = calculate_body_safe_crop(face_bbox, src_height, src_width)
    assert crop.top >= 0
    assert crop.bottom <= src_height
```

---

### `pipeline/video_processor.py`

**File:** `tests/unit/test_video_processor.py`

| Test                                      | Deskripsi                                 |
| ----------------------------------------- | ----------------------------------------- |
| `test_output_resolution_always_1080x1920` | Semua output harus 1080×1920              |
| `test_ffmpeg_command_single_mode`         | SINGLE mode → FFmpeg command yang benar   |
| `test_ffmpeg_command_split_mode`          | SPLIT mode → FFmpeg `vstack` filter chain |
| `test_gpu_flags_applied_when_cuda`        | CUDA tersedia → `h264_nvenc` di command   |
| `test_cpu_flags_applied_when_no_cuda`     | Tidak ada CUDA → `libx264` di command     |
| `test_split_panels_equal_height`          | Kedua panel SPLIT masing-masing 1080×960  |

> ⚠️ **Catatan:** Test FFmpeg command dilakukan dengan mock subprocess — jangan jalankan FFmpeg real di unit test.
> Untuk integration test FFmpeg, gunakan `tests/fixtures/test_clip_short.mp4`.

---

### `batch/job_queue.py`

**File:** `tests/unit/test_job_queue.py`

| Test                    | Deskripsi                                    |
| ----------------------- | -------------------------------------------- |
| `test_put_and_get_job`  | Put job → get job → sama                     |
| `test_status_lifecycle` | Status: PENDING → RUNNING → DONE             |
| `test_status_failed`    | Status: PENDING → RUNNING → FAILED           |
| `test_queue_order_fifo` | Job diproses sesuai urutan masuk             |
| `test_get_all_jobs`     | Return semua job dengan status masing-masing |

---

### `batch/batch_runner.py`

**File:** `tests/unit/test_batch_runner.py`

| Test                                   | Deskripsi                                            |
| -------------------------------------- | ---------------------------------------------------- |
| `test_requeue_pending_from_jobs_json`  | Startup → baca `output/jobs.json` → re-queue PENDING |
| `test_failed_job_does_not_block_queue` | Job FAILED tidak memblok job berikutnya              |
| `test_job_state_persisted_to_json`     | Setiap update status → `output/jobs.json` diupdate   |

---

## 🔗 Integration Tests

### `tests/integration/test_pipeline_full.py`

**Prerequisite:** FFmpeg, yt-dlp, dan Whisper API key tersedia di environment.

| Test                              | Deskripsi                                                        | Timeout |
| --------------------------------- | ---------------------------------------------------------------- | ------- |
| `test_download_real_video`        | Download video YouTube pendek (< 1 menit)                        | 60s     |
| `test_full_pipeline_short_video`  | Full pipeline: download → transcribe → highlight → cut → caption | 300s    |
| `test_portrait_output_resolution` | Output dari pipeline selalu 1080×1920                            | 120s    |

> ⚠️ **Integration test tidak dijalankan di CI default.** Jalankan manual saat milestone besar.
> Set env var `RUN_INTEGRATION=1` untuk mengaktifkan:
>
> ```bash
> RUN_INTEGRATION=1 pytest tests/integration/ -v
> ```

---

## 📊 Target Coverage

| Modul                            | Target Minimum |
| -------------------------------- | -------------- |
| `providers/config_manager.py`    | 90%            |
| `providers/ai_client.py`         | 70%            |
| `pipeline/highlight_detector.py` | 85%            |
| `pipeline/speaker_layout.py`     | 85%            |
| `pipeline/video_processor.py`    | 65%            |
| `pipeline/caption_generator.py`  | 65%            |
| `pipeline/orchestrator.py`       | 70%            |
| `batch/job_queue.py`             | 90%            |
| `batch/batch_runner.py`          | 75%            |

---

## 📝 Checklist Sebelum Merge ke Main

* [ ] Semua unit test pass (`pytest tests/unit/ -v`)
* [ ] Coverage tidak turun dari baseline
* [ ] Tidak ada hardcoded API key atau path
* [ ] Typed exceptions dipakai (bukan bare `Exception`)
* [ ] Fungsi publik baru sudah punya minimal 1 unit test
