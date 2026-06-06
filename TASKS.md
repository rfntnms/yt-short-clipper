# 📋 TASKS.md — Backlog & Prioritas YT-Short-Clipper v2

> **Cara pakai:** Update file ini setiap kali task selesai, dimulai, atau ditambah.
> Status: `[ ]` = TODO · `[~]` = WIP · `[x]` = DONE · `[!]` = BLOCKED

---

## 🔴 P0 — Core Pipeline (Blocker untuk semua fitur lain)

### `pipeline/downloader.py`

* [ ] Implementasi `download(url, output_dir, cookies_path)` via `yt-dlp`
* [ ] Return path video + path `.srt` (jika tersedia di YouTube)
* [ ] Raise `DownloadError` jika gagal
* [ ] Support cookies.txt untuk video age-restricted

### `pipeline/transcriber.py`

* [ ] Implementasi `transcribe(video_path, config)` via Whisper endpoint
* [ ] Output format: word-level JSON (`{word, start, end}` per item)
* [ ] Skip transcription jika `.srt` sudah tersedia dari downloader
* [ ] Raise `TranscriptionError` jika API gagal

### `pipeline/highlight_detector.py`

* [ ] Baca `prompts/SYSTEM_PROMPT.md` sebagai system prompt
* [ ] Kirim transcript ke LLM via `providers/ai_client.py`
* [ ] Parse JSON response → `List[Highlight(start, end, hook_text, score)]`
* [ ] Retry loop maks 2x jika JSON malformed
* [ ] Log semua raw LLM response ke `app.log`
* [ ] Raise `HighlightDetectionError` setelah retry habis

### `pipeline/video_processor.py`

* [ ] Implementasi `cut(video_path, highlight)` → raw clip `.mp4`
* [ ] Implementasi `convert_to_portrait(clip_path, config)` → 9:16 output
* [ ] Integrasi `speaker_layout.analyze()` untuk SINGLE vs SPLIT decision
* [ ] SINGLE mode: standard 9:16 crop centered pada active speaker
* [ ] SPLIT mode: dual-speaker split-screen via FFmpeg `vstack` (lihat AGENTS.md §3a)
* [ ] Easing/smoothing pada crop window (anti-jitter)
* [ ] GPU hwaccel via `gpu_detector.py` (`h264_nvenc` vs `libx264`)
* [ ] Output selalu `1080×1920`

### `pipeline/speaker_layout.py`

* [ ] Per-frame face detection (OpenCV DNN atau MediaPipe)
* [ ] Active speaker scoring (lip movement delta atau optical flow)
* [ ] Sliding window mode decision (SINGLE vs SPLIT)
* [ ] Hysteresis window (default 3s, dari `config["portrait"]["split_hysteresis_sec"]`)
* [ ] Body-safe crop calculation (HEAD_PAD_RATIO + BODY_PAD_RATIO)
* [ ] Return `List[LayoutSegment(start_sec, end_sec, mode, speaker_crops)]`
* [ ] Handle semua edge cases (lihat tabel di AGENTS.md §3a)

### `pipeline/caption_generator.py`

* [ ] Extract audio dari clip
* [ ] Kirim ke Whisper → word-level timestamps
* [ ] Build file `.ass` (yellow highlight style, bold, configurable)
* [ ] FFmpeg burn-in captions ke final clip

### `pipeline/orchestrator.py`

* [ ] Implementasi `run_job(job: JobConfig)` — sequential full pipeline
* [ ] Implementasi `run_job_streaming(url, config)` — generator untuk live Gradio update
* [ ] Catch typed exceptions dari tiap modul pipeline
* [ ] Map exception → job status + user-facing message
* [ ] Tulis `output/<job_id>/data.json` setelah selesai

---

## 🟡 P1 — Provider & Config Foundation

### `providers/ai_client.py`

* [ ] Implementasi `get_client(config)` → return `openai.OpenAI` instance
* [ ] Support `base_url` custom (Ollama, LM Studio, dll.)
* [ ] **Tidak boleh** ada provider-specific branching di luar file ini

### `providers/config_manager.py`

* [ ] `load_config(path)` → return `dict`
* [ ] `save_config(config, path)`
* [ ] Validasi required keys saat load
* [ ] **Jangan** log API keys

### `utils/logger.py`

* [ ] Setup structured logger `ytclipper`
* [ ] Output ke `app.log` + console
* [ ] Rotasi log (max 10MB, keep 3 files)

### `utils/gpu_detector.py`

* [ ] Detect CUDA availability
* [ ] Return FFmpeg hwaccel flags yang tepat

### `utils/dependency_check.py`

* [ ] Validasi `ffmpeg` tersedia di PATH
* [ ] Validasi `yt-dlp` tersedia
* [ ] Raise error deskriptif jika tidak ada (tampil di Gradio startup)

---

## 🟡 P1 — Batch & Scheduling

### `batch/job_queue.py`

* [ ] Wrapper `queue.Queue` dengan status tracking
* [ ] Status lifecycle: `PENDING → RUNNING → DONE | FAILED`
* [ ] `put(job)`, `get()`, `update_status(job_id, status)`

### `batch/batch_runner.py`

* [ ] Background thread (start sekali di `server.py` init)
* [ ] Consume queue, run `orchestrator` per job
* [ ] Emit progress via `queue.Queue` → Gradio generator poll
* [ ] Re-queue `PENDING` jobs dari `output/jobs.json` saat startup
* [ ] Persist semua job state ke `output/jobs.json` setiap update

### `scheduler.py`

* [ ] APScheduler setup (BackgroundScheduler)
* [ ] `add_scheduled_job(cron_expr, url, config)`
* [ ] `remove_scheduled_job(job_id)`
* [ ] Parse cron expression dari string

---

## 🟡 P1 — Gradio UI (`server.py`)

* [ ] Tab **Process**: URL input, run button, progress log, output gallery
* [ ] Tab **Batch**: multi-URL textarea, queue status table, start/stop
* [ ] Tab **Schedule**: add/remove scheduled jobs, next run times
* [ ] Tab **Settings**: LLM config, Whisper config, portrait settings, caption style
* [ ] Tab **Logs**: tail `app.log` (auto-refresh setiap 5s)
* [ ] Semua long tasks menggunakan generator (`yield`) atau `gr.Progress`
* [ ] Tidak ada blocking call di Gradio event loop
* [ ] UI responsiveness test dengan pipeline mock

---

## 🟢 P2 — Deployment & Infrastructure

### Docker

* [ ] `Dockerfile` (base: `python:3.11-slim`, FFmpeg via apt, non-root user)
* [ ] `docker-compose.yml` (port 7860, volume output + config.json + cookies.txt)
* [ ] `docker-compose.gpu.yml` override (base: `nvidia/cuda:12.x-runtime`)
* [ ] `requirements.txt` lengkap

### Prompts

* [ ] Tulis `prompts/SYSTEM_PROMPT.md` awal
* [ ] Test prompt terhadap 3 tipe transcript (talk show, tutorial, podcast)
* [ ] Pastikan format output JSON eksplisit di dalam prompt

---

## 🔵 P2 — Testing

* [ ] Unit test `highlight_detector.py` (mock LLM, test retry logic)
* [ ] Unit test `speaker_layout.py` (synthetic face bboxes di edge cases)
* [ ] Unit test body-safe crop calculator
* [ ] Unit test `batch/job_queue.py`
* [ ] Integration test full pipeline dengan video pendek (< 5 menit)
* [ ] Test FFmpeg `vstack` filter chain pada static test clip

---

## 📌 Task Baru / Ide Tambahan

> Tambahkan task baru di sini sebelum di-prioritaskan

* [ ] *(kosong — tambahkan di sini)*

---

## ✅ Sudah Selesai

> Pindahkan task yang `[x]` ke sini beserta tanggal selesai

### RFN-15 — Repo Audit & MIGRATION_MAP.md
- [x] Inventory all v1 files/directories (`app.py`, `clipper_core.py`, `pages/`, `components/`, `services/`, etc.)
- [x] Map each v1 component to its v2 replacement (per AGENTS.md migration notes)
- [x] Identify files that can be reused vs must be rewritten
- [x] Create `MIGRATION_MAP.md` at project root
- [x] Identify migration risks (dependency conflicts, API changes, feature gaps)
*(Selesai: 2026-06-05)*

### RFN-16 — Development Safety Rules & Branching Strategy
- [x] Define branching strategy (feature branches, main protection, naming conventions)
- [x] Document safety rules: no code changes without issue, no ADR overrides
- [x] Create `.hermes/LINEAR_WORKFLOW.md` with complete workflow + branch conventions
- [x] Link to `.hermes/LINEAR_WORKFLOW.md` from AGENTS.md
- [x] Update CONTRIBUTING.md with v2 safety rules (migrate dari generic v1 ke v2-specific)
- [x] Document branch protection rules in CONTRIBUTING.md
*(Selesai: 2026-06-05)*
