# MIGRATION_MAP.md — v1 to v2 Component Mapping

This document maps the legacy v1 CustomTkinter architecture to the new v2 Gradio/Docker modular architecture.

## 1. Top-Level Entrypoints

| v1 File (Legacy) | v2 Target | Action | Notes |
|------------------|-----------|--------|-------|
| `app.py` | `server.py` | REWRITE | Migrate logic to `gr.Blocks`. Discard CustomTkinter UI. |
| `clipper_core.py` | `pipeline/orchestrator.py` | REWRITE | Break down monolith into the `pipeline/` package. |
| `webview_app.py` | N/A | REMOVE | No longer needed for web deployment. |
| `tiktok_uploader.py` | N/A (for MVP) | HOLD | Out of scope for M0-M8. |
| `youtube_uploader.py`| N/A (for MVP) | HOLD | Out of scope for M0-M8. |

## 2. Core Processing Logic (`clipper_core.py` teardown)

The 3379-line monolith is broken into modular pipeline steps:

| v1 Logic Block | v2 Target Module | Action | Notes |
|----------------|------------------|--------|-------|
| yt-dlp wrapper | `pipeline/downloader.py` | REWRITE | Standardize exception handling. Keep cookie logic. |
| OpenAI Transcription | `pipeline/transcriber.py` | REWRITE | Must use new `ai_client.py` wrapper. |
| LLM Highlighting | `pipeline/highlight_detector.py` | REWRITE | Add robust retry logic (ADR-005). |
| FFmpeg cutting/cropping | `pipeline/video_processor.py` | REWRITE | Use GPU detector, strict 9:16 output. |
| Face detection/tracking | `pipeline/speaker_layout.py` | REWRITE | Decouple from FFmpeg execution. Add SPLIT mode. |
| Subtitle creation | `pipeline/caption_generator.py` | REWRITE | Standalone ASS generation. |

## 3. Providers & Utilities

| v1 File / Logic | v2 Target | Action | Notes |
|-----------------|-----------|--------|-------|
| `config/config_manager.py` | `providers/config_manager.py` | REWRITE | Remove TK variables, use pure dicts. |
| `config/ai_provider_config.py`| `providers/ai_client.py` | REWRITE | Switch to universal OpenAI SDK factory (ADR-003). |
| `utils/logger.py` | `utils/logger.py` | KEEP/MODIFY | Refactor to standard `logging`, no UI hooks. |
| `utils/dependency_manager.py` | `utils/dependency_check.py` | REWRITE | Simplify for Docker environment (no auto-download). |
| `utils/gpu_detector.py` | `utils/gpu_detector.py` | KEEP/MODIFY | Standardize return values for FFmpeg flags. |

## 4. UI Components (`pages/`, `components/`, `dialogs/`)

**Action:** REMOVE ALL
CustomTkinter files (`pages/`, `dialogs/`, `components/`) cannot be ported to Gradio. They serve as business-logic references only but will not survive the migration.
* Replacements: Gradio Tabs in `server.py`.

## 5. Deployment & Configuration

| v1 Component | v2 Target | Action | Notes |
|--------------|-----------|--------|-------|
| `requirements.txt` | `requirements.txt` | REWRITE | Remove desktop UI libs (customtkinter, pywebview). Add Gradio, APScheduler. |
| `.env` / UI Config | `config.json` | REWRITE | Unified JSON config mounted as Docker volume. |
| `build.spec` (PyInstaller) | `Dockerfile` | REWRITE | Switch to Docker containerization. |
| N/A | `docker-compose.yml` | NEW | Standard app + output volumes. |
| N/A | `docker-compose.gpu.yml`| NEW | NVIDIA runtime override. |

## 6. Known Migration Risks

1. **Gradio UI Blocking:** v1 relied heavily on Tkinter `.after()` loops and thread checks. In v2, long-running processes must `yield` via generator or use background queues. (Mitigation: `batch/job_queue.py` + `orchestrator.py` streaming).
2. **AI Provider Abstraction:** v1 had highly specific code paths for different LLM settings. v2 flattens this using OpenAI-compatible specs (ADR-003). Users with legacy non-compliant endpoints will break. (Mitigation: clear error messages).
3. **FFmpeg Portability:** v1 downloaded FFmpeg binaries. v2 relies on `apt-get install ffmpeg` inside Docker. (Mitigation: ensure Dockerfile base image has hardware encoding support if needed).
4. **Speaker Layout Complexities:** Single to dual speaker hysteresis is new in v2. (Mitigation: Unit tests with synthetic bboxes before video integration).