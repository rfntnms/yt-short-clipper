# 🤖 AGENTS.md — AI Developer Guide for YT-Short-Clipper (v2)

> **This document reflects the v2 architecture.**
> Major changes from v1: CustomTkinter → Gradio, monolithic `clipper_core.py` → modular pipeline,
> multi-provider AI via OpenAI-compatible endpoint, batch processing + scheduling, Docker deployment.
>
> **Linear workflow rules:** All development execution must follow `.hermes/LINEAR_WORKFLOW.md`.
> Every code change must be tied to exactly one Linear issue (RFN-XXX), use the documented branch convention, and respect the status flow `Todo → In Progress → In Review → Done`.

---

## 📌 Project Overview

**YT-Short-Clipper** is a self-hosted web application that automates creation of short-form content
(TikTok, Reels, YouTube Shorts) from long-form YouTube videos.

It uses AI for highlight detection and captioning, and Computer Vision (OpenCV) for smart portrait cropping.
The v2 rewrite targets **self-hosted Docker deployment** with a **Gradio web UI** and a fully
**modular processing pipeline** supporting both cloud and local LLM providers.

---

## 🏗️ Architecture & Tech Stack

### Core Technology

| Layer              | v1 (Legacy)                | v2 (Current Target)                                      |
| ------------------ | -------------------------- | -------------------------------------------------------- |
| UI                 | CustomTkinter (desktop)    | **Gradio** (web, self-hosted)                            |
| Backend Entrypoint | `app.py` monolith          | `server.py` (Gradio app)                                 |
| Core Logic         | `clipper_core.py` monolith | Modular `pipeline/` package                              |
| AI Provider        | OpenAI only                | **OpenAI-compatible generic endpoint**                   |
| Deployment         | PyInstaller `.exe`         | **Docker Compose**                                       |
| Scheduling         | None                       | **APScheduler** (via `scheduler.py`)                     |
| Job Queue          | None                       | **In-process queue** (`queue.Queue`) or Redis (optional) |

### AI Provider Strategy

All AI calls MUST go through `providers/ai_client.py`. This abstraction supports:

* **OpenAI** (`gpt-4`, `gpt-4o`, `whisper-1`) — via `https://api.openai.com/v1`
* **Local LLM** (Ollama, LM Studio, any server) — via user-configured `base_url` using the
  OpenAI-compatible API (`/v1/chat/completions`, `/v1/audio/transcriptions`)

The config distinguishes providers by `base_url` and `model` only. There is NO provider-specific
branching in business logic. If a provider supports the OpenAI-compatible REST spec, it works.

```python
# providers/ai_client.py — canonical usage pattern
from openai import OpenAI

def get_client(config: dict) -> OpenAI:
    """Returns an OpenAI client pointed at the configured endpoint."""
    return OpenAI(
        api_key=config.get("api_key", "ollama"),   # local servers often ignore this
        base_url=config.get("base_url", "https://api.openai.com/v1"),
    )
```

**Config keys** (stored in `config.json`):

```json
{
  "llm": {
    "base_url": "http://localhost:11434/v1",
    "model": "llama3",
    "api_key": "ollama"
  },
  "transcription": {
    "base_url": "https://api.openai.com/v1",
    "model": "whisper-1",
    "api_key": "sk-..."
  },
  "portrait": {
    "face_backend": "opencv",          // "opencv" | "mediapipe"
    "split_enabled": true,             // enable dual-speaker split-screen
    "split_active_threshold": 0.15,    // lip movement delta to count as ACTIVE
    "split_window_ratio": 0.6,         // % of frames in window needed to trigger SPLIT
    "split_hysteresis_sec": 3.0,       // seconds before mode flip is committed
    "body_head_pad_ratio": 0.30,       // padding above head (fraction of face height)
    "body_lower_pad_ratio": 1.20       // padding below chin (fraction of face height)
  }
}
```

> ⚠️ LLM and transcription can use **different** endpoints. For example: local LLM for highlight
> detection, OpenAI Whisper for transcription (since local Whisper endpoints vary in quality).

---

## 📂 Directory Structure (v2 Target)

```
yt-short-clipper/
│
├── server.py                  # Gradio app entrypoint — defines all UI tabs and event bindings
├── config.json                # User settings (API keys, paths, preferences)
├── scheduler.py               # APScheduler setup for batch/cron jobs
│
├── pipeline/                  # ✅ Modular pipeline — one responsibility per module
│   ├── __init__.py
│   ├── orchestrator.py        # Runs the full pipeline for a single video job
│   ├── downloader.py          # yt-dlp download logic
│   ├── transcriber.py         # Whisper API call → word-level SRT/JSON
│   ├── highlight_detector.py  # LLM prompt → parse JSON highlights
│   ├── video_processor.py     # FFmpeg clip cutting + portrait conversion (OpenCV/MediaPipe)
│   ├── speaker_layout.py      # ✅ Speaker count detection → layout mode decision (SINGLE/SPLIT)
│   └── caption_generator.py   # ASS subtitle generation + FFmpeg burn-in
│
├── providers/
│   ├── ai_client.py           # OpenAI-compatible client factory (see above)
│   └── config_manager.py      # Load/save config.json
│
├── batch/
│   ├── job_queue.py           # In-process queue.Queue wrapper with status tracking
│   └── batch_runner.py        # Consumes queue, runs orchestrator per job, emits progress
│
├── utils/
│   ├── logger.py              # Structured logging → app.log
│   ├── gpu_detector.py        # CUDA/CPU detection for FFmpeg hwaccel flags
│   └── dependency_check.py    # Validates FFmpeg, yt-dlp presence at startup
│
├── assets/                    # Static files served by Gradio (icons, CSS overrides)
├── prompts/
│   └── SYSTEM_PROMPT.md       # Default LLM prompt for highlight detection
├── output/                    # Generated clips and job metadata (data.json per job)
│
├── docker-compose.yml         # Production deployment definition
├── Dockerfile
└── requirements.txt
```

---

## 🔄 Core Workflows (v2)

### 1. Single Video Pipeline

`server.py` (Gradio event) → `pipeline/orchestrator.py` → sequential steps:

```
orchestrator.run_job(job: JobConfig)
  │
  ├── 1. downloader.download(url)          → raw video + .srt file
  ├── 2. transcriber.transcribe(video)     → word-level JSON (if no .srt or forced)
  ├── 3. highlight_detector.find(srt, cfg) → List[Highlight] with timestamps + hook text
  ├── 4. FOR EACH highlight:
  │       video_processor.cut(video, highlight)          → raw clip (mp4)
  │       video_processor.convert_to_portrait(clip)      → 9:16 crop (OpenCV)
  │       caption_generator.generate_and_burn(clip, cfg) → final clip with captions
  └── 5. Write output/job_id/data.json                  → metadata for Gradio gallery
```

Each step is a **standalone function/class** that can be unit-tested independently.
`orchestrator.py` is the only module that imports across pipeline steps.

### 2. Highlight Detection Flow

`highlight_detector.py` → `find_highlights(srt_text, config)`:

1. Reads `prompts/SYSTEM_PROMPT.md` as system prompt.
2. Calls `providers/ai_client.py` → `client.chat.completions.create(...)`.
3. Parses LLM JSON response → validates schema (start, end, hook_text, score).
4. Returns `List[Highlight]` sorted by score descending.

> ⚠️ If LLM returns malformed JSON, `highlight_detector.py` MUST retry up to 2 times with a
> corrective follow-up message before raising. Log all raw LLM responses to `app.log`.

### 3. Portrait Conversion Flow

`video_processor.py` → `convert_to_portrait(clip_path, config)`:

1. Call `speaker_layout.analyze(clip_path)` → returns per-segment `LayoutPlan`.
2. For each segment in `LayoutPlan`:

   * `SINGLE` mode → standard 9:16 crop centered on active speaker (existing logic).
   * `SPLIT` mode → dual-speaker split-screen (see **Section 3a** below).
3. Smooth crop window with easing within each segment — avoid jitter.
4. FFmpeg filter applied with GPU hwaccel if CUDA detected (`gpu_detector.py`).
5. Output is always **1080×1920 (9:16)** regardless of layout mode.

### 3a. Dual-Speaker Split-Screen Flow

`speaker_layout.py` → `analyze(clip_path)` + `video_processor.py` → `render_split_segment(...)`:

**Step 1 — Per-frame speaker analysis (done in `speaker_layout.py`):**

```
For each sampled frame (every N frames, default N=5):
  1. Detect all face bounding boxes (OpenCV DNN or MediaPipe)
  2. For each face, compute "active speaker" score:
       - MediaPipe: lip landmark delta between consecutive frames
       - OpenCV fallback: optical flow magnitude in lip region
  3. Tag faces: ACTIVE (score > threshold) or SILENT
  4. Record: frame_idx → List[FaceInfo(bbox, active_score)]
```

**Step 2 — Segment mode decision (sliding window, `speaker_layout.py`):**

```
For each 1-second window of frames:
  - Count frames where 2+ ACTIVE faces detected simultaneously
  - If > 60% of frames in window have dual-active → mode = SPLIT
  - Otherwise → mode = SINGLE (track highest-score face)
  - Apply hysteresis: require 3 consecutive windows to flip mode
    (prevents flickering when one speaker briefly pauses)

Output: List[LayoutSegment(start_sec, end_sec, mode, speaker_crops)]
```

**Step 3 — Body-safe crop calculation (critical — prevents people being cut in half):**

For each face bbox `(x, y, w, h)` in SPLIT mode:

```python
# Body-safe crop constants
HEAD_PAD_RATIO   = 0.30   # 30% of face height above the top of head
BODY_PAD_RATIO   = 1.20   # 120% of face height below chin (reaches ~shoulders/chest)
PANEL_W          = 1080
PANEL_H          = 960    # half of 1920

# Vertical bounds — clamped to source frame
crop_top    = max(0, int(y - h * HEAD_PAD_RATIO))
crop_bottom = min(src_height, int(y + h + h * BODY_PAD_RATIO))
crop_height = crop_bottom - crop_top

# Horizontal bounds — centered on face, full panel width
crop_cx  = x + w // 2
crop_x   = max(0, min(src_width - PANEL_W, crop_cx - PANEL_W // 2))

# Scale cropped region to PANEL_W × PANEL_H
# Do NOT crop at face bbox directly — always use body-safe bounds
```

> ⚠️ **NEVER** set `crop_top` or `crop_bottom` to the raw face bbox `y` / `y+h`.
> Always apply `HEAD_PAD_RATIO` and `BODY_PAD_RATIO`. If the safe crop region
> exceeds frame bounds, clamp and accept off-center — do NOT shrink padding.

**Step 4 — FFmpeg filter chain for SPLIT rendering (`video_processor.py`):**

Each SPLIT segment is rendered as a separate FFmpeg call, then concatenated:

```bash
# For a split segment: speaker A (top panel) + speaker B (bottom panel)
ffmpeg -i input.mp4 \
  -filter_complex "
    [0:v]crop={cw_A}:{ch_A}:{cx_A}:{cy_A},scale=1080:960[top];
    [0:v]crop={cw_B}:{ch_B}:{cx_B}:{cy_B},scale=1080:960[bot];
    [top][bot]vstack=inputs=2[out]
  " \
  -map "[out]" \
  -t {segment_duration} \
  segment_split_{i}.mp4
```

**Step 5 — Segment assembly:**

After all segments (SINGLE + SPLIT) are rendered individually, concatenate with FFmpeg:

```bash
ffmpeg -f concat -safe 0 -i segments.txt -c copy output_portrait.mp4
```

**Layout decision diagram:**

```
                    ┌─────────────────────┐
                    │   Frame Analysis     │
                    │  (every 5 frames)    │
                    └──────────┬──────────┘
                               │
              ┌────────────────▼────────────────┐
              │  Dual ACTIVE speakers detected?  │
              └───────┬───────────────┬──────────┘
                     YES              NO
                      │               │
         ┌────────────▼──┐    ┌───────▼────────┐
         │  SPLIT mode   │    │  SINGLE mode   │
         │               │    │                │
         │  ┌─────────┐  │    │  ┌──────────┐  │
         │  │Speaker A│  │    │  │ Active   │  │
         │  │ TOP     │  │    │  │ speaker  │  │
         │  │1080×960 │  │    │  │ 1080×1920│  │
         │  ├─────────┤  │    │  └──────────┘  │
         │  │Speaker B│  │    └────────────────┘
         │  │ BOTTOM  │  │
         │  │1080×960 │  │
         │  └─────────┘  │
         └───────────────┘
              1080×1920
```

**Edge cases — MUST handle:**

| Situation                               | Behavior                                               |
| --------------------------------------- | ------------------------------------------------------ |
| 2 faces side-by-side (not stacked)      | Use SPLIT — each panel crops their half of the frame   |
| One speaker leaves mid-segment          | Switch to SINGLE after hysteresis window (3s default)  |
| Face detected but score below threshold | Treat as SILENT — do not include in SPLIT decision     |
| 3+ active speakers                      | SPLIT with top-2 highest-score speakers; ignore others |
| Face near top/bottom frame edge         | Clamp crop, accept partial body-safe padding           |
| No face detected at all                 | Fall back to center-crop (existing logic)              |

### 4. Captioning Flow

`caption_generator.py` → `generate_and_burn(clip_path, config)`:

1. Extract audio from clip.
2. Send to Whisper endpoint → word-level timestamps JSON.
3. Build `.ass` subtitle file (yellow highlight style, bold font, configurable).
4. FFmpeg `subtitles` filter burns into final clip.

### 5. Batch Processing Flow

`batch/job_queue.py` + `batch/batch_runner.py`:

1. Gradio UI enqueues multiple URLs → `job_queue.put(JobConfig)`.
2. `batch_runner.py` runs in a **background thread** (started once at server init).
3. Processes one job at a time by default (configurable concurrency).
4. Each job emits status updates via `queue.Queue` → Gradio polls via generator.
5. Job statuses: `PENDING → RUNNING → DONE | FAILED`.
6. All job state is stored in memory + persisted to `output/jobs.json` on each update.

### 6. Scheduling Flow

`scheduler.py` using **APScheduler**:

* Configured via Gradio settings tab (cron expression or interval).
* Scheduler triggers `batch_runner.submit(JobConfig)` at defined time.
* Example use case: process a YouTube playlist daily at 03:00.

```python
# scheduler.py — example setup
from apscheduler.schedulers.background import BackgroundScheduler

scheduler = BackgroundScheduler()

def add_scheduled_job(cron_expr: str, url: str, config: dict):
    scheduler.add_job(
        func=lambda: batch_runner.submit(JobConfig(url=url, config=config)),
        trigger="cron",
        **parse_cron(cron_expr),
        id=f"scheduled_{url[:20]}"
    )
```

---

## 🖥️ Gradio UI Structure (`server.py`)

Organize the Gradio app into **Tabs**, not one flat layout:

| Tab          | Purpose                                                               |
| ------------ | --------------------------------------------------------------------- |
| **Process**  | Single URL input, run button, progress log, output video gallery      |
| **Batch**    | Multi-URL textarea, queue status table, start/stop controls           |
| **Schedule** | Add/remove scheduled jobs, view next run times                        |
| **Settings** | LLM endpoint config, Whisper config, portrait settings, caption style |
| **Logs**     | Tail of `app.log` (auto-refresh every 5s)                             |

**Threading rules for Gradio:**

* Long tasks MUST use `gr.Progress` or generator functions (`yield`) for live updates.
* Never block the Gradio event loop. All pipeline calls run in threads via `concurrent.futures`.
* `batch_runner` background thread is started once in `server.py` at app startup.

```python
# server.py — generator pattern for live progress
def run_single(url: str, config: dict):
    for status in orchestrator.run_job_streaming(url, config):
        yield status  # Gradio updates UI on each yield
```

---

## 🐳 Docker Deployment

### `docker-compose.yml` structure

```yaml
services:
  yt-short-clipper:
    build: .
    ports:
      - "7860:7860"        # Gradio default port
    volumes:
      - ./output:/app/output
      - ./config.json:/app/config.json
      - ./cookies.txt:/app/cookies.txt
    environment:
      - GRADIO_SERVER_NAME=0.0.0.0
    restart: unless-stopped
```

### `Dockerfile` guidelines

* Base image: `python:3.11-slim`
* Install `ffmpeg` via `apt-get` in the image (do NOT rely on PATH injection).
* `yt-dlp` installed via pip (in `requirements.txt`), not binary injection.
* Run as non-root user.
* `CMD ["python", "server.py"]`

### GPU Support (Optional)

* Use `nvidia/cuda:12.x-runtime` base image for GPU builds.
* `gpu_detector.py` auto-detects CUDA and sets FFmpeg hwaccel flags accordingly.
* Document as optional via a separate `docker-compose.gpu.yml` override.

---

## 📝 Coding Standards & Conventions

### Module Boundaries

* Each `pipeline/` module exports **one primary public function or class**.
* Modules in `pipeline/` MUST NOT import from each other — only `orchestrator.py` does cross-imports.
* `providers/ai_client.py` is the **only** place that instantiates `openai.OpenAI`.

### Config Handling

* All config is loaded via `providers/config_manager.py`.
* Modules receive config as a plain `dict` — no global state, no `configparser`.
* Sensitive keys (API keys) are never logged.

### Error Handling

* `pipeline/` modules raise typed exceptions (`DownloadError`, `TranscriptionError`, etc.).
* `orchestrator.py` catches typed exceptions and maps to job status + user-facing messages.
* Gradio UI catches `orchestrator` errors and displays via `gr.Warning` or `gr.Error`.
* All exceptions are logged to `app.log` with stack trace via `utils/logger.py`.

### Logging

```python
# utils/logger.py — use structured logging
import logging

logger = logging.getLogger("ytclipper")

# Usage in any module:
from utils.logger import logger
logger.info("Starting download", extra={"url": url, "job_id": job_id})
logger.error("LLM parse failed", exc_info=True)
```

### Type Hinting

Required for all public functions in `pipeline/` and `providers/`:

```python
def find_highlights(srt_text: str, config: dict) -> list[Highlight]:
    ...
```

Use `dataclasses` or `pydantic.BaseModel` for structured data (`JobConfig`, `Highlight`, etc.).

### Async vs Threading

* Use `threading.Thread` or `concurrent.futures.ThreadPoolExecutor` for blocking I/O.
* Do **NOT** use `asyncio` unless Gradio explicitly requires it for a specific pattern.
* Gradio's generator pattern (`yield`) is the primary mechanism for live UI updates.

---

## 🤖 AI Agent Tips

### When modifying `pipeline/highlight_detector.py`

* The system prompt in `prompts/SYSTEM_PROMPT.md` is tightly coupled to the JSON schema expected.
  If you change the output schema, update the prompt AND the parser together.
* Always add a retry loop (max 2 retries) for JSON parse failures — local LLMs can return inconsistent formatting.

### When modifying `pipeline/video_processor.py`

* FFmpeg command construction lives here. Test FFmpeg commands in isolation before integrating.
* GPU hwaccel flags differ between CUDA (`h264_nvenc`) and CPU (`libx264`). Always check `gpu_detector.py` output first.
* Portrait crop math is sensitive — add unit tests for edge cases (no face detected, multiple faces, face near frame edge).
* For SPLIT mode, **always build and test the FFmpeg `vstack` filter chain on a static test clip first** before wiring to the live pipeline.
* Segment concat via `ffmpeg -f concat` requires all segments to share the same resolution, fps, and codec. Enforce this at render time.

### When modifying `pipeline/speaker_layout.py`

* This module is **read-only** relative to the video — it only analyzes, never writes frames.
* The hysteresis window (default 3s / `HYSTERESIS_SEC`) is configurable via `config["portrait"]["split_hysteresis_sec"]`. Do not hardcode it.
* The `BODY_PAD_RATIO` and `HEAD_PAD_RATIO` constants MUST NOT be reduced below `1.0` and `0.2` respectively — values below these will cause visible body cutoff.
* When sampling frames for analysis, use `cv2.VideoCapture` seek (`cap.set(cv2.CAP_PROP_POS_FRAMES, idx)`) — do not decode every frame sequentially.
* Unit test the body-safe crop calculator with synthetic face bboxes at frame edges (top-left, bottom-right, center) before integrating.

### When modifying `batch/`

* The job queue is in-memory. On server restart, `PENDING` jobs are lost unless `output/jobs.json` is re-read at startup. Implement this re-queue logic in `batch_runner.py`.

### When modifying `server.py` (Gradio)

* Keep Gradio component definitions at the top of each Tab block for readability.
* Gradio state (`gr.State`) should hold minimal data — heavy state goes in `batch/job_queue.py`.
* Test UI responsiveness with a slow pipeline mock before integrating real processing.

### When updating `prompts/SYSTEM_PROMPT.md`

* Always specify output format explicitly: `Respond ONLY with valid JSON. No preamble. No markdown fences.`
* Include schema example in the prompt itself for few-shot guidance.
* After changes, test against at least 3 different transcript types (talk show, tutorial, podcast).

---

## 🔗 Important Files for AI Agents

> **Mandatory reading order for every Hermes Agent / AI development session.**
> Do not start editing code before reading these files. If any file is missing, create it or report it as a blocker.

| Order | File           | Purpose                                                                                    | Agent Action                                                                                                                |
| ----: | -------------- | ------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------- |
|     1 | `CONTEXT.md`   | Session handoff memory, current status, last modified files, pending decisions, next steps | Read first. Continue from `Next Steps`. Update `Sesi Terakhir`, `Next Steps`, and `Riwayat Sesi` before ending the session. |
|     2 | `DECISIONS.md` | Architecture Decision Records (ADR): accepted design decisions and trade-offs              | Read before proposing architecture changes. Never override `ACCEPTED` decisions unless explicitly asked by the user.        |
|     3 | `TASKS.md`     | Prioritized backlog split by P0/P1/P2                                                      | Pick work from the highest-priority unfinished task. Mark task `[~]` when started and `[x]` only after tests pass.          |
|     4 | `PROGRESS.md`  | Implementation status per module, known bugs, migration status, coverage target            | Update when a module changes status, a bug is found/fixed, or coverage changes.                                             |
|     5 | `TEST_PLAN.md` | Unit/integration testing strategy, fixtures, coverage goals, merge checklist               | Use before implementation. Add/update tests for every public function or behavior change.                                   |
|     6 | `AGENTS.md`    | Project architecture, coding standards, pipeline design, module boundaries                 | Keep open as the primary development constitution. Update when architecture guidance changes.                               |

### Supporting Project Files

| File                       | Purpose                                |
| -------------------------- | -------------------------------------- |
| `README.md`                | General user/setup guide               |
| `GUIDE.md`                 | Detailed usage instructions            |
| `BUILD.md`                 | Docker build and local dev setup       |
| `COOKIES.md`               | YouTube authentication via cookies.txt |
| `prompts/SYSTEM_PROMPT.md` | LLM highlight detection prompt         |
| `output/jobs.json`         | Persisted job state (auto-generated)   |

### Recommended Additional Agent Guidance Files

Create these files if long-term development becomes harder to track:

| File                  | Purpose                                                                                               |
| --------------------- | ----------------------------------------------------------------------------------------------------- |
| `REPO_MAP.md`         | Human-readable map of directories, entrypoints, data flow, and ownership per module                   |
| `ROADMAP.md`          | Milestone-based delivery plan, e.g. M1 foundation, M2 pipeline, M3 UI, M4 Docker, M5 GPU optimization |
| `SPEC.md`             | Product requirements: user flows, accepted inputs/outputs, non-goals, and acceptance criteria         |
| `DEBUGGING.md`        | Known debugging playbooks for FFmpeg, yt-dlp, Gradio, provider config, GPU/NVENC, and subtitles       |
| `SECURITY.md`         | Rules for API keys, cookies.txt, user uploads, logs, Docker permissions, and prompt injection risks   |
| `REVIEW_CHECKLIST.md` | Pre-merge checklist: architecture boundary, tests, logging, config, security, and UX                  |
| `.env.example`        | Safe template only. No real secrets. Use this only if environment variables are later introduced.     |
| `MCP.md`              | List of approved MCP servers/plugins, allowed permissions, setup commands, and security limits        |

---

## 🗺️ Migration Notes (v1 → v2)

| v1 Component                     | v2 Replacement                                           | Action                             |
| -------------------------------- | -------------------------------------------------------- | ---------------------------------- |
| `app.py` (CustomTkinter)         | `server.py` (Gradio)                                     | Rewrite                            |
| `clipper_core.py`                | `pipeline/` package                                      | Refactor — split into 5 modules    |
| `pages/`                         | Gradio Tabs in `server.py`                               | Rewrite                            |
| `components/ai_provider_card.py` | `providers/ai_client.py` + Settings Tab                  | Rewrite                            |
| `utils/dependency_manager.py`    | `utils/dependency_check.py` + Dockerfile                 | Simplify — Docker handles binaries |
| `build.spec`                     | `Dockerfile` + `docker-compose.yml`                      | Replace                            |
| No batch system                  | `batch/` package                                         | New                                |
| No scheduling                    | `scheduler.py`                                           | New                                |
| Single-speaker portrait only     | `speaker_layout.py` + SPLIT mode in `video_processor.py` | New                                |
