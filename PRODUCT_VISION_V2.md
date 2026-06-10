# Product Vision & Architecture Mapping (v2)

## Product Vision
YT-Short-Clipper v2 transforms the legacy desktop application into a robust, self-hosted web service via Docker and Gradio. The core mission is to automate the creation of engaging short-form video content (TikTok, Reels, Shorts) from long-form YouTube videos, powered by AI (highlight detection/captioning) and Computer Vision (smart portrait cropping).

Key pillars of the v2 rewrite:
1. **Accessibility & Deployment**: Shift from a local Python desktop app (CustomTkinter/PyInstaller) to a self-hosted web interface (Gradio) deployable anywhere via Docker Compose.
2. **Extensibility & Modularity**: Dismantle the `clipper_core.py` monolith into a discrete pipeline (`pipeline/` directory) where downloading, transcribing, AI analysis, video processing, and captioning are isolated, testable modules orchestrated centrally.
3. **AI Agnosticism**: Standardize all AI interactions through a universal OpenAI-compatible endpoint (`providers/ai_client.py`), allowing seamless switching between cloud providers (OpenAI) and local/self-hosted LLMs (Ollama, LM Studio) without provider-specific logic branching.
4. **Advanced Video Processing**: Introduce intelligent "SINGLE vs SPLIT" layout decisions based on active speaker detection, ensuring dynamic and visually engaging portrait (9:16) crops with body-safe boundaries to prevent awkward cuts.
5. **Batch & Automation**: Support queue-based batch processing and cron-style scheduling (APScheduler) for unattended, continuous content generation.

## Architecture Mapping

### UI & Presentation Layer
*   **Legacy**: CustomTkinter desktop app (`app.py`, `pages/`, `components/`, `dialogs/`).
*   **v2 Target**: Gradio web interface (`server.py`). Organized by functional Tabs (Process, Batch, Schedule, Settings, Logs). UI runs independently, using yields/generators and queue polling for progress updates to prevent blocking the event loop.

### Core Pipeline & Business Logic
*   **Legacy**: 3000+ line `clipper_core.py` monolith.
*   **v2 Target**: Modular package in `pipeline/` managed by `orchestrator.py`.
    *   `downloader.py`: yt-dlp wrapper for fetching video and `.srt`.
    *   `transcriber.py`: Whisper API integration for word-level JSON timestamps.
    *   `highlight_detector.py`: LLM prompt execution and JSON parsing with robust retry logic.
    *   `video_processor.py`: FFmpeg cutting, 9:16 conversion (easing, hwaccel).
    *   `speaker_layout.py`: Computer Vision (OpenCV/MediaPipe) active speaker detection, SINGLE/SPLIT sliding window mode decision, body-safe crop calculation.
    *   `caption_generator.py`: ASS subtitle generation and FFmpeg burn-in.

### AI Integration
*   **Legacy**: Tightly coupled OpenAI specific implementation.
*   **v2 Target**: `providers/ai_client.py` factory returning a standard `openai.OpenAI` client, configured via user-provided `base_url` (ADR-003). Single integration path for LLMs and Transcription.

### State & Background Processing
*   **Legacy**: Thread checks and Tkinter `.after()` loops.
*   **v2 Target**:
    *   `batch/job_queue.py`: In-memory `queue.Queue` tracking PENDING/RUNNING/DONE states, persisted to `output/jobs.json`.
    *   `batch/batch_runner.py`: Background consumer thread invoking the orchestrator.
    *   `scheduler.py`: APScheduler for cron-based job injection.

### Infrastructure & Deployment
*   **Legacy**: PyInstaller `.exe`, binary dependency downloads, `.env` config.
*   **v2 Target**:
    *   `Dockerfile` (Python 3.11-slim, apt-installed ffmpeg, non-root).
    *   `docker-compose.yml` (App port 7860, volume mounts for config/output).
    *   `docker-compose.gpu.yml` (NVIDIA runtime override).
    *   Unified `config.json` loaded via `providers/config_manager.py`.

### Telemetry & Validation
*   **Logging**: Move from UI-hooked logs to standard structured `logging` outputting to `app.log` via `utils/logger.py`.
*   **Dependencies**: Transition `dependency_manager.py` to `utils/dependency_check.py` purely for startup validation in the Docker context, relying on system-level provisioning.
*   **Hardware Setup**: Maintain `utils/gpu_detector.py` to dictate dynamic FFmpeg flags.

## PR Quality Gate Workflow

All features are developed via a three-agent workflow that ensures code quality before merge:

### Flow
```
Linear/Kanban task (RFN-XX)
  → clipper-dev agent implements in isolated git worktree
  → commits, pushes branch, opens GitHub PR
  → code-reviewer agent reviews PR (diff, tests, logic, security)
      → if PASS: approve PR, create "ready to merge" Kanban task
      → if FAIL: request changes on PR, create bug-fixer task as child
  → bug-fixer agent checks out PR branch, fixes only blocking issues
  → pushes fix to same PR branch
  → code-reviewer re-reviews the updated PR
  → loop until PASS
  → Rifqi manually approves and merges
```

### Rules
- clipper-dev NEVER commits to master. Always uses git worktree on a feature branch.
- code-reviewer NEVER modifies code. Read-only review only.
- bug-fixer ONLY fixes blocking issues found by code-reviewer. No unrelated refactors.
- No auto-merge. Final approval is always human (Rifqi).
- PR must pass both code review and (optionally) security review before merge.

### Agent Profiles
| Role | Profile | Responsibility |
|------|---------|----------------|
| Implement | clipper-dev | Code in worktree, commit, push, open PR |
| Review | code-reviewer | Review PR diff, approve/request changes |
| Fix | bug-fixer | Fix review findings on same PR branch |
| Security | security-reviewer | Optional security gate before merge |