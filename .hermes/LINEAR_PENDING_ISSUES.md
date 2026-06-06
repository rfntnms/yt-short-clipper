# Pending Linear Issues (M3 - M8)

*Linear MCP was timing out during batch creation. These are the remaining issues to be created in the `YT-Short-Clipper v2 Migration` project once the MCP connection is stable.*

---

## Milestone 3 — Testing & Reliability

### Issue: Unit Tests & Reliability Hardening
**Labels:** `P0`, `area:testing`
**Dependencies:** Blocked by M2 (RFN-21 to RFN-26)
**Description:**
Implement comprehensive Unit Testing & Reliability foundation based on TEST_PLAN.md.
- Create `tests/conftest.py` with shared fixtures.
- Implement unit tests for `config_manager`, `logger`, `ai_client`.
- Implement unit tests for `highlight_detector` (including LLM retry logic).
- Ensure all exceptions raised in `pipeline/` are typed.
- Audit logging to ensure no secrets/API keys are leaked.

---

## Milestone 4 — Gradio UI MVP

### Issue: Gradio UI MVP — server.py
**Labels:** `P0`, `area:ui`
**Dependencies:** Blocked by RFN-18 (config) and RFN-26 (orchestrator)
**Description:**
Implement `server.py` — the Gradio UI MVP for single-video processing.
- Set up `server.py` with Gradio Blocks.
- Build Process tab: URL input, Run button, Progress text, Video Gallery output.
- Build Settings tab: AI endpoints, models, API keys.
- Build Logs tab: Auto-refreshing tail of `app.log`.
- Wire Process tab to `orchestrator.run_job_streaming`.
- Ensure no blocking calls in Gradio event loop.

---

## Milestone 5 — Batch & Scheduling

### Issue: Batch Processing System & Queue Persistence
**Labels:** `P1`, `area:batch`
**Dependencies:** Blocked by RFN-26 (orchestrator) and Gradio UI MVP
**Description:**
Implement `batch/job_queue.py` and `batch/batch_runner.py` for asynchronous multi-URL processing.
- Track job status lifecycle (PENDING, RUNNING, DONE, FAILED).
- Persist queue state to `output/jobs.json`.
- On server startup, requeue PENDING jobs.
- Update Gradio UI with a Batch tab.

### Issue: APScheduler Integration for Cron Jobs
**Labels:** `P1`, `area:batch`
**Dependencies:** Blocked by Batch Processing System and Gradio UI MVP
**Description:**
Implement APScheduler setup in `scheduler.py` for automated cron-based processing.
- Set up BackgroundScheduler.
- Implement `add_scheduled_job` and `remove_scheduled_job`.
- Persist schedule to disk.
- Add Schedule tab to Gradio UI.

---

## Milestone 6 — Portrait Smart Crop & Split Mode

### Issue: Portrait Smart Crop & SPLIT Mode
**Labels:** `P1`, `area:pipeline`
**Dependencies:** Blocked by RFN-24 (video_processor) and Unit Tests
**Description:**
Implement `pipeline/speaker_layout.py` and dual-speaker FFmpeg split mode.
- Per-frame face detection.
- Calculate lip movement/optical flow to tag faces ACTIVE.
- Implement sliding window hysteresis (default 3s).
- Implement body-safe crop calculator.
- Update `pipeline/video_processor.py` to render SPLIT segments via FFmpeg `vstack`.

---

## Milestone 7 — Docker Deployment

### Issue: Docker Deployment & Runtime Volumes
**Labels:** `P2`, `area:docker`
**Dependencies:** Blocked by RFN-20 (requirements) and Gradio UI MVP
**Description:**
Create Docker deployment files for the v2 Gradio application.
- Create `Dockerfile` (python:3.11-slim, FFmpeg via apt).
- Create `docker-compose.yml` (port 7860, volumes).
- Create `docker-compose.gpu.yml` override.
- Run as non-root user.

---

## Milestone 8 — Integration Validation

### Issue: Final Integration Validation & Documentation Update
**Labels:** `P2`, `area:testing`, `area:docker`
**Dependencies:** Blocked by all previous milestones.
**Description:**
Run final integration validation across local, UI, and Docker environments.
- Run a short real video through full pipeline.
- Verify Gradio Process tab works in browser.
- Verify Docker smoke test passes.
- Verify output resolution is exactly 1080x1920.
- Update README/GUIDE/BUILD docs.

---

## Missing Dependencies on Existing Issues
If you manually link dependencies in Linear later, here is the map for M2 (already created):
- RFN-20 (requirements) blocked by RFN-15 (audit)
- RFN-19 (ai_client) blocked by RFN-18 (config)
- RFN-21 (downloader) blocked by RFN-20 and RFN-17
- RFN-22 (transcriber) blocked by RFN-19 and RFN-21
- RFN-23 (highlight) blocked by RFN-19 and RFN-22
- RFN-24 (video) blocked by RFN-17 and RFN-23
- RFN-25 (caption) blocked by RFN-22 and RFN-24
- RFN-26 (orchestrator) blocked by RFN-21, 22, 23, 24, 25