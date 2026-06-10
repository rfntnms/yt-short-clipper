"""Gradio web UI entrypoint for YT-Short-Clipper v2.

5-tab layout:
  Process  — single URL, run button, progress log, output gallery
  Batch    — multi-URL textarea, queue status table, start/stop
  Schedule — add/remove cron jobs, next-run times
  Settings — LLM config, Whisper config, portrait settings, caption style
  Logs     — tail of app.log (auto-refresh every 5 s)

All long tasks use generator functions so the Gradio event loop never blocks.
"""

from __future__ import annotations

import threading
import uuid
from pathlib import Path
from typing import Any

import gradio as gr

from batch.job_queue import JobQueue, QueueJob
from batch.batch_runner import BatchRunner
from pipeline.orchestrator import run_job_streaming
from providers.config_manager import load_config, save_config
from scheduler import (
    add_scheduled_job,
    list_scheduled_jobs,
    remove_scheduled_job,
    start as scheduler_start,
)
from utils.logger import logger

# ── paths ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.json"
OUTPUT_DIR = PROJECT_ROOT / "output"
LOG_PATH = PROJECT_ROOT / "app.log"

# ── shared state ────────────────────────────────────────────────────────
_job_queue = JobQueue()
_batch_runner = BatchRunner(_job_queue)

# Start background services once at import time
_batch_runner.start()
scheduler_start()
logger.info("server.py initialised — BatchRunner + Scheduler started")

# Dependency check flag — set False from tests to simulate missing deps.
_DEPENDENCIES_OK: bool = True


def _parse_subtitle_lang(lang_str: str) -> str:
    """Parse language code from 'en - English' or plain 'en'."""
    if " - " in lang_str:
        return lang_str.split(" - ", 1)[0].strip()
    return lang_str.strip()


def save_settings(
    llm_url: str,
    llm_model: str,
    llm_key: str,
    tr_url: str,
    tr_model: str,
    tr_key: str,
    face_backend: str,
    split_enabled_str: str,
) -> str:
    """Save settings (unit-test-friendly wrapper around _settings_save)."""
    cfg = _ui_to_config({
        "llm_base_url": llm_url,
        "llm_model": llm_model,
        "llm_api_key": llm_key,
        "tr_base_url": tr_url,
        "tr_model": tr_model,
        "tr_api_key": tr_key,
        "face_backend": face_backend,
        "split_enabled": split_enabled_str.lower() == "true",
    })
    try:
        save_config(cfg, DEFAULT_CONFIG_PATH)
        return "Settings saved successfully."
    except Exception as e:
        return f"Failed to save settings: {e}"


def load_settings() -> dict[str, Any]:
    """Return current config dict (unit-test-friendly wrapper)."""
    return _get_config()


def process_video(
    url: str,
    max_clips: int = 5,
    subtitle_lang: str = "en",
    progress: Any = None,
) -> tuple[str, list[str]]:
    """Process a single video URL (unit-test-friendly wrapper).

    This is a synchronous wrapper used by unit tests. The Gradio
    tab uses the async generator ``_run_process`` instead.
    """
    if not url or not url.strip():
        return "Please enter a YouTube URL.", []

    if not _DEPENDENCIES_OK:
        return "Missing dependencies. Check FFmpeg and yt-dlp.", []

    from pipeline.orchestrator import JobConfig, run_job

    cfg = _get_config()
    job = JobConfig(
        id=uuid.uuid4().hex[:12],
        url=url.strip(),
        config=cfg,
        max_clips=max_clips,
        subtitle_lang=_parse_subtitle_lang(subtitle_lang),
    )
    result = run_job(job)

    if progress:
        progress()

    if result.status.value == "COMPLETED":
        clip_paths = [c["path"] for c in (result.clips or [])]
        return f"Done. {len(clip_paths)} clip(s) generated.", clip_paths
    else:
        return f"Error: {result.error}", []


# ── helpers ──────────────────────────────────────────────────────────────


def _get_config() -> dict[str, Any]:
    """Load config from disk, falling back to defaults."""
    try:
        return dict(load_config(DEFAULT_CONFIG_PATH))
    except Exception:
        return {}


def _tail_log(n_lines: int = 80) -> str:
    """Return the last N lines of app.log."""
    try:
        if not LOG_PATH.exists():
            return "(no log yet)"
        lines = LOG_PATH.read_text(encoding="utf-8", errors="replace").splitlines()
        return "\n".join(lines[-n_lines:])
    except Exception as e:
        return f"(error reading log: {e})"


def _config_to_ui(cfg: dict[str, Any]) -> dict[str, Any]:
    """Flatten a nested config dict into individual UI field values."""
    llm = cfg.get("llm", {})
    tr = cfg.get("transcription", {})
    portrait = cfg.get("portrait", {})
    caption = cfg.get("caption", {})
    return {
        "llm_base_url": llm.get("base_url", "https://api.openai.com/v1"),
        "llm_model": llm.get("model", "gpt-4"),
        "llm_api_key": llm.get("api_key", ""),
        "tr_base_url": tr.get("base_url", "https://api.openai.com/v1"),
        "tr_model": tr.get("model", "whisper-1"),
        "tr_api_key": tr.get("api_key", ""),
        "face_backend": portrait.get("face_backend", "opencv"),
        "split_enabled": portrait.get("split_enabled", True),
        "caption_font": caption.get("font", "Arial"),
        "caption_fontsize": caption.get("fontsize", 52),
        "caption_outline": caption.get("outline", 3),
    }


def _ui_to_config(values: dict[str, Any]) -> dict[str, Any]:
    """Rebuild nested config dict from individual UI field values."""
    return {
        "llm": {
            "base_url": values.get("llm_base_url", "https://api.openai.com/v1"),
            "model": values.get("llm_model", "gpt-4"),
            "api_key": values.get("llm_api_key", ""),
        },
        "transcription": {
            "base_url": values.get("tr_base_url", "https://api.openai.com/v1"),
            "model": values.get("tr_model", "whisper-1"),
            "api_key": values.get("tr_api_key", ""),
        },
        "portrait": {
            "face_backend": values.get("face_backend", "opencv"),
            "split_enabled": values.get("split_enabled", True),
        },
        "caption": {
            "font": values.get("caption_font", "Arial"),
            "fontsize": int(values.get("caption_fontsize", 52)),
            "outline": int(values.get("caption_outline", 3)),
        },
    }


# ── Process tab handlers ────────────────────────────────────────────────


def _run_process(url: str, state_cfg: dict[str, Any]):
    """Generator: run the pipeline and yield progress + gallery updates."""
    if not url or not url.strip():
        yield "⚠ Please enter a YouTube URL.", None
        return

    cfg = state_cfg if state_cfg else _get_config()
    gallery_items: list[tuple[str, str]] = []

    for msg in run_job_streaming(url.strip(), cfg):
        # When job completes, scan for output clips
        if msg.startswith("✓ Job complete"):
            try:
                output_dirs = sorted(
                    OUTPUT_DIR.iterdir(),
                    key=lambda p: p.stat().st_mtime,
                    reverse=True,
                )
                if output_dirs:
                    clips_dir = output_dirs[0] / "clips"
                    if clips_dir.exists():
                        for mp4 in sorted(clips_dir.glob("*_captioned.mp4")):
                            gallery_items.append((str(mp4), mp4.name))
                        if not gallery_items:
                            for mp4 in sorted(clips_dir.glob("*_portrait.mp4")):
                                gallery_items.append((str(mp4), mp4.name))
            except Exception:
                pass

        yield msg, gallery_items or None


# ── Batch tab handlers ──────────────────────────────────────────────────


def _batch_add(urls_text: str, state_cfg: dict[str, Any]):
    """Add multiple URLs to the queue from a newline-separated textarea."""
    if not urls_text.strip():
        yield "⚠ Enter at least one URL.", _batch_table()
        return

    cfg = state_cfg if state_cfg else _get_config()
    added = 0
    for line in urls_text.strip().splitlines():
        url = line.strip()
        if not url:
            continue
        job = QueueJob(
            id=uuid.uuid4().hex[:12],
            url=url,
            config=cfg,
        )
        _job_queue.put(job)
        added += 1

    yield f"✓ {added} job(s) added to queue.", _batch_table()


def _batch_table() -> list[list[str]]:
    """Return queue data for gr.Dataframe."""
    rows: list[list[str]] = []
    for job in _job_queue.list_jobs():
        status = job.status.value
        result_summary = ""
        if job.result and isinstance(job.result, dict):
            clips = job.result.get("clips", [])
            result_summary = f"{len(clips)} clips"
            if job.result.get("error"):
                result_summary = job.result["error"]
        rows.append([job.id, job.url, status, result_summary])
    return rows


def _batch_stop() -> tuple[str, list]:
    """Stop the batch runner."""
    _batch_runner.stop()
    return "Batch runner stopped.", _batch_table()


def _batch_start() -> tuple[str, list]:
    """Start / restart the batch runner."""
    global _batch_runner
    _batch_runner = BatchRunner(_job_queue)
    _batch_runner.start()
    return "Batch runner started.", _batch_table()


# ── Schedule tab handlers ───────────────────────────────────────────────


def _schedule_add(cron_expr: str, url: str, state_cfg: dict[str, Any]):
    """Add a cron-scheduled job."""
    if not cron_expr.strip():
        yield "⚠ Cron expression required.", _schedule_table()
        return
    if not url.strip():
        yield "⚠ URL required.", _schedule_table()
        return

    cfg = state_cfg if state_cfg else _get_config()
    try:
        job_id = add_scheduled_job(cron_expr.strip(), url.strip(), cfg)
        yield f"✓ Scheduled job {job_id}", _schedule_table()
    except Exception as e:
        yield f"⚠ Failed: {e}", _schedule_table()


def _schedule_remove(job_id: str):
    """Remove a scheduled job by ID."""
    if not job_id.strip():
        yield "⚠ Enter a job ID.", _schedule_table()
        return

    removed = remove_scheduled_job(job_id.strip())
    status = "✓ Removed" if removed else "⚠ Not found"
    yield f"{status} {job_id.strip()}", _schedule_table()


def _schedule_table() -> list[list[str]]:
    """Return schedule data for gr.Dataframe."""
    rows: list[list[str]] = []
    for job in list_scheduled_jobs():
        rows.append([
            job["id"],
            job.get("trigger", ""),
            str(job.get("next_run_time", "pending")),
        ])
    return rows


# ── Settings tab handlers ───────────────────────────────────────────────


def _settings_save(
    llm_base_url: str,
    llm_model: str,
    llm_api_key: str,
    tr_base_url: str,
    tr_model: str,
    tr_api_key: str,
    face_backend: str,
    split_enabled: bool,
    caption_font: str,
    caption_fontsize: float,
    caption_outline: float,
) -> tuple[str, dict[str, Any]]:
    """Persist settings and return (status_message, updated_config)."""
    cfg = _ui_to_config({
        "llm_base_url": llm_base_url,
        "llm_model": llm_model,
        "llm_api_key": llm_api_key,
        "tr_base_url": tr_base_url,
        "tr_model": tr_model,
        "tr_api_key": tr_api_key,
        "face_backend": face_backend,
        "split_enabled": split_enabled,
        "caption_font": caption_font,
        "caption_fontsize": caption_fontsize,
        "caption_outline": caption_outline,
    })
    try:
        save_config(cfg, DEFAULT_CONFIG_PATH)
        return "✓ Settings saved.", cfg
    except Exception as e:
        return f"⚠ Save failed: {e}", cfg


# ── Build Gradio app ────────────────────────────────────────────────────

# Pre-load config so Settings tab fields show current values on startup
_initial_cfg = _get_config()
_ui_defaults = _config_to_ui(_initial_cfg)

with gr.Blocks(title="YT-Short-Clipper v2") as app:
    # Shared config state — synced from Settings tab
    state_config = gr.State(value=_initial_cfg)

    gr.Markdown("# 🎬 YT-Short-Clipper v2\n*Self-hosted AI short-form content creator*")

    with gr.Tabs():
        # ── Process tab ──────────────────────────────────────────────
        with gr.Tab("Process"):
            with gr.Row():
                with gr.Column(scale=3):
                    inp_url = gr.Textbox(
                        label="YouTube URL",
                        placeholder="https://www.youtube.com/watch?v=...",
                        lines=1,
                    )
                with gr.Column(scale=1):
                    btn_run = gr.Button("▶ Run", variant="primary", size="lg")

            out_log = gr.Textbox(
                label="Progress",
                lines=12,
                interactive=False,
            )
            out_gallery = gr.Gallery(
                label="Generated Clips",
                columns=3,
                height="auto",
                object_fit="contain",
            )

            btn_run.click(
                fn=_run_process,
                inputs=[inp_url, state_config],
                outputs=[out_log, out_gallery],
                show_progress="minimal",
            )

        # ── Batch tab ────────────────────────────────────────────────
        with gr.Tab("Batch"):
            gr.Markdown("Enter one YouTube URL per line.")
            with gr.Row():
                with gr.Column(scale=3):
                    batch_urls = gr.Textbox(
                        label="URLs (one per line)",
                        lines=6,
                        placeholder="https://www.youtube.com/watch?v=...\nhttps://youtu.be/...",
                    )
                with gr.Column(scale=1):
                    batch_add_btn = gr.Button("＋ Add to Queue", variant="primary")
                    batch_start_btn = gr.Button("▶ Start Batch")
                    batch_stop_btn = gr.Button("⏹ Stop")

            batch_table = gr.Dataframe(
                headers=["Job ID", "URL", "Status", "Result"],
                label="Queue",
                interactive=False,
                wrap=True,
            )

            batch_add_btn.click(
                fn=_batch_add,
                inputs=[batch_urls, state_config],
                outputs=[out_log, batch_table],
            )
            batch_start_btn.click(
                fn=_batch_start,
                inputs=None,
                outputs=[out_log, batch_table],
            )
            batch_stop_btn.click(
                fn=_batch_stop,
                inputs=None,
                outputs=[out_log, batch_table],
            )

        # ── Schedule tab ─────────────────────────────────────────────
        with gr.Tab("Schedule"):
            with gr.Row():
                with gr.Column(scale=2):
                    sched_cron = gr.Textbox(
                        label="Cron Expression",
                        placeholder="0 0 9 * * *  (daily at 09:00 UTC)",
                    )
                with gr.Column(scale=2):
                    sched_url = gr.Textbox(
                        label="YouTube URL",
                        placeholder="https://www.youtube.com/watch?v=...",
                    )
                with gr.Column(scale=1):
                    sched_add_btn = gr.Button("＋ Add", variant="primary")

            schedule_table = gr.Dataframe(
                headers=["Job ID", "Trigger", "Next Run"],
                label="Scheduled Jobs",
                interactive=False,
            )

            with gr.Row():
                sched_remove_id = gr.Textbox(
                    label="Job ID to remove",
                    placeholder="paste job id…",
                )
                sched_remove_btn = gr.Button("🗑 Remove")

            sched_add_btn.click(
                fn=_schedule_add,
                inputs=[sched_cron, sched_url, state_config],
                outputs=[out_log, schedule_table],
            )
            sched_remove_btn.click(
                fn=_schedule_remove,
                inputs=[sched_remove_id],
                outputs=[out_log, schedule_table],
            )

        # ── Settings tab ─────────────────────────────────────────────
        with gr.Tab("Settings"):
            gr.Markdown("## LLM (Highlight Detection)")
            with gr.Row():
                s_llm_base_url = gr.Textbox(
                    label="Base URL",
                    value=_ui_defaults.get("llm_base_url", "https://api.openai.com/v1"),
                )
                s_llm_model = gr.Textbox(
                    label="Model",
                    value=_ui_defaults.get("llm_model", "gpt-4"),
                )
            s_llm_api_key = gr.Textbox(
                label="API Key",
                type="password",
                value=_ui_defaults.get("llm_api_key", ""),
            )

            gr.Markdown("## Transcription (Whisper)")
            with gr.Row():
                s_tr_base_url = gr.Textbox(
                    label="Base URL",
                    value=_ui_defaults.get("tr_base_url", "https://api.openai.com/v1"),
                )
                s_tr_model = gr.Textbox(
                    label="Model",
                    value=_ui_defaults.get("tr_model", "whisper-1"),
                )
            s_tr_api_key = gr.Textbox(
                label="API Key",
                type="password",
                value=_ui_defaults.get("tr_api_key", ""),
            )

            gr.Markdown("## Portrait")
            with gr.Row():
                s_face_backend = gr.Dropdown(
                    label="Face Detection Backend",
                    choices=["opencv", "mediapipe"],
                    value=_ui_defaults.get("face_backend", "opencv"),
                )
                s_split_enabled = gr.Checkbox(
                    label="Enable Dual-Speaker Split",
                    value=_ui_defaults.get("split_enabled", True),
                )

            gr.Markdown("## Captions")
            with gr.Row():
                s_caption_font = gr.Textbox(
                    label="Font",
                    value=_ui_defaults.get("caption_font", "Arial"),
                )
                s_caption_fontsize = gr.Number(
                    label="Font Size",
                    value=_ui_defaults.get("caption_fontsize", 52),
                )
                s_caption_outline = gr.Number(
                    label="Outline Width",
                    value=_ui_defaults.get("caption_outline", 3),
                )

            btn_save_settings = gr.Button("💾 Save Settings", variant="primary")
            settings_status = gr.Textbox(label="Status", interactive=False)

            all_settings_inputs = [
                s_llm_base_url,
                s_llm_model,
                s_llm_api_key,
                s_tr_base_url,
                s_tr_model,
                s_tr_api_key,
                s_face_backend,
                s_split_enabled,
                s_caption_font,
                s_caption_fontsize,
                s_caption_outline,
            ]

            btn_save_settings.click(
                fn=_settings_save,
                inputs=all_settings_inputs,
                outputs=[settings_status, state_config],
            )

        # ── Logs tab ─────────────────────────────────────────────────
        with gr.Tab("Logs"):
            log_viewer = gr.Textbox(
                label="app.log",
                lines=30,
                interactive=False,
                max_lines=200,
            )
            log_timer = gr.Timer(value=5, active=True)
            log_timer.tick(
                fn=lambda: _tail_log(80),
                outputs=[log_viewer],
            )

    gr.Markdown("---\n*YT-Short-Clipper v2 · Self-hosted AI clipper · RFN-28*")


# ── Entrypoint ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    logger.info("Launching Gradio server on 0.0.0.0:7860")
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )
