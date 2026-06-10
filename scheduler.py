"""APScheduler setup for batch/cron jobs.

Provides a thin wrapper around APScheduler BackgroundScheduler with:
  - add_scheduled_job(cron_expr, url, config) → job_id
  - remove_scheduled_job(job_id)
  - list_scheduled_jobs() → list[dict]
  - shutdown()

Cron expression format (6-field, APScheduler default):
  second minute hour day month day_of_week
  Example: "0 0 9 * * *" → daily at 09:00:00
"""

from __future__ import annotations

from typing import Any

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.jobstores.base import JobLookupError
from apscheduler.triggers.cron import CronTrigger

from utils.logger import logger

# Module-level scheduler instance
_scheduler: BackgroundScheduler | None = None


def _get_scheduler() -> BackgroundScheduler:
    """Return the module-level BackgroundScheduler, creating it if needed."""
    global _scheduler
    if _scheduler is None:
        _scheduler = BackgroundScheduler()
        logger.info("Scheduler initialised (BackgroundScheduler)")
    return _scheduler


def _scheduled_job_target(url: str, config: dict[str, Any]) -> None:
    """Execute one pipeline job — called by APScheduler on each cron tick."""
    from pipeline.orchestrator import JobConfig, run_job

    job = JobConfig(url=url, config=config)
    logger.info("Scheduled job triggered: url=%s job_id=%s", url, job.id)
    try:
        run_job(job)
    except Exception:
        logger.exception("Scheduled job failed: job_id=%s", job.id)


def add_scheduled_job(
    cron_expr: str,
    url: str,
    config: dict[str, Any],
    job_id: str | None = None,
) -> str:
    """Register a cron-triggered job that runs the full pipeline.

    Args:
        cron_expr: 6-field cron string (second minute hour day month day_of_week).
        url: YouTube URL to process.
        config: Pipeline config dict (mirrors server.py Gradio config).
        job_id: Optional APScheduler job id. Auto-generated if None.

    Returns:
        The APScheduler job id string.
    """
    sched = _get_scheduler()
    trigger_fields = _parse_cron_expr(cron_expr)

    job = sched.add_job(
        _scheduled_job_target,
        trigger=CronTrigger(**trigger_fields, timezone="UTC"),
        args=[url],
        kwargs={"config": config},
        id=job_id or None,
        replace_existing=True,
    )
    logger.info("Added scheduled job %s: cron=%s url=%s", job.id, cron_expr, url)
    return job.id


def remove_scheduled_job(job_id: str) -> bool:
    """Remove a scheduled job by its APScheduler id.

    Returns True if the job was found and removed, False if not found.
    """
    sched = _get_scheduler()
    try:
        sched.remove_job(job_id)
        logger.info("Removed scheduled job %s", job_id)
        return True
    except JobLookupError:
        logger.warning("Scheduled job %s not found — nothing to remove", job_id)
        return False


def list_scheduled_jobs() -> list[dict[str, Any]]:
    """Return a list of all scheduled jobs with their next run time."""
    sched = _get_scheduler()
    jobs = sched.get_jobs()
    result = []
    for job in jobs:
        # APScheduler keeps jobs "pending" before scheduler.start(); pending jobs
        # do not have next_run_time populated yet.
        next_run_time = getattr(job, "next_run_time", None)
        result.append({
            "id": job.id,
            "next_run_time": str(next_run_time) if next_run_time else None,
            "trigger": str(job.trigger),
            "pending": bool(getattr(job, "pending", False)),
        })
    return result


def start() -> None:
    """Start the background scheduler (non-blocking)."""
    sched = _get_scheduler()
    if not sched.running:
        sched.start()
        logger.info("Scheduler started")
    else:
        logger.debug("Scheduler already running — start() skipped")


def shutdown(wait: bool = True) -> None:
    """Shut down the scheduler and release resources."""
    global _scheduler
    if _scheduler is not None and _scheduler.running:
        _scheduler.shutdown(wait=wait)
        logger.info("Scheduler shutdown complete")
    _scheduler = None


def _parse_cron_expr(expr: str) -> dict[str, str]:
    """Parse a 6-field cron string into APScheduler CronTrigger kwargs.

    Supports both 5-field (minute hour day month day_of_week) and
    6-field (second minute hour day month day_of_week) formats.
    """
    parts = expr.strip().split()
    if len(parts) == 6:
        fields = ["second", "minute", "hour", "day", "month", "day_of_week"]
    elif len(parts) == 5:
        fields = ["minute", "hour", "day", "month", "day_of_week"]
    else:
        raise ValueError(
            f"Cron expression must have 5 or 6 fields, got {len(parts)}: {expr!r}"
        )
    return dict(zip(fields, parts))


__all__ = [
    "add_scheduled_job",
    "list_scheduled_jobs",
    "remove_scheduled_job",
    "shutdown",
    "start",
]
