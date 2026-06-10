"""Job Queue — thread-safe queue with persistent state.

Wraps queue.Queue to maintain in-process job definitions.
Saves queue state to output/jobs.json on every status change.
"""

from __future__ import annotations

import json
import queue
import threading
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from utils.logger import logger

BASE_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"
JOBS_FILE = BASE_OUTPUT_DIR / "jobs.json"


class JobQueueStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    DONE = "DONE"
    FAILED = "FAILED"


@dataclass
class QueueJob:
    """Representation of a job in the queue."""

    id: str
    url: str
    config: dict[str, Any] = field(default_factory=dict)
    status: JobQueueStatus = JobQueueStatus.PENDING
    result: dict[str, Any] | None = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


class JobQueue:
    """A thread-safe job queue backed by JSON persistence."""

    def __init__(self, persist_path: str | Path | None = None) -> None:
        self.persist_path = Path(persist_path) if persist_path is not None else JOBS_FILE
        self._inner: queue.Queue[QueueJob] = queue.Queue()
        self._jobs: dict[str, QueueJob] = {}
        self._lock = threading.RLock()
        self._load(enqueue_pending=True)

    def _job_to_dict(self, job: QueueJob) -> dict[str, Any]:
        data = asdict(job)
        data["status"] = job.status.value
        return data

    def _load(self, enqueue_pending: bool = True) -> None:
        """Load persisted jobs. Supports current dict format and legacy {jobs: []}."""
        if not self.persist_path.exists():
            return

        try:
            raw = json.loads(self.persist_path.read_text(encoding="utf-8"))
            if isinstance(raw, dict) and "jobs" in raw and isinstance(raw["jobs"], list):
                items = {item["id"]: item for item in raw["jobs"]}
            elif isinstance(raw, dict):
                items = raw
            else:
                logger.error("Invalid jobs persistence shape in %s", self.persist_path)
                return

            for item in items.values():
                status = JobQueueStatus(item.get("status", JobQueueStatus.PENDING.value))
                job = QueueJob(
                    id=item["id"],
                    url=item["url"],
                    config=item.get("config", {}),
                    status=status,
                    result=item.get("result"),
                    created_at=float(item.get("created_at", time.time())),
                    updated_at=float(item.get("updated_at", time.time())),
                )
                self._jobs[job.id] = job
                if enqueue_pending and job.status == JobQueueStatus.PENDING:
                    self._inner.put(job)
            logger.info("Loaded %d jobs from %s", len(self._jobs), self.persist_path)
        except Exception as e:
            logger.error("Failed to load jobs persistence %s: %s", self.persist_path, e)
            self._jobs = {}
            self._inner = queue.Queue()

    def _save(self) -> None:
        """Save jobs to JSON under lock."""
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            data = {job_id: self._job_to_dict(job) for job_id, job in self._jobs.items()}
            temp_file = self.persist_path.with_suffix(self.persist_path.suffix + ".tmp")
            try:
                temp_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
                temp_file.replace(self.persist_path)
            except Exception as e:
                logger.error("Failed to save jobs persistence %s: %s", self.persist_path, e)

    def put(self, job: QueueJob) -> None:
        """Enqueue a new job."""
        with self._lock:
            now = time.time()
            if not job.created_at:
                job.created_at = now
            job.updated_at = now
            job.status = JobQueueStatus(job.status)
            self._jobs[job.id] = job
            if job.status == JobQueueStatus.PENDING:
                self._inner.put(job)
            self._save()
            logger.info("Job %s enqueued", job.id)

    def get(self, block: bool = True, timeout: float | None = None) -> QueueJob:
        """Dequeue a pending job and mark it RUNNING."""
        job = self._inner.get(block=block, timeout=timeout)
        with self._lock:
            job.status = JobQueueStatus.RUNNING
            job.updated_at = time.time()
            self._save()
        return job

    def task_done(self) -> None:
        self._inner.task_done()

    def join(self) -> None:
        self._inner.join()

    def qsize(self) -> int:
        return self._inner.qsize()

    def update_status(
        self, job_id: str, status: JobQueueStatus, result: dict[str, Any] | None = None
    ) -> None:
        """Update a job's status and optional result payload."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                logger.warning("Ignoring status update for unknown job: %s", job_id)
                return
            job.status = JobQueueStatus(status)
            job.updated_at = time.time()
            if result is not None:
                job.result = result
            self._save()

    def mark_running(self, job_id: str) -> None:
        self.update_status(job_id, JobQueueStatus.RUNNING)

    def mark_done(self, job_id: str, result: dict[str, Any] | None = None) -> None:
        self.update_status(job_id, JobQueueStatus.DONE, result=result)

    def mark_failed(self, job_id: str, result: dict[str, Any] | None = None) -> None:
        self.update_status(job_id, JobQueueStatus.FAILED, result=result)

    def get_job(self, job_id: str) -> QueueJob | None:
        with self._lock:
            return self._jobs.get(job_id)

    def get_status(self, job_id: str) -> JobQueueStatus | None:
        job = self.get_job(job_id)
        return job.status if job is not None else None

    def get_result(self, job_id: str) -> dict[str, Any] | None:
        job = self.get_job(job_id)
        return job.result if job is not None else None

    def list_jobs(self, status: JobQueueStatus | None = None) -> list[QueueJob] | list[tuple[str, JobQueueStatus]]:
        """Return job snapshot.

        Without a filter returns QueueJob objects. With a status filter returns
        lightweight (job_id, status) tuples used by integration tests/UI tables.
        """
        with self._lock:
            if status is None:
                return list(self._jobs.values())
            status = JobQueueStatus(status)
            return [(job.id, job.status) for job in self._jobs.values() if job.status == status]

    def pending_count(self) -> int:
        with self._lock:
            return sum(1 for job in self._jobs.values() if job.status == JobQueueStatus.PENDING)

    def startup_requeue_pending(self) -> int:
        """Rebuild the in-memory queue from persisted PENDING jobs."""
        with self._lock:
            self._inner = queue.Queue()
            count = 0
            for job in self._jobs.values():
                if job.status == JobQueueStatus.PENDING:
                    self._inner.put(job)
                    count += 1
            return count

    def shutdown(self) -> None:
        self._save()


__all__ = ["JobQueue", "JobQueueStatus", "QueueJob"]
