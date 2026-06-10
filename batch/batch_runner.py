"""Batch runner — background thread consuming the job queue.

Pulls `QueueJob` items, transforms them into `JobConfig`, runs the
orchestrator pipeline, and updates the queue status.
"""

from __future__ import annotations

import queue
import threading

from batch.job_queue import JobQueue, JobQueueStatus, QueueJob
from pipeline.orchestrator import JobConfig, run_job
from utils.logger import logger


class BatchRunner:
    """Background worker that processes jobs from a JobQueue."""

    def __init__(self, job_queue: JobQueue) -> None:
        self.queue = job_queue
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Start the runner thread and re-queue persisted PENDING jobs."""
        if self._thread is not None and self._thread.is_alive():
            logger.warning("BatchRunner is already running.")
            return

        requeued = self.queue.startup_requeue_pending()
        if requeued:
            logger.info("BatchRunner startup re-queued %d pending job(s).", requeued)

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop,
            name="BatchRunnerThread",
            daemon=True,
        )
        self._thread.start()
        logger.info("BatchRunner started.")

    def join(self, timeout: float | None = None) -> None:
        """Wait for all currently queued jobs to finish."""
        self.queue.join()
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    def stop(self) -> None:
        """Signal the runner to stop and wait for it."""
        if self._thread is None or not self._thread.is_alive():
            return
            
        logger.info("BatchRunner stopping...")
        self._stop_event.set()
        self._thread.join(timeout=5.0)
        logger.info("BatchRunner stopped.")

    def _run_loop(self) -> None:
        """Continuously pull and process jobs until stopped."""
        while not self._stop_event.is_set():
            try:
                # Use a timeout to allow checking the stop event
                job: QueueJob = self.queue.get(block=True, timeout=1.0)
            except queue.Empty:
                continue

            logger.info("BatchRunner picked up job: %s", job.id)
            try:
                self._process_job(job)
            except Exception as e:
                logger.exception("BatchRunner encountered unhandled error on job %s: %s", job.id, e)
                self.queue.update_status(job.id, JobQueueStatus.FAILED)
            finally:
                self.queue.task_done()

    def _process_job(self, job: QueueJob) -> None:
        """Convert a QueueJob to JobConfig, run the pipeline, and store result."""
        # Convert queue job payload to orchestrator job
        job_config = JobConfig(
            id=job.id,
            url=job.url,
            config=job.config,
            # max_clips and subtitle_lang are optional, fallback to defaults or parse from config
            max_clips=job.config.get("max_clips", 5),
            subtitle_lang=job.config.get("subtitle_lang", "en"),
        )

        self.queue.mark_running(job.id)

        try:
            result = run_job(job_config)

            # Map JobResult to dict for storage
            result_dict = {
                "id": result.id,
                "status": result.status.value,
                "clips": result.clips,
                "error": result.error,
                "data_file": result.data_file,
            }

            if result.status.value == "COMPLETED":
                self.queue.update_status(job.id, JobQueueStatus.DONE, result=result_dict)
                logger.info("BatchRunner completed job: %s", job.id)
            else:
                self.queue.update_status(job.id, JobQueueStatus.FAILED, result=result_dict)
                logger.error("BatchRunner failed job %s: %s", job.id, result.error)
                
        except Exception as e:
            logger.exception("Pipeline failed for job %s: %s", job.id, e)
            self.queue.update_status(job.id, JobQueueStatus.FAILED, result={"error": str(e)})

