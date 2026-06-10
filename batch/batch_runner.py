import queue
import threading
import logging
import json
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)

class BatchRunner:
    def __init__(self, output_dir: str = "output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.job_queue = queue.Queue()
        self.jobs: Dict[str, dict] = {}
        self.worker_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._load_jobs()

    def _load_jobs(self):
        jobs_file = self.output_dir / "jobs.json"
        if jobs_file.exists():
            try:
                with open(jobs_file, "r") as f:
                    self.jobs = json.load(f)
                    for job_id, job in self.jobs.items():
                        if job.get("status") == "PENDING":
                            from pipeline.orchestrator import JobConfig
                            self.job_queue.put(JobConfig(url=job["url"], job_id=job_id, config=job.get("config", {})))
            except Exception as e:
                logger.error(f"Failed to load jobs: {e}")

    def _save_jobs(self):
        jobs_file = self.output_dir / "jobs.json"
        try:
            with open(jobs_file, "w") as f:
                json.dump(self.jobs, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save jobs: {e}")

    def submit(self, job_config) -> str:
        if not hasattr(job_config, 'job_id') or not job_config.job_id:
            import uuid
            job_config.job_id = str(uuid.uuid4())

        self.jobs[job_config.job_id] = {
            "id": job_config.job_id,
            "url": job_config.url,
            "status": "PENDING",
            "progress": 0.0,
            "config": job_config.config
        }
        self._save_jobs()
        self.job_queue.put(job_config)
        return job_config.job_id

    def start(self):
        if self.worker_thread is None or not self.worker_thread.is_alive():
            self._stop_event.clear()
            self.worker_thread = threading.Thread(target=self._run_loop, daemon=True)
            self.worker_thread.start()
            logger.info("BatchRunner started")

    def stop(self):
        self._stop_event.set()
        if self.worker_thread:
            self.worker_thread.join(timeout=2.0)
            logger.info("BatchRunner stopped")

    def _run_loop(self):
        from pipeline.orchestrator import run_job_streaming

        while not self._stop_event.is_set():
            try:
                job_config = self.job_queue.get(timeout=1.0)
                job_id = job_config.job_id

                self.jobs[job_id]["status"] = "RUNNING"
                self._save_jobs()

                try:
                    for status in run_job_streaming(job_config):
                        self.jobs[job_id]["status"] = status.status
                        self.jobs[job_id]["progress"] = status.progress
                        self._save_jobs()
                        if self._stop_event.is_set():
                            break
                except Exception as e:
                    self.jobs[job_id]["status"] = "FAILED"
                    self.jobs[job_id]["error"] = str(e)
                    self._save_jobs()
                finally:
                    self.job_queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in batch runner loop: {e}")
