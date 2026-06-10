import pytest
from unittest.mock import MagicMock, patch
import queue
import time
import threading

from batch.job_queue import JobQueue, JobQueueStatus, QueueJob
from batch.batch_runner import BatchRunner
from pipeline.orchestrator import JobConfig, JobResult, JobStatus


class TestBatchRunner:
    def test_init(self):
        q = JobQueue()
        runner = BatchRunner(q)
        assert runner.queue is q
        assert runner._thread is None
        assert not runner._stop_event.is_set()

    def test_start_stop(self):
        q = JobQueue()
        runner = BatchRunner(q)
        runner.start()
        assert runner._thread is not None
        assert runner._thread.is_alive()
        
        # calling start again should not spawn new thread
        t1 = runner._thread
        runner.start()
        assert runner._thread is t1
        
        runner.stop()
        assert not runner._thread.is_alive()

    def test_stop_without_start(self):
        q = JobQueue()
        runner = BatchRunner(q)
        runner.stop()  # Should not raise exception

    @patch("batch.batch_runner.run_job")
    def test_process_job_success(self, mock_run_job):
        q = JobQueue()
        runner = BatchRunner(q)
        job = QueueJob(id="job_123", url="http://youtube.com/watch?v=123", config={"max_clips": 3})
        q.put(job)
        
        # Pull it so it's "running"
        popped_job = q.get()
        
        # Mock orchestrator response
        mock_result = JobResult(
            id="job_123",
            status=JobStatus.COMPLETED,
            clips=[{"path": "/tmp/clip1.mp4"}],
            data_file="/tmp/data.json"
        )
        mock_run_job.return_value = mock_result
        
        runner._process_job(popped_job)
        
        # Check queue status and result
        assert q.get_status("job_123") == JobQueueStatus.DONE
        result = q.get_result("job_123")
        assert result is not None
        assert result["status"] == "COMPLETED"
        assert len(result["clips"]) == 1

    @patch("batch.batch_runner.run_job")
    def test_process_job_pipeline_failure(self, mock_run_job):
        q = JobQueue()
        runner = BatchRunner(q)
        job = QueueJob(id="job_failed", url="http://youtube.com/watch?v=123", config={})
        q.put(job)
        popped_job = q.get()
        
        # Mock orchestrator failure response
        mock_result = JobResult(
            id="job_failed",
            status=JobStatus.FAILED,
            error="Download failed"
        )
        mock_run_job.return_value = mock_result
        
        runner._process_job(popped_job)
        
        assert q.get_status("job_failed") == JobQueueStatus.FAILED
        result = q.get_result("job_failed")
        assert result is not None
        assert result["error"] == "Download failed"

    @patch("batch.batch_runner.run_job")
    def test_process_job_unhandled_exception(self, mock_run_job):
        q = JobQueue()
        runner = BatchRunner(q)
        job = QueueJob(id="job_exc", url="http://youtube.com/watch?v=123", config={})
        q.put(job)
        popped_job = q.get()
        
        # Orchestrator raises unhandled exception
        mock_run_job.side_effect = RuntimeError("Disk full")
        
        runner._process_job(popped_job)
        
        assert q.get_status("job_exc") == JobQueueStatus.FAILED
        result = q.get_result("job_exc")
        assert result is not None
        assert "Disk full" in result["error"]

    @patch("batch.batch_runner.run_job")
    def test_run_loop_integration(self, mock_run_job):
        q = JobQueue()
        runner = BatchRunner(q)
        
        mock_result = JobResult(
            id="test_id",
            status=JobStatus.COMPLETED
        )
        mock_run_job.return_value = mock_result
        
        job = QueueJob(id="test_id", url="http://youtube.com/watch?v=123", config={})
        q.put(job)
        
        runner.start()
        
        # Wait for queue to be processed
        q.join()
        
        runner.stop()
        
        assert q.get_status("test_id") == JobQueueStatus.DONE
