import pytest
from batch.batch_runner import BatchRunner
from pipeline.orchestrator import JobConfig
import tempfile
import threading
import time

def test_batch_runner_submit():
    with tempfile.TemporaryDirectory() as tmpdir:
        runner = BatchRunner(output_dir=tmpdir)
        job_config = JobConfig(url="http://test.com", job_id="test1")

        job_id = runner.submit(job_config)

        assert job_id == "test1"
        assert runner.jobs["test1"]["status"] == "PENDING"
        assert runner.job_queue.qsize() == 1

def test_batch_runner_start_stop():
    with tempfile.TemporaryDirectory() as tmpdir:
        runner = BatchRunner(output_dir=tmpdir)
        runner.start()
        assert runner.worker_thread.is_alive()
        runner.stop()
        assert not runner.worker_thread.is_alive()
