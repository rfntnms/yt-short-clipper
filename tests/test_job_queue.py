"""Tests for batch.job_queue — JobQueue, JobQueueStatus, QueueJob."""

from __future__ import annotations

import json
import queue
import threading
import time
from pathlib import Path

import pytest

from batch.job_queue import JobQueue, JobQueueStatus, QueueJob


@pytest.fixture
def tmp_persist(tmp_path: Path) -> Path:
    """Return a temporary persistence path."""
    return tmp_path / "jobs.json"


@pytest.fixture
def jq(tmp_persist: Path) -> JobQueue:
    """Return a JobQueue with a temporary persistence path."""
    return JobQueue(persist_path=tmp_persist)


# ── QueueJob dataclass ──────────────────────────────────────────────────────

class TestQueueJob:
    def test_defaults(self):
        job = QueueJob(id="abc", url="https://youtu.be/test", config={"k": "v"})
        assert job.status == JobQueueStatus.PENDING
        assert job.result is None
        assert isinstance(job.created_at, float)
        assert isinstance(job.updated_at, float)

    def test_explicit_status(self):
        job = QueueJob(id="x", url="u", config={}, status=JobQueueStatus.RUNNING)
        assert job.status == JobQueueStatus.RUNNING


# ── JobQueueStatus enum ─────────────────────────────────────────────────────

class TestJobQueueStatus:
    def test_values(self):
        assert JobQueueStatus.PENDING.value == "PENDING"
        assert JobQueueStatus.RUNNING.value == "RUNNING"
        assert JobQueueStatus.DONE.value == "DONE"
        assert JobQueueStatus.FAILED.value == "FAILED"

    def test_str_enum(self):
        """JobQueueStatus is a str enum — usable as dict key directly."""
        d = {JobQueueStatus.PENDING: 1}
        assert d["PENDING"] == 1


# ── JobQueue core ───────────────────────────────────────────────────────────

class TestJobQueueCore:
    def test_put_and_get(self, jq: JobQueue, tmp_persist: Path):
        job = QueueJob(id="j1", url="https://youtu.be/1", config={})
        jq.put(job)
        assert jq.qsize() == 1

        pulled = jq.get(timeout=1.0)
        assert pulled.id == "j1"
        jq.task_done()

    def test_put_persists(self, jq, tmp_persist):
        job = QueueJob(id="j_persist", url="https://youtu.be/p", config={"a": 1})
        jq.put(job)
        # File should exist and contain the job
        data = json.loads(tmp_persist.read_text())
        assert "j_persist" in data
        assert data["j_persist"]["url"] == "https://youtu.be/p"
        assert data["j_persist"]["status"] == "PENDING"

    def test_get_timeout_on_empty(self, jq):
        with pytest.raises(queue.Empty):
            jq.get(block=True, timeout=0.1)

    def test_order_is_fifo(self, jq):
        jq.put(QueueJob(id="a", url="u", config={}))
        jq.put(QueueJob(id="b", url="u", config={}))
        assert jq.get(timeout=0.5).id == "a"
        assert jq.get(timeout=0.5).id == "b"
        jq.task_done()
        jq.task_done()


# ── Status transitions ──────────────────────────────────────────────────────

class TestStatusTransitions:
    def test_update_status(self, jq, tmp_persist):
        job = QueueJob(id="s1", url="u", config={})
        jq.put(job)
        jq.update_status("s1", JobQueueStatus.RUNNING)

        status = jq.get_status("s1")
        assert status == JobQueueStatus.RUNNING

        # Persistence should reflect new status
        data = json.loads(tmp_persist.read_text())
        assert data["s1"]["status"] == "RUNNING"
        assert data["s1"]["updated_at"] > data["s1"]["created_at"]

    def test_update_status_with_result(self, jq):
        jq.put(QueueJob(id="s2", url="u", config={}))
        jq.update_status("s2", JobQueueStatus.DONE, result={"clips": ["clip.mp4"]})

        result = jq.get_result("s2")
        assert result == {"clips": ["clip.mp4"]}

    def test_update_unknown_job_logs_no_crash(self, jq):
        # Should not raise
        jq.update_status("nonexistent", JobQueueStatus.RUNNING)

    def test_shorthand_methods(self, jq):
        jq.put(QueueJob(id="sh", url="u", config={}))

        jq.mark_running("sh")
        assert jq.get_status("sh") == JobQueueStatus.RUNNING

        jq.mark_done("sh", result={"ok": True})
        assert jq.get_status("sh") == JobQueueStatus.DONE
        assert jq.get_result("sh") == {"ok": True}

        jq2_id = "sh2"
        jq.put(QueueJob(id=jq2_id, url="u", config={}))
        jq.mark_failed(jq2_id, result={"error": "boom"})
        assert jq.get_status(jq2_id) == JobQueueStatus.FAILED
        assert jq.get_result(jq2_id) == {"error": "boom"}

    def test_full_lifecycle(self, jq):
        jq.put(QueueJob(id="life", url="u", config={}))
        assert jq.get_status("life") == JobQueueStatus.PENDING

        jq.mark_running("life")
        assert jq.get_status("life") == JobQueueStatus.RUNNING

        jq.mark_done("life")
        assert jq.get_status("life") == JobQueueStatus.DONE

    def test_get_job_and_get_status(self, jq):
        jq.put(QueueJob(id="q1", url="https://youtu.be/x", config={"k": "v"}))
        job = jq.get_job("q1")
        assert job is not None
        assert job.url == "https://youtu.be/x"
        assert jq.get_status("q1") == JobQueueStatus.PENDING

    def test_get_job_none_for_unknown(self, jq):
        assert jq.get_job("nope") is None
        assert jq.get_status("nope") is None
        assert jq.get_result("nope") is None


# ── Listing and counting ────────────────────────────────────────────────────

class TestListing:
    def test_list_jobs(self, jq):
        jq.put(QueueJob(id="l1", url="u", config={}))
        jq.put(QueueJob(id="l2", url="u", config={}))
        jobs = jq.list_jobs()
        ids = {j.id for j in jobs}
        assert ids == {"l1", "l2"}

    def test_pending_count(self, jq):
        jq.put(QueueJob(id="p1", url="u", config={}))
        jq.put(QueueJob(id="p2", url="u", config={}))
        assert jq.pending_count() == 2

        jq.mark_running("p1")
        assert jq.pending_count() == 1

        jq.mark_done("p1")
        assert jq.pending_count() == 1  # p2 still pending


# ── Persistence round-trip ──────────────────────────────────────────────────

class TestPersistence:
    def test_load_pending_on_init(self, tmp_persist):
        """Pre-populate the JSON, then init — PENDING jobs should be loaded."""
        job_data = {
            "old_job": {
                "id": "old_job",
                "url": "https://youtu.be/old",
                "config": {},
                "status": "PENDING",
                "result": None,
                "created_at": 1.0,
                "updated_at": 2.0,
            }
        }
        tmp_persist.parent.mkdir(parents=True, exist_ok=True)
        tmp_persist.write_text(json.dumps(job_data))

        jq2 = JobQueue(persist_path=tmp_persist)
        job = jq2.get_job("old_job")
        assert job is not None
        assert job.status == JobQueueStatus.PENDING

    def test_skip_non_pending_on_init(self, tmp_persist):
        """RUNNING/DONE/FAILED jobs are loaded but not re-queued."""
        job_data = {
            "done_job": {
                "id": "done_job",
                "url": "u",
                "config": {},
                "status": "DONE",
                "result": {"ok": True},
                "created_at": 1.0,
                "updated_at": 2.0,
            }
        }
        tmp_persist.parent.mkdir(parents=True, exist_ok=True)
        tmp_persist.write_text(json.dumps(job_data))

        jq2 = JobQueue(persist_path=tmp_persist)
        assert jq2.get_job("done_job") is not None
        assert jq2.pending_count() == 0

    def test_startup_requeue_pending(self, tmp_persist):
        """Re-enqueue PENDING jobs loaded from disk."""
        job_data = {
            "req1": {
                "id": "req1",
                "url": "https://youtu.be/r1",
                "config": {},
                "status": "PENDING",
                "result": None,
                "created_at": 1.0,
                "updated_at": 1.0,
            },
            "req2": {
                "id": "req2",
                "url": "https://youtu.be/r2",
                "config": {},
                "status": "PENDING",
                "result": None,
                "created_at": 2.0,
                "updated_at": 2.0,
            },
        }
        tmp_persist.parent.mkdir(parents=True, exist_ok=True)
        tmp_persist.write_text(json.dumps(job_data))

        jq2 = JobQueue(persist_path=tmp_persist)
        count = jq2.startup_requeue_pending()
        assert count == 2

        # Should be pullable
        j1 = jq2.get(timeout=0.5)
        j2 = jq2.get(timeout=0.5)
        assert {j1.id, j2.id} == {"req1", "req2"}

    def test_corrupted_json_recovers(self, tmp_persist):
        """Corrupted JSON should not crash init — starts empty."""
        tmp_persist.parent.mkdir(parents=True, exist_ok=True)
        tmp_persist.write_text("not valid json {{{")

        jq2 = JobQueue(persist_path=tmp_persist)
        assert jq2.list_jobs() == []

    def test_shutdown_persists_pending(self, jq, tmp_persist):
        """Shutdown should write all PENDING jobs to disk."""
        jq.put(QueueJob(id="sd1", url="u", config={}))
        jq.put(QueueJob(id="sd2", url="u", config={}))
        jq.mark_done("sd1")
        # sd2 is still PENDING

        jq.shutdown()

        data = json.loads(tmp_persist.read_text())
        assert "sd1" in data
        assert "sd2" in data
        assert data["sd2"]["status"] == "PENDING"


# ── Thread safety ───────────────────────────────────────────────────────────

class TestThreadSafety:
    def test_concurrent_put_and_status_updates(self, jq):
        """Multiple threads putting + updating should not crash."""
        errors: list[Exception] = []

        def producer(n: int):
            try:
                for i in range(5):
                    jq.put(QueueJob(id=f"t{n}_{i}", url="u", config={}))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=producer, args=(n,)) for n in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)
        assert not errors
        assert len(jq.list_jobs()) == 20
