"""Tests for scheduler.py — cron parsing, job lifecycle, scheduler control."""

from __future__ import annotations

import pytest
import scheduler
from scheduler import (
    _parse_cron_expr,
    add_scheduled_job,
    list_scheduled_jobs,
    remove_scheduled_job,
    shutdown,
    start,
)


class TestParseCronExpr:
    def test_six_fields(self):
        result = _parse_cron_expr("0 0 9 * * *")
        assert result == {
            "second": "0",
            "minute": "0",
            "hour": "9",
            "day": "*",
            "month": "*",
            "day_of_week": "*",
        }

    def test_five_fields(self):
        result = _parse_cron_expr("0 9 * * *")
        assert result == {
            "minute": "0",
            "hour": "9",
            "day": "*",
            "month": "*",
            "day_of_week": "*",
        }

    def test_complex_cron(self):
        result = _parse_cron_expr("0 30 8-17/2 * * 1-5")
        assert result["minute"] == "30"
        assert result["hour"] == "8-17/2"
        assert result["day_of_week"] == "1-5"

    def test_reject_fewer_than_five(self):
        with pytest.raises(ValueError, match="5 or 6 fields"):
            _parse_cron_expr("1 2 3")

    def test_reject_seven_fields(self):
        with pytest.raises(ValueError, match="5 or 6 fields"):
            _parse_cron_expr("0 0 9 * * * *")


class TestSchedulerLifecycle:
    def setup_method(self):
        if scheduler._scheduler:
            scheduler.shutdown(wait=False)
            scheduler._scheduler = None

    def teardown_method(self):
        if scheduler._scheduler:
            scheduler.shutdown(wait=False)
            scheduler._scheduler = None

    def test_add_and_remove_job(self):
        """Add a job, verify it shows in list, remove it, verify empty."""
        jid = add_scheduled_job(
            "0 0 9 * * *",
            "https://youtube.com/watch?v=test",
            {"output_dir": "output"},
            job_id="ut-test-add-remove",
        )
        assert jid == "ut-test-add-remove"

        jobs = list_scheduled_jobs()
        ids = [j["id"] for j in jobs]
        assert jid in ids

        removed = remove_scheduled_job(jid)
        assert removed is True

        jobs_after = list_scheduled_jobs()
        assert jid not in [j["id"] for j in jobs_after]

    def test_remove_nonexistent_job(self):
        """Removing a job that doesn't exist returns False without raising."""
        result = remove_scheduled_job("no-such-job-xyz")
        assert result is False

    def test_add_with_default_job_id(self):
        """Without job_id, APScheduler auto-generates a hex id."""
        jid = add_scheduled_job(
            "30 */2 * * *",
            "https://youtu.be/auto-id",
            {"output_dir": "output"},
        )
        assert jid and len(jid) > 0
        assert list_scheduled_jobs()
        remove_scheduled_job(jid)

    def test_job_pending_flag(self):
        """Jobs added before start() are reported as pending."""
        jid = add_scheduled_job(
            "0 0 */3 * * *",
            "https://youtu.be/pending-test",
            {},
            job_id="ut-pending-test",
        )
        jobs = list_scheduled_jobs()
        match = next(j for j in jobs if j["id"] == jid)
        assert match["pending"] is True
        assert match["next_run_time"] is None
        remove_scheduled_job(jid)

    def test_start_and_shutdown(self):
        """start() and shutdown() are idempotent and don't raise."""
        # Should not raise
        start()
        # Starting again should be a no-op
        start()
        shutdown(wait=False)
        # Second shutdown should be safe
        shutdown(wait=False)


class TestSchedulerIntegration:
    """Integration tests that exercise the started scheduler."""

    def test_job_shows_next_run_time_after_start(self):
        """When scheduler is running, next_run_time should be populated."""
        jid = add_scheduled_job(
            "0 0 9 * * *",
            "https://youtu.be/next-run-test",
            {},
            job_id="ut-next-run-test",
        )
        start()

        try:
            jobs = list_scheduled_jobs()
            match = next(j for j in jobs if j["id"] == jid)
            # Once the scheduler is started, next_run_time should be set
            assert match["next_run_time"] is not None, (
                f"Expected next_run_time for started scheduler, got None. "
                f"Full jobs list: {jobs}"
            )
            assert match["pending"] is False
        finally:
            remove_scheduled_job(jid)
            shutdown(wait=False)

    def test_multiple_jobs(self):
        """Multiple scheduled jobs are all listed."""
        ids = []
        for i in range(3):
            jid = add_scheduled_job(
                f"0 {i} * * *",
                f"https://youtu.be/multi-{i}",
                {},
                job_id=f"ut-multi-{i}",
            )
            ids.append(jid)

        jobs = [j for j in list_scheduled_jobs() if j["id"] in ids]
        assert len(jobs) == 3
        returned_ids = {j["id"] for j in jobs}
        assert set(ids) == returned_ids

        for jid in ids:
            remove_scheduled_job(jid)

    def test_replace_existing(self):
        """Adding with same job_id replaces the existing trigger."""
        start()
        try:
            add_scheduled_job("0 0 9 * * *", "https://youtu.be/old", {}, job_id="ut-replace")
            add_scheduled_job("30 10 * * *", "https://youtu.be/new", {}, job_id="ut-replace")

            jobs = list_scheduled_jobs()
            match = next(j for j in jobs if j["id"] == "ut-replace")
            # The trigger should reflect the new cron expression
            assert "30" in match["trigger"], f"Expected minute=30 in trigger, got: {match['trigger']}"
            assert "10" in match["trigger"], f"Expected hour=10 in trigger, got: {match['trigger']}"
        finally:
            remove_scheduled_job("ut-replace")
            shutdown(wait=False)
