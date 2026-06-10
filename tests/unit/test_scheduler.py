from apscheduler.schedulers.background import BackgroundScheduler
import pytest

from scheduler import AppScheduler

def test_scheduler_initialization():
    app_scheduler = AppScheduler()
    assert isinstance(app_scheduler.scheduler, BackgroundScheduler)
    assert not app_scheduler.scheduler.running

def test_scheduler_start_stop():
    app_scheduler = AppScheduler()
    app_scheduler.start()
    assert app_scheduler.scheduler.running
    app_scheduler.stop()
    assert not app_scheduler.scheduler.running

def test_add_scheduled_job():
    app_scheduler = AppScheduler()
    app_scheduler.start()
    
    job_id = app_scheduler.add_scheduled_job(
        cron_expr="0 3 * * *",
        url="https://youtube.com/watch?v=123",
        config={}
    )
    
    assert job_id is not None
    job = app_scheduler.scheduler.get_job(job_id)
    assert job is not None
    
    app_scheduler.stop()

def test_remove_scheduled_job():
    app_scheduler = AppScheduler()
    app_scheduler.start()
    
    job_id = app_scheduler.add_scheduled_job(
        cron_expr="0 3 * * *",
        url="https://youtube.com/watch?v=123",
        config={}
    )
    app_scheduler.remove_scheduled_job(job_id)
    
    job = app_scheduler.scheduler.get_job(job_id)
    assert job is None
    
    app_scheduler.stop()
