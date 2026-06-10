import apscheduler
print(f"APScheduler version: {apscheduler.__version__}")

from apscheduler.schedulers.background import BackgroundScheduler
s = BackgroundScheduler()
s.add_job(lambda: None, trigger="cron", minute="0", id="probe")
jobs = s.get_jobs()
if jobs:
    j = jobs[0]
    print(f"Job id: {j.id}")
    print(f"Has next_run_time attr: {hasattr(j, 'next_run_time')}")
    print(f"Has _next_run_time attr: {hasattr(j, '_next_run_time')}")
    # Try common access patterns
    try:
        print(f"next_run_time: {j.next_run_time}")
    except Exception as e:
        print(f"next_run_time error: {e}")
    try:
        print(f"_next_run_time: {j._next_run_time}")
    except Exception as e:
        print(f"_next_run_time error: {e}")
s.shutdown(wait=False)
