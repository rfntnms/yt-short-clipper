from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
import logging

logger = logging.getLogger(__name__)

class AppScheduler:
    def __init__(self, batch_runner=None):
        self.scheduler = BackgroundScheduler()
        self.batch_runner = batch_runner

    def start(self):
        if not self.scheduler.running:
            self.scheduler.start()
            logger.info("Scheduler started.")

    def stop(self):
        if self.scheduler.running:
            self.scheduler.shutdown()
            logger.info("Scheduler stopped.")

    def _parse_cron(self, cron_expr: str):
        parts = cron_expr.split()
        if len(parts) != 5:
            raise ValueError(f"Invalid cron expression: {cron_expr}. Expected 5 parts.")
        
        return {
            'minute': parts[0],
            'hour': parts[1],
            'day': parts[2],
            'month': parts[3],
            'day_of_week': parts[4]
        }

    def add_scheduled_job(self, cron_expr: str, url: str, config: dict):
        try:
            cron_kwargs = self._parse_cron(cron_expr)
            job_id = f"scheduled_{url[:20]}"
            
            def run_job():
                if self.batch_runner:
                    # Delaying import to avoid circular dependency
                    from pipeline.orchestrator import JobConfig
                    job_config = JobConfig(url=url, config=config)
                    self.batch_runner.submit(job_config)
                    logger.info(f"Scheduled job submitted for {url}")
            
            job = self.scheduler.add_job(
                func=run_job,
                trigger=CronTrigger(**cron_kwargs),
                id=job_id,
                replace_existing=True
            )
            logger.info(f"Added scheduled job {job_id} with cron '{cron_expr}'")
            return job.id
            
        except Exception as e:
            logger.error(f"Failed to add scheduled job: {e}")
            raise

    def remove_scheduled_job(self, job_id: str):
        try:
            self.scheduler.remove_job(job_id)
            logger.info(f"Removed scheduled job {job_id}")
        except Exception as e:
            logger.error(f"Failed to remove scheduled job {job_id}: {e}")
            raise
