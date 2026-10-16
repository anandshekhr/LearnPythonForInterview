from celery import shared_task

from apps.automation.runs import track_run
from apps.jobs.models import JobPosting

from . import services


@shared_task(bind=True, autoretry_for=(OSError,), retry_backoff=True, retry_kwargs={"max_retries": 3}, acks_late=True)
def match_job(self, job_id: int):
    job = JobPosting.objects.select_related("company").filter(pk=job_id).first()
    if job is None:
        return 0
    return services.match_job_for_all_users(job)


@shared_task(bind=True, acks_late=True)
def rematch_user(self, user_id: int, force: bool = False):
    from django.contrib.auth import get_user_model

    user = get_user_model().objects.filter(pk=user_id).first()
    if user is None:
        return 0
    with track_run("rematch", user=user, task_id=self.request.id) as run:
        count = services.match_all_jobs_for_user(user, force=force)
        run.stats = {"matches_updated": count}
        return count
