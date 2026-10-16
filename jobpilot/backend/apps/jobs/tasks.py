import logging
from datetime import timedelta

from celery import shared_task
from django.conf import settings
from django.db.models import F, Q
from django.utils import timezone

from apps.automation.runs import track_run

from .http import SourceHTTPError
from .models import JobPosting, JobSource
from .services.requirements import analyze_job
from .services.sync import expire_job, sync_source

logger = logging.getLogger(__name__)


@shared_task(bind=True, max_retries=3, acks_late=True)
def sync_source_task(self, source_id: int, user_id: int | None = None):
    from django.contrib.auth import get_user_model

    source = JobSource.objects.select_related("company").filter(pk=source_id, enabled=True).first()
    if source is None:
        return {"status": "skipped"}
    user = get_user_model().objects.filter(pk=user_id).first() if user_id else None
    with track_run("source_sync", user=user, task_id=self.request.id, target=source) as run:
        try:
            result = sync_source(source)
        except SourceHTTPError as exc:
            run.fail(str(exc))
            if exc.retryable and self.request.retries < self.max_retries:
                raise self.retry(countdown=60 * (2 ** self.request.retries))
            _notify_source_failure(source, str(exc))
            return {"status": "failed", "error": str(exc)}
        run.stats = result.as_stats()
    for job_id in result.created + result.updated + result.reactivated:
        process_job.delay(job_id)
    if result.expired:
        notify_expired_jobs.delay(result.expired)
    return {"status": "ok", **result.as_stats()}


def _notify_source_failure(source, message):
    from apps.notifications.models import Notification
    from apps.notifications.service import notify

    source.refresh_from_db()
    if source.created_by and source.status == JobSource.Status.ERROR:
        notify(source.created_by, Notification.Kind.SOURCE_ERROR, f"Job source failing: {source.company.name}",
               body=message[:500], link="/sources", obj=source,
               dedupe_key=f"source_error:{source.pk}:{source.last_error_at:%Y%m%d}")


@shared_task(bind=True, autoretry_for=(OSError,), retry_backoff=True, retry_kwargs={"max_retries": 3}, acks_late=True)
def process_job(self, job_id: int):
    """Extract requirements + embedding, then (re)score the job for every user."""
    job = JobPosting.objects.select_related("company").filter(pk=job_id).first()
    if job is None:
        return
    analyze_job(job)
    from apps.matching.tasks import match_job

    match_job.delay(job.pk)


@shared_task
def notify_expired_jobs(job_ids):
    """Tell users when a job they shortlisted or applied to is no longer published."""
    from apps.applications.models import Application
    from apps.matching.models import JobMatch
    from apps.notifications.models import Notification
    from apps.notifications.service import notify

    jobs = JobPosting.objects.filter(pk__in=job_ids).select_related("company")
    for job in jobs:
        users = set(JobMatch.objects.filter(job=job, user_state=JobMatch.UserState.SHORTLISTED).values_list("user_id", flat=True))
        users |= set(Application.objects.filter(job=job).exclude(status__in=Application.TERMINAL).values_list("user_id", flat=True))
        from django.contrib.auth import get_user_model

        for user in get_user_model().objects.filter(pk__in=users):
            notify(user, Notification.Kind.JOB_EXPIRED, f"Posting closed: {job.title} at {job.company.name}",
                   link=f"/jobs/{job.pk}", obj=job, dedupe_key=f"job_expired:{job.pk}:{job.expired_at:%Y%m%d}")


@shared_task
def sync_due_sources():
    now = timezone.now()
    due = JobSource.objects.filter(enabled=True).exclude(status=JobSource.Status.DISABLED).filter(
        Q(last_synced_at__isnull=True) | Q(last_synced_at__lte=now - timedelta(minutes=1) * F("sync_interval_minutes"))
    )
    ids = list(due.values_list("pk", flat=True))
    for pk in ids:
        sync_source_task.delay(pk)
    return len(ids)


@shared_task
def check_sources_health():
    """Flag sources with no successful sync for 3x their interval."""
    now = timezone.now()
    flagged = 0
    for source in JobSource.objects.filter(enabled=True, last_success_at__isnull=False):
        if source.last_success_at < now - timedelta(minutes=3 * source.sync_interval_minutes) and source.status == JobSource.Status.ACTIVE:
            JobSource.objects.filter(pk=source.pk).update(status=JobSource.Status.DEGRADED,
                                                         last_error="No successful sync within 3 intervals")
            flagged += 1
    return flagged


@shared_task
def expire_stale_jobs():
    """Safety net: jobs not seen for JOB_EXPIRY_GRACE_DAYS although their source synced successfully since."""
    now = timezone.now()
    cutoff = now - timedelta(days=settings.JOB_EXPIRY_GRACE_DAYS)
    stale = JobPosting.objects.filter(status=JobPosting.Status.ACTIVE, last_seen_at__lt=cutoff,
                                      source__last_success_at__gt=F("last_seen_at"))
    ids = []
    for job in stale:
        expire_job(job, now)
        ids.append(job.pk)
    if ids:
        notify_expired_jobs.delay(ids)
    return len(ids)
