import logging
from datetime import timedelta

from celery import shared_task
from django.db import transaction
from django.db.models import Q
from django.utils import timezone

from .models import AutomationRun, ScheduledSearch

logger = logging.getLogger(__name__)
MIN_RESYNC = timedelta(minutes=30)  # several users sharing a source do not trigger duplicate syncs
STALE_RUN = timedelta(minutes=30)


@shared_task
def dispatch_due_searches():
    from apps.jobs.models import JobSource
    from apps.jobs.tasks import sync_source_task

    now = timezone.now()
    queued: set[int] = set()
    with transaction.atomic():
        due = list(ScheduledSearch.objects.select_for_update(skip_locked=True).filter(enabled=True, next_run_at__lte=now))
        for search in due:
            sources = search.sources.filter(enabled=True)
            if not sources.exists():
                sources = JobSource.objects.filter(enabled=True)
            for source in sources.exclude(status=JobSource.Status.DISABLED):
                if source.pk in queued:
                    continue
                if source.last_synced_at and source.last_synced_at > now - MIN_RESYNC:
                    continue
                queued.add(source.pk)
                transaction.on_commit(lambda pk=source.pk, uid=search.user_id: sync_source_task.delay(pk, uid))
            search.schedule_next(now)
            search.save(update_fields=["last_run_at", "next_run_at", "updated_at"])
    return {"searches": len(due), "sources_queued": len(queued)}


@shared_task
def recover_interrupted_runs():
    """Mark runs whose worker died (no heartbeat) as interrupted, and release stuck submissions."""
    from apps.applications.services.submission import recover_stuck_submissions

    cutoff = timezone.now() - STALE_RUN
    count = AutomationRun.objects.filter(status=AutomationRun.Status.RUNNING, heartbeat_at__lt=cutoff).update(
        status=AutomationRun.Status.INTERRUPTED, finished_at=timezone.now(), error="Worker stopped before completion"
    )
    released = recover_stuck_submissions()
    return {"interrupted_runs": count, "released_submissions": released}


@shared_task
def purge_expired_data():
    """Apply per-user retention: delete old generated documents and closed applications."""
    from apps.applications.models import Application
    from apps.documents.models import GeneratedDocument
    from apps.profiles.models import CandidateProfile

    deleted_docs = deleted_apps = 0
    for profile in CandidateProfile.objects.filter(data_retention_days__gt=0).select_related("user"):
        cutoff = timezone.now() - timedelta(days=profile.data_retention_days)
        old_docs = GeneratedDocument.objects.filter(user=profile.user, created_at__lt=cutoff).filter(
            resume_applications__isnull=True)
        for doc in old_docs:
            doc.file.delete(save=False)
            doc.delete()
            deleted_docs += 1
        closed = Application.objects.filter(user=profile.user, updated_at__lt=cutoff).filter(
            Q(status__in=Application.TERMINAL))
        deleted_apps += closed.count()
        closed.delete()
    AutomationRun.objects.filter(started_at__lt=timezone.now() - timedelta(days=90)).delete()
    return {"documents": deleted_docs, "applications": deleted_apps}
