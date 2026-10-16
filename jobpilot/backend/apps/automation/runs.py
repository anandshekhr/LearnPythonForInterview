"""Context manager that records background work in AutomationRun."""
import contextlib
import logging

from django.utils import timezone

from .models import AutomationRun

logger = logging.getLogger(__name__)


class _Run:
    def __init__(self, record: AutomationRun):
        self.record = record
        self.stats = {}
        self._failed = None
        self._skipped = None

    def fail(self, message: str):
        self._failed = message

    def skip(self, message: str):
        self._skipped = message

    def heartbeat(self):
        AutomationRun.objects.filter(pk=self.record.pk).update(heartbeat_at=timezone.now())


@contextlib.contextmanager
def track_run(kind, user=None, task_id="", target=None):
    record = AutomationRun.objects.create(
        user=user if getattr(user, "pk", None) else None,
        kind=kind,
        task_id=task_id or "",
        target_type=target.__class__.__name__ if target is not None else "",
        target_id=str(target.pk) if target is not None else "",
    )
    run = _Run(record)
    try:
        yield run
    except Exception as exc:
        AutomationRun.objects.filter(pk=record.pk).update(
            status=AutomationRun.Status.FAILED, finished_at=timezone.now(), error=f"{type(exc).__name__}: {str(exc)[:900]}",
            stats=run.stats,
        )
        raise
    else:
        status = AutomationRun.Status.SUCCEEDED
        error = ""
        if run._failed:
            status, error = AutomationRun.Status.FAILED, run._failed
        elif run._skipped:
            status, error = AutomationRun.Status.SKIPPED, run._skipped
        AutomationRun.objects.filter(pk=record.pk).update(
            status=status, finished_at=timezone.now(), error=error[:1000], stats=run.stats
        )
