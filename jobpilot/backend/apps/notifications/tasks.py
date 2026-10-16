from datetime import timedelta

from celery import shared_task
from django.utils import timezone

from .models import Notification
from .service import notify


@shared_task
def send_due_reminders():
    """Interview reminders (next 24h) and follow-up reminders (due today or overdue)."""
    from apps.applications.models import FollowUp, Interview

    now = timezone.now()
    sent = 0
    for interview in Interview.objects.select_related("application", "application__user").filter(
        scheduled_at__gte=now, scheduled_at__lte=now + timedelta(hours=24), reminder_sent_at__isnull=True,
        outcome=Interview.Outcome.PENDING,
    ):
        app = interview.application
        if notify(app.user, Notification.Kind.INTERVIEW, f"Interview soon: {app.company_name} – {interview.round_name}",
                  body=f"Scheduled at {timezone.localtime(interview.scheduled_at):%d %b %Y %H:%M}",
                  link=f"/applications/{app.pk}", obj=interview, dedupe_key=f"interview:{interview.pk}"):
            sent += 1
        Interview.objects.filter(pk=interview.pk).update(reminder_sent_at=now)
    for follow_up in FollowUp.objects.select_related("application", "application__user").filter(
        due_date__lte=timezone.localdate(), completed_at__isnull=True, reminder_sent_at__isnull=True,
    ):
        app = follow_up.application
        if notify(app.user, Notification.Kind.FOLLOW_UP, f"Follow up: {app.job_title} at {app.company_name}",
                  body=follow_up.note, link=f"/applications/{app.pk}", obj=follow_up, dedupe_key=f"follow_up:{follow_up.pk}"):
            sent += 1
        FollowUp.objects.filter(pk=follow_up.pk).update(reminder_sent_at=now)
    return sent
