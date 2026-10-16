from django.db import IntegrityError, transaction

from .models import Notification

PREFERENCE_FIELD = {
    Notification.Kind.HIGH_MATCH: "notify_new_matches",
    Notification.Kind.AWAITING_REVIEW: "notify_awaiting_review",
    Notification.Kind.SUBMITTED: "notify_submissions",
    Notification.Kind.SUBMISSION_FAILED: "notify_submissions",
    Notification.Kind.INTERVIEW: "notify_interviews",
    Notification.Kind.FOLLOW_UP: "notify_follow_ups",
    Notification.Kind.JOB_EXPIRED: "notify_expired_jobs",
}


def notify(user, kind, title, body="", link="", obj=None, dedupe_key=None, respect_preferences=True):
    """Create an in-app notification once per dedupe_key. Returns the notification or None."""
    if respect_preferences and kind in PREFERENCE_FIELD:
        from apps.automation.services import get_settings

        if not getattr(get_settings(user), PREFERENCE_FIELD[kind]):
            return None
    key = dedupe_key or f"{kind}:{obj.__class__.__name__ if obj else ''}:{getattr(obj, 'pk', '')}"
    try:
        with transaction.atomic():
            return Notification.objects.create(
                user=user, kind=kind, title=title[:200], body=body, link=link[:300],
                object_type=obj.__class__.__name__ if obj is not None else "",
                object_id=str(obj.pk) if obj is not None else "", dedupe_key=key[:200],
            )
    except IntegrityError:
        return None
