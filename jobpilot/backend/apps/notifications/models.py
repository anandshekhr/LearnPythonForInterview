from django.conf import settings
from django.db import models


class Notification(models.Model):
    class Kind(models.TextChoices):
        HIGH_MATCH = "high_match", "New high-match job"
        AWAITING_REVIEW = "awaiting_review", "Awaiting your review"
        SUBMITTED = "submitted", "Application submitted"
        SUBMISSION_FAILED = "submission_failed", "Submission failed"
        ACTION_REQUIRED = "action_required", "Action required"
        INTERVIEW = "interview", "Upcoming interview"
        FOLLOW_UP = "follow_up", "Follow-up reminder"
        JOB_EXPIRED = "job_expired", "Job posting expired"
        SOURCE_ERROR = "source_error", "Job source failing"
        SYSTEM = "system", "System"

    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="notifications")
    kind = models.CharField(max_length=30, choices=Kind.choices, db_index=True)
    title = models.CharField(max_length=200)
    body = models.TextField(blank=True)
    link = models.CharField(max_length=300, blank=True, help_text="Frontend route, e.g. /jobs/42")
    object_type = models.CharField(max_length=60, blank=True)
    object_id = models.CharField(max_length=64, blank=True)
    dedupe_key = models.CharField(max_length=200)
    read_at = models.DateTimeField(null=True, blank=True)
    emailed_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)

    class Meta:
        ordering = ["-created_at"]
        constraints = [models.UniqueConstraint(fields=["user", "dedupe_key"], name="uniq_notification_dedupe")]
        indexes = [models.Index(fields=["user", "read_at", "-created_at"])]
