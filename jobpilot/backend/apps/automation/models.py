from datetime import timedelta

from django.conf import settings
from django.core.validators import MaxValueValidator, MinValueValidator
from django.db import models
from django.utils import timezone

from apps.core.models import TimeStampedModel


class ApplicationMode(models.TextChoices):
    DISCOVERY = "discovery", "Discovery only"
    ASSISTED = "assisted", "Assisted application"
    AUTOMATED = "automated", "Authorized automated submission"


class AutomationSettings(TimeStampedModel):
    user = models.OneToOneField(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="automation_settings")
    application_mode = models.CharField(max_length=20, choices=ApplicationMode.choices, default=ApplicationMode.DISCOVERY)
    # Explicit, timestamped consent is required before any automated submission.
    automated_submission_consent_at = models.DateTimeField(null=True, blank=True)
    max_submissions_per_day = models.PositiveSmallIntegerField(default=5, validators=[MaxValueValidator(25)])
    auto_shortlist_min_score = models.PositiveSmallIntegerField(null=True, blank=True, validators=[MaxValueValidator(100)])
    auto_prepare_documents_min_score = models.PositiveSmallIntegerField(null=True, blank=True, validators=[MaxValueValidator(100)])
    notify_min_score = models.PositiveSmallIntegerField(default=75, validators=[MaxValueValidator(100)])
    notify_new_matches = models.BooleanField(default=True)
    notify_awaiting_review = models.BooleanField(default=True)
    notify_submissions = models.BooleanField(default=True)
    notify_interviews = models.BooleanField(default=True)
    notify_follow_ups = models.BooleanField(default=True)
    notify_expired_jobs = models.BooleanField(default=True)
    email_notifications = models.BooleanField(default=False, help_text="Reserved for a future email integration")

    @property
    def automated_submission_allowed(self):
        return self.application_mode == ApplicationMode.AUTOMATED and self.automated_submission_consent_at is not None


class AutomationRun(models.Model):
    """Execution history for background work: syncs, parsing, matching, submissions."""

    class Status(models.TextChoices):
        RUNNING = "running", "Running"
        SUCCEEDED = "succeeded", "Succeeded"
        FAILED = "failed", "Failed"
        INTERRUPTED = "interrupted", "Interrupted"
        SKIPPED = "skipped", "Skipped"

    user = models.ForeignKey(settings.AUTH_USER_MODEL, null=True, blank=True, on_delete=models.CASCADE, related_name="automation_runs")
    kind = models.CharField(max_length=40, db_index=True)
    status = models.CharField(max_length=20, choices=Status.choices, default=Status.RUNNING, db_index=True)
    task_id = models.CharField(max_length=64, blank=True)
    target_type = models.CharField(max_length=60, blank=True)
    target_id = models.CharField(max_length=64, blank=True)
    started_at = models.DateTimeField(default=timezone.now, db_index=True)
    heartbeat_at = models.DateTimeField(default=timezone.now)
    finished_at = models.DateTimeField(null=True, blank=True)
    stats = models.JSONField(default=dict, blank=True)
    error = models.CharField(max_length=1000, blank=True)

    class Meta:
        ordering = ["-started_at"]
        indexes = [models.Index(fields=["user", "-started_at"]), models.Index(fields=["status", "heartbeat_at"])]

    @property
    def duration_seconds(self):
        if not self.finished_at:
            return None
        return round((self.finished_at - self.started_at).total_seconds(), 2)


class ScheduledSearch(TimeStampedModel):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="scheduled_searches")
    name = models.CharField(max_length=120)
    enabled = models.BooleanField(default=True)
    interval_hours = models.PositiveSmallIntegerField(default=12, validators=[MinValueValidator(1), MaxValueValidator(168)])
    # Empty = every enabled source the user follows.
    sources = models.ManyToManyField("jobs.JobSource", blank=True, related_name="scheduled_searches")
    last_run_at = models.DateTimeField(null=True, blank=True)
    next_run_at = models.DateTimeField(default=timezone.now, db_index=True)

    class Meta:
        ordering = ["name"]
        constraints = [models.UniqueConstraint(fields=["user", "name"], name="uniq_search_name_per_user")]

    def schedule_next(self, now=None):
        now = now or timezone.now()
        self.last_run_at = now
        self.next_run_at = now + timedelta(hours=self.interval_hours)
