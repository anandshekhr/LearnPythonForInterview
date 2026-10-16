from django.conf import settings
from django.db import models
from django.db.models import Q

from apps.core.models import TimeStampedModel


class ApplicationStatus(models.TextChoices):
    DISCOVERED = "discovered", "Discovered"
    SHORTLISTED = "shortlisted", "Shortlisted"
    RESUME_PREPARED = "resume_prepared", "Resume prepared"
    AWAITING_APPROVAL = "awaiting_approval", "Awaiting approval"
    READY_TO_APPLY = "ready_to_apply", "Ready to apply"
    SUBMITTING = "submitting", "Submitting"  # internal lock while a submission is in flight
    SUBMITTED = "submitted", "Application submitted"
    SUBMISSION_FAILED = "submission_failed", "Submission failed"
    ASSESSMENT = "assessment", "Assessment"
    INTERVIEW_SCHEDULED = "interview_scheduled", "Interview scheduled"
    INTERVIEW_COMPLETED = "interview_completed", "Interview completed"
    REJECTED = "rejected", "Rejected"
    OFFER_RECEIVED = "offer_received", "Offer received"
    WITHDRAWN = "withdrawn", "Withdrawn"
    CLOSED = "closed", "Closed"


S = ApplicationStatus
POST_SUBMISSION = {S.ASSESSMENT, S.INTERVIEW_SCHEDULED, S.INTERVIEW_COMPLETED, S.REJECTED, S.OFFER_RECEIVED, S.WITHDRAWN, S.CLOSED}
# Allowed transitions. Submission states are only reachable through the submission service.
TRANSITIONS: dict[str, set[str]] = {
    S.DISCOVERED: {S.SHORTLISTED, S.RESUME_PREPARED, S.WITHDRAWN, S.CLOSED},
    S.SHORTLISTED: {S.DISCOVERED, S.RESUME_PREPARED, S.AWAITING_APPROVAL, S.WITHDRAWN, S.CLOSED},
    S.RESUME_PREPARED: {S.AWAITING_APPROVAL, S.SHORTLISTED, S.WITHDRAWN, S.CLOSED},
    S.AWAITING_APPROVAL: {S.READY_TO_APPLY, S.RESUME_PREPARED, S.WITHDRAWN, S.CLOSED},
    S.READY_TO_APPLY: {S.SUBMITTING, S.SUBMITTED, S.AWAITING_APPROVAL, S.WITHDRAWN, S.CLOSED},
    S.SUBMITTING: {S.SUBMITTED, S.SUBMISSION_FAILED, S.READY_TO_APPLY},
    S.SUBMISSION_FAILED: {S.READY_TO_APPLY, S.SUBMITTED, S.WITHDRAWN, S.CLOSED},
    S.SUBMITTED: POST_SUBMISSION,
    S.ASSESSMENT: POST_SUBMISSION - {S.ASSESSMENT},
    S.INTERVIEW_SCHEDULED: POST_SUBMISSION - {S.INTERVIEW_SCHEDULED},
    S.INTERVIEW_COMPLETED: POST_SUBMISSION,
    S.REJECTED: {S.CLOSED},
    S.OFFER_RECEIVED: {S.CLOSED, S.WITHDRAWN},
    S.WITHDRAWN: {S.CLOSED},
    S.CLOSED: set(),
}
# Statuses the user may set by hand (submission outcomes come from the submission flow,
# except "submitted", which the user confirms after applying on the official page).
MANUAL_ALLOWED = set(ApplicationStatus.values) - {S.SUBMITTING, S.SUBMISSION_FAILED, S.READY_TO_APPLY, S.AWAITING_APPROVAL,
                                                   S.RESUME_PREPARED}


class SubmissionChannel(models.TextChoices):
    NONE = "", "Not submitted"
    API = "api", "Authorized API"
    BROWSER_ASSISTED = "browser_assisted", "Browser-assisted (user submitted)"
    MANUAL = "manual", "Manual (user confirmed)"


class Application(TimeStampedModel):
    TERMINAL = {S.REJECTED, S.WITHDRAWN, S.CLOSED}
    SUBMITTED_STATES = {S.SUBMITTED} | POST_SUBMISSION

    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="applications")
    job = models.ForeignKey("jobs.JobPosting", null=True, blank=True, on_delete=models.SET_NULL, related_name="applications")
    match = models.ForeignKey("matching.JobMatch", null=True, blank=True, on_delete=models.SET_NULL, related_name="applications")
    # Snapshots so history survives source/job deletion.
    company_name = models.CharField(max_length=200)
    job_title = models.CharField(max_length=300)
    job_url = models.URLField(max_length=600)
    source_type = models.CharField(max_length=20, blank=True)
    job_fingerprint = models.CharField(max_length=64, blank=True, db_index=True)
    match_score = models.PositiveSmallIntegerField(null=True, blank=True)
    status = models.CharField(max_length=24, choices=ApplicationStatus.choices, default=S.DISCOVERED, db_index=True)
    mode = models.CharField(max_length=20, default="discovery")
    resume_version = models.ForeignKey("profiles.ResumeVersion", null=True, blank=True, on_delete=models.PROTECT,
                                       related_name="applications")
    resume_document = models.ForeignKey("documents.GeneratedDocument", null=True, blank=True, on_delete=models.PROTECT,
                                        related_name="resume_applications")
    cover_letter = models.ForeignKey("documents.GeneratedDocument", null=True, blank=True, on_delete=models.SET_NULL,
                                     related_name="cover_letter_applications")
    approved_at = models.DateTimeField(null=True, blank=True)
    submitted_at = models.DateTimeField(null=True, blank=True, db_index=True)
    submission_channel = models.CharField(max_length=20, choices=SubmissionChannel.choices, blank=True)
    submission_reference = models.CharField(max_length=200, blank=True)
    submission_response = models.JSONField(default=dict, blank=True)
    submission_attempts = models.PositiveSmallIntegerField(default=0)
    submission_started_at = models.DateTimeField(null=True, blank=True)
    idempotency_key = models.CharField(max_length=80, blank=True)
    last_error = models.CharField(max_length=1000, blank=True)
    notes = models.TextField(blank=True, help_text="Recruiter notes and personal notes")
    follow_up_date = models.DateField(null=True, blank=True)
    last_verified_at = models.DateTimeField(null=True, blank=True, help_text="When the status was last confirmed")

    class Meta:
        ordering = ["-updated_at"]
        constraints = [
            # One application per user per posting.
            models.UniqueConstraint(fields=["user", "job"], condition=Q(job__isnull=False), name="uniq_application_user_job"),
            models.UniqueConstraint(fields=["user", "idempotency_key"], condition=~Q(idempotency_key=""),
                                    name="uniq_application_idempotency"),
            models.CheckConstraint(condition=Q(status__in=ApplicationStatus.values), name="application_status_valid"),
        ]
        indexes = [models.Index(fields=["user", "status"]), models.Index(fields=["user", "-submitted_at"])]

    def __str__(self):
        return f"{self.job_title} @ {self.company_name} [{self.status}]"

    @property
    def is_submitted(self):
        return self.status in self.SUBMITTED_STATES


class ApplicationStatusHistory(models.Model):
    application = models.ForeignKey(Application, on_delete=models.CASCADE, related_name="history")
    from_status = models.CharField(max_length=24, blank=True)
    to_status = models.CharField(max_length=24)
    changed_by = models.ForeignKey(settings.AUTH_USER_MODEL, null=True, blank=True, on_delete=models.SET_NULL)
    actor = models.CharField(max_length=20, default="user", help_text="user / system")
    note = models.CharField(max_length=1000, blank=True)
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)

    class Meta:
        ordering = ["created_at", "id"]


class ApplicationAnswer(TimeStampedModel):
    class Origin(models.TextChoices):
        PROFILE = "profile", "From verified profile"
        GENERATED = "generated", "Drafted (needs review)"
        USER = "user", "Entered by you"

    application = models.ForeignKey(Application, on_delete=models.CASCADE, related_name="answers")
    question_key = models.CharField(max_length=200)
    question_label = models.CharField(max_length=1000)
    field_type = models.CharField(max_length=20, default="text")
    required = models.BooleanField(default=False)
    options = models.JSONField(default=list, blank=True)
    answer = models.TextField(blank=True)
    origin = models.CharField(max_length=10, choices=Origin.choices, default=Origin.USER)
    # Legal, demographic, disability, sponsorship and similar questions: never auto-filled
    # and never submitted unless the user explicitly reviewed them.
    sensitive = models.BooleanField(default=False)
    reviewed_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["id"]
        constraints = [models.UniqueConstraint(fields=["application", "question_key"], name="uniq_answer_per_question")]


class Interview(TimeStampedModel):
    class Mode(models.TextChoices):
        VIDEO = "video", "Video"
        PHONE = "phone", "Phone"
        ONSITE = "onsite", "On-site"

    class Outcome(models.TextChoices):
        PENDING = "pending", "Pending"
        PASSED = "passed", "Passed"
        FAILED = "failed", "Not progressed"
        CANCELLED = "cancelled", "Cancelled"

    application = models.ForeignKey(Application, on_delete=models.CASCADE, related_name="interviews")
    round_name = models.CharField(max_length=120)
    scheduled_at = models.DateTimeField(db_index=True)
    duration_minutes = models.PositiveSmallIntegerField(default=60)
    mode = models.CharField(max_length=10, choices=Mode.choices, default=Mode.VIDEO)
    location = models.CharField(max_length=300, blank=True, help_text="Address or meeting link")
    interviewers = models.CharField(max_length=300, blank=True)
    notes = models.TextField(blank=True)
    outcome = models.CharField(max_length=10, choices=Outcome.choices, default=Outcome.PENDING)
    reminder_sent_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["scheduled_at"]


class FollowUp(TimeStampedModel):
    application = models.ForeignKey(Application, on_delete=models.CASCADE, related_name="follow_ups")
    due_date = models.DateField(db_index=True)
    note = models.CharField(max_length=500, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    reminder_sent_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["due_date"]
