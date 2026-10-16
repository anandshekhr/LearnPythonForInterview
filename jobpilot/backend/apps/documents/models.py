import uuid

from django.conf import settings
from django.db import models

from apps.core.models import TimeStampedModel
from apps.core.storage import get_private_storage


def document_upload_path(instance, filename):
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else "bin"
    return f"users/{instance.user_id}/documents/{uuid.uuid4().hex}.{ext}"


class GeneratedDocument(TimeStampedModel):
    """A rendered resume or cover letter associated with a job."""

    class Kind(models.TextChoices):
        RESUME = "resume", "Resume"
        COVER_LETTER = "cover_letter", "Cover letter"

    class Format(models.TextChoices):
        PDF = "pdf", "PDF"
        DOCX = "docx", "DOCX"
        TEXT = "text", "Plain text"

    class Status(models.TextChoices):
        DRAFT = "draft", "Draft"
        AWAITING_APPROVAL = "awaiting_approval", "Awaiting approval"
        APPROVED = "approved", "Approved"
        REJECTED = "rejected", "Rejected"

    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="documents")
    job = models.ForeignKey("jobs.JobPosting", null=True, blank=True, on_delete=models.SET_NULL, related_name="documents")
    resume_version = models.ForeignKey("profiles.ResumeVersion", null=True, blank=True, on_delete=models.CASCADE,
                                       related_name="documents")
    kind = models.CharField(max_length=20, choices=Kind.choices)
    format = models.CharField(max_length=10, choices=Format.choices)
    template = models.CharField(max_length=40, default="classic")
    status = models.CharField(max_length=20, choices=Status.choices, default=Status.AWAITING_APPROVAL, db_index=True)
    title = models.CharField(max_length=200)
    text_content = models.TextField(blank=True, help_text="Cover letter body (editable before approval)")
    file = models.FileField(upload_to=document_upload_path, storage=get_private_storage, max_length=300, blank=True)
    size_bytes = models.PositiveIntegerField(default=0)
    sha256 = models.CharField(max_length=64, blank=True)
    generator = models.CharField(max_length=40, blank=True)
    warnings = models.JSONField(default=list, blank=True)
    approved_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["-created_at"]
        indexes = [models.Index(fields=["user", "kind", "status"])]

    def __str__(self):
        return self.title

    @property
    def content_type(self):
        return {
            "pdf": "application/pdf",
            "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "text": "text/plain",
        }[self.format]
