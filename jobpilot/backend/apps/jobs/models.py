from django.conf import settings
from django.contrib.postgres.fields import ArrayField
from django.db import models
from django.utils.text import slugify
from pgvector.django import HnswIndex, VectorField

from apps.core import crypto
from apps.core.models import TimeStampedModel


class Company(TimeStampedModel):
    """Global company registry. Any user can add a company; only its creator or staff can edit it."""

    name = models.CharField(max_length=200)
    slug = models.SlugField(max_length=220, unique=True)
    website = models.URLField(blank=True)
    careers_url = models.URLField(blank=True)
    industry = models.CharField(max_length=120, blank=True)
    company_type = models.CharField(max_length=60, blank=True, help_text="e.g. Product, SaaS, AI startup, IT services")
    headquarters = models.CharField(max_length=200, blank=True)
    notes = models.TextField(blank=True)
    created_by = models.ForeignKey(settings.AUTH_USER_MODEL, null=True, blank=True, on_delete=models.SET_NULL,
                                   related_name="companies_created")

    class Meta:
        ordering = ["name"]
        verbose_name_plural = "companies"

    def save(self, *args, **kwargs):
        if not self.slug:
            base = slugify(self.name)[:200] or "company"
            slug, n = base, 2
            while Company.objects.filter(slug=slug).exclude(pk=self.pk).exists():
                slug, n = f"{base}-{n}", n + 1
            self.slug = slug
        super().save(*args, **kwargs)

    def __str__(self):
        return self.name


class JobSource(TimeStampedModel):
    class SourceType(models.TextChoices):
        GREENHOUSE = "greenhouse", "Greenhouse Job Board API"
        LEVER = "lever", "Lever Postings API"
        WORKDAY = "workday", "Workday career site"
        CAREER_PAGE = "career_page", "Company career page (schema.org JobPosting)"

    class Status(models.TextChoices):
        UNVERIFIED = "unverified", "Not yet synchronised"
        ACTIVE = "active", "Active"
        DEGRADED = "degraded", "Failing (retrying)"
        ERROR = "error", "Error"
        DISABLED = "disabled", "Disabled"

    company = models.ForeignKey(Company, on_delete=models.CASCADE, related_name="sources")
    source_type = models.CharField(max_length=20, choices=SourceType.choices)
    identifier = models.CharField(max_length=500, help_text="Board token / site name, Workday site URL, or careers page URL")
    config = models.JSONField(default=dict, blank=True, help_text="Adapter-specific options, e.g. {'region': 'eu'}")
    enabled = models.BooleanField(default=True)
    status = models.CharField(max_length=20, choices=Status.choices, default=Status.UNVERIFIED, db_index=True)
    sync_interval_minutes = models.PositiveIntegerField(default=360)
    min_request_interval_seconds = models.FloatField(default=1.0, help_text="Per-source throttle between HTTP calls")
    last_synced_at = models.DateTimeField(null=True, blank=True)
    last_success_at = models.DateTimeField(null=True, blank=True)
    last_error = models.CharField(max_length=1000, blank=True)
    last_error_at = models.DateTimeField(null=True, blank=True)
    consecutive_failures = models.PositiveSmallIntegerField(default=0)
    active_job_count = models.PositiveIntegerField(default=0)
    # Authorized submission: only with an API key issued by the employer for this board.
    submission_enabled = models.BooleanField(default=False)
    submission_api_key_encrypted = models.TextField(blank=True)
    submission_terms_confirmed_at = models.DateTimeField(null=True, blank=True)
    created_by = models.ForeignKey(settings.AUTH_USER_MODEL, null=True, blank=True, on_delete=models.SET_NULL,
                                   related_name="sources_created")

    class Meta:
        ordering = ["company__name"]
        constraints = [models.UniqueConstraint(fields=["source_type", "identifier"], name="uniq_source_identifier")]

    def __str__(self):
        return f"{self.company} ({self.source_type}:{self.identifier})"

    def set_submission_api_key(self, key: str):
        self.submission_api_key_encrypted = crypto.encrypt(key) if key else ""

    def get_submission_api_key(self) -> str:
        return crypto.decrypt(self.submission_api_key_encrypted)

    @property
    def has_submission_key(self):
        return bool(self.submission_api_key_encrypted)


class WorkModelChoice(models.TextChoices):
    REMOTE = "remote", "Remote"
    HYBRID = "hybrid", "Hybrid"
    ONSITE = "onsite", "On-site"
    UNKNOWN = "unknown", "Unknown"


class Seniority(models.TextChoices):
    INTERN = "intern", "Intern"
    JUNIOR = "junior", "Junior"
    MID = "mid", "Mid-level"
    SENIOR = "senior", "Senior"
    LEAD = "lead", "Lead / Staff"
    PRINCIPAL = "principal", "Principal"
    MANAGER = "manager", "Manager"
    UNKNOWN = "unknown", "Unknown"


class JobPosting(TimeStampedModel):
    class Status(models.TextChoices):
        ACTIVE = "active", "Active"
        EXPIRED = "expired", "Expired"

    source = models.ForeignKey(JobSource, on_delete=models.CASCADE, related_name="jobs")
    company = models.ForeignKey(Company, on_delete=models.CASCADE, related_name="jobs")
    external_id = models.CharField(max_length=120)
    canonical_url = models.URLField(max_length=600)
    apply_url = models.URLField(max_length=600, blank=True)
    title = models.CharField(max_length=300)
    normalized_title = models.CharField(max_length=300, db_index=True)
    department = models.CharField(max_length=200, blank=True)
    location_raw = models.CharField(max_length=500, blank=True)
    locations = ArrayField(models.CharField(max_length=120), default=list, blank=True)
    work_model = models.CharField(max_length=10, choices=WorkModelChoice.choices, default=WorkModelChoice.UNKNOWN, db_index=True)
    employment_type = models.CharField(max_length=20, blank=True, db_index=True)
    seniority = models.CharField(max_length=20, choices=Seniority.choices, default=Seniority.UNKNOWN)
    description_text = models.TextField(blank=True)
    salary_min = models.PositiveBigIntegerField(null=True, blank=True)
    salary_max = models.PositiveBigIntegerField(null=True, blank=True)
    salary_currency = models.CharField(max_length=3, blank=True)
    salary_period = models.CharField(max_length=10, blank=True, help_text="year/month/hour")
    salary_text = models.CharField(max_length=200, blank=True)
    published_at = models.DateTimeField(null=True, blank=True, db_index=True)
    source_updated_at = models.DateTimeField(null=True, blank=True)
    first_seen_at = models.DateTimeField(auto_now_add=True, db_index=True)
    last_seen_at = models.DateTimeField(db_index=True)
    status = models.CharField(max_length=10, choices=Status.choices, default=Status.ACTIVE, db_index=True)
    expired_at = models.DateTimeField(null=True, blank=True)
    content_hash = models.CharField(max_length=64)
    description_version = models.PositiveIntegerField(default=1)
    fingerprint = models.CharField(max_length=64, db_index=True, help_text="company+title+location hash for cross-source dedup")
    duplicate_of = models.ForeignKey("self", null=True, blank=True, on_delete=models.SET_NULL, related_name="duplicates")
    # Extracted requirements (derived data, regenerated when content_hash changes).
    requirements = models.JSONField(default=dict, blank=True)
    requirements_hash = models.CharField(max_length=64, blank=True)
    min_years_experience = models.DecimalField(max_digits=4, decimal_places=1, null=True, blank=True)
    embedding = VectorField(dimensions=settings.EMBEDDING_DIMENSIONS, null=True, blank=True)
    embedding_hash = models.CharField(max_length=64, blank=True)
    raw_metadata = models.JSONField(default=dict, blank=True)
    # When full details were last downloaded (adapters skip re-downloading recent ones).
    details_fetched_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["-published_at", "-first_seen_at"]
        constraints = [models.UniqueConstraint(fields=["source", "external_id"], name="uniq_job_per_source")]
        indexes = [
            models.Index(fields=["status", "-published_at"]),
            models.Index(fields=["canonical_url"]),
            models.Index(fields=["company", "status"]),
            HnswIndex(name="job_embedding_hnsw", fields=["embedding"], m=16, ef_construction=64, opclasses=["vector_cosine_ops"]),
        ]

    def __str__(self):
        return f"{self.title} @ {self.company}"

    @property
    def official_url(self):
        return self.apply_url or self.canonical_url


class JobSkill(models.Model):
    class Importance(models.TextChoices):
        REQUIRED = "required", "Required"
        PREFERRED = "preferred", "Preferred"

    job = models.ForeignKey(JobPosting, on_delete=models.CASCADE, related_name="skills")
    skill = models.ForeignKey("profiles.Skill", on_delete=models.PROTECT, related_name="jobs")
    importance = models.CharField(max_length=10, choices=Importance.choices)

    class Meta:
        constraints = [models.UniqueConstraint(fields=["job", "skill"], name="uniq_job_skill")]
        indexes = [models.Index(fields=["skill", "importance"])]


class JobPostingChange(models.Model):
    """History of material changes detected on a posting during sync."""

    job = models.ForeignKey(JobPosting, on_delete=models.CASCADE, related_name="changes")
    detected_at = models.DateTimeField(auto_now_add=True)
    change_type = models.CharField(max_length=20)  # updated / expired / reactivated
    fields_changed = ArrayField(models.CharField(max_length=60), default=list, blank=True)
    previous_content_hash = models.CharField(max_length=64, blank=True)

    class Meta:
        ordering = ["-detected_at"]
