import uuid

from django.conf import settings
from django.contrib.postgres.fields import ArrayField
from django.core.validators import MaxValueValidator, MinValueValidator
from django.db import models
from django.db.models import Q
from pgvector.django import VectorField

from apps.core.models import TimeStampedModel
from apps.core.storage import get_private_storage


def short_list(max_len=120, size=None):
    """Postgres text array for short preference lists (roles, locations…).

    Arrays keep the small, user-edited preference lists on one row while staying
    indexable/queryable; anything with its own attributes gets a real table.
    """
    return ArrayField(models.CharField(max_length=max_len), default=list, blank=True, size=size)


class WorkModel(models.TextChoices):
    REMOTE = "remote", "Remote"
    HYBRID = "hybrid", "Hybrid"
    ONSITE = "onsite", "On-site"


class EmploymentType(models.TextChoices):
    FULL_TIME = "full_time", "Full-time"
    PART_TIME = "part_time", "Part-time"
    CONTRACT = "contract", "Contract"
    INTERNSHIP = "internship", "Internship"
    TEMPORARY = "temporary", "Temporary"


class Skill(models.Model):
    """Global canonical skill (shared across users and job postings)."""

    name = models.CharField(max_length=100, unique=True)
    category = models.CharField(max_length=40, default="other", db_index=True)

    class Meta:
        ordering = ["name"]

    def __str__(self):
        return self.name


class CandidateProfile(TimeStampedModel):
    user = models.OneToOneField(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="profile")
    full_name = models.CharField(max_length=200, blank=True)
    headline = models.CharField(max_length=200, blank=True)
    summary = models.TextField(blank=True)
    current_title = models.CharField(max_length=200, blank=True)
    location = models.CharField(max_length=200, blank=True)
    email = models.EmailField(blank=True)
    phone = models.CharField(max_length=40, blank=True)
    linkedin_url = models.URLField(blank=True)
    github_url = models.URLField(blank=True)
    portfolio_url = models.URLField(blank=True)
    # Configurable, not derived silently: the parser proposes a value, the user confirms it.
    total_experience_years = models.DecimalField(
        max_digits=4, decimal_places=1, default=0, validators=[MinValueValidator(0), MaxValueValidator(60)]
    )
    verified_at = models.DateTimeField(null=True, blank=True, help_text="When the user last confirmed the profile")
    # Data-retention: generated docs / closed applications older than this are purged (0 = keep).
    data_retention_days = models.PositiveIntegerField(default=0)
    embedding = VectorField(dimensions=settings.EMBEDDING_DIMENSIONS, null=True, blank=True)
    embedding_hash = models.CharField(max_length=64, blank=True)

    def __str__(self):
        return self.full_name or str(self.user)


class CareerPreference(TimeStampedModel):
    profile = models.OneToOneField(CandidateProfile, on_delete=models.CASCADE, related_name="preferences")
    target_roles = short_list(200)
    preferred_locations = short_list(120)
    work_models = short_list(20)
    employment_types = short_list(20)
    industries = short_list(120)
    company_types = short_list(120)
    excluded_companies = short_list(200)
    excluded_keywords = short_list(120)
    salary_currency = models.CharField(max_length=3, default="INR")
    # Annual amounts in salary_currency major units (e.g. 3000000 = ₹30 LPA).
    salary_min = models.PositiveBigIntegerField(null=True, blank=True)
    salary_max = models.PositiveBigIntegerField(null=True, blank=True)
    min_match_score = models.PositiveSmallIntegerField(default=60, validators=[MaxValueValidator(100)])
    open_to_relocation = models.BooleanField(default=False)
    notice_period_days = models.PositiveSmallIntegerField(null=True, blank=True)

    class Meta:
        constraints = [
            models.CheckConstraint(
                condition=Q(salary_min__isnull=True) | Q(salary_max__isnull=True) | Q(salary_min__lte=models.F("salary_max")),
                name="pref_salary_range_valid",
            )
        ]


class Proficiency(models.TextChoices):
    BEGINNER = "beginner", "Beginner"
    INTERMEDIATE = "intermediate", "Intermediate"
    ADVANCED = "advanced", "Advanced"
    EXPERT = "expert", "Expert"


class CandidateSkill(TimeStampedModel):
    profile = models.ForeignKey(CandidateProfile, on_delete=models.CASCADE, related_name="skills")
    skill = models.ForeignKey(Skill, on_delete=models.PROTECT, related_name="candidates")
    years = models.DecimalField(max_digits=4, decimal_places=1, null=True, blank=True)
    proficiency = models.CharField(max_length=20, choices=Proficiency.choices, blank=True)
    is_primary = models.BooleanField(default=False)
    # Only verified skills are used as evidence in matching and tailoring.
    verified = models.BooleanField(default=False)
    source = models.CharField(max_length=20, default="manual")

    class Meta:
        ordering = ["-is_primary", "skill__name"]
        constraints = [models.UniqueConstraint(fields=["profile", "skill"], name="uniq_profile_skill")]


class EmploymentHistory(TimeStampedModel):
    profile = models.ForeignKey(CandidateProfile, on_delete=models.CASCADE, related_name="employment")
    company = models.CharField(max_length=200)
    title = models.CharField(max_length=200)
    location = models.CharField(max_length=200, blank=True)
    start_date = models.DateField(null=True, blank=True)
    end_date = models.DateField(null=True, blank=True)
    is_current = models.BooleanField(default=False)
    description = models.TextField(blank=True)
    achievements = ArrayField(models.TextField(), default=list, blank=True)
    technologies = short_list(100)
    sort_order = models.PositiveSmallIntegerField(default=0)
    verified = models.BooleanField(default=False)

    class Meta:
        ordering = ["sort_order", "-start_date"]
        constraints = [
            models.CheckConstraint(
                condition=Q(end_date__isnull=True) | Q(start_date__isnull=True) | Q(end_date__gte=models.F("start_date")),
                name="employment_dates_valid",
            )
        ]


class Education(TimeStampedModel):
    profile = models.ForeignKey(CandidateProfile, on_delete=models.CASCADE, related_name="education")
    institution = models.CharField(max_length=200)
    degree = models.CharField(max_length=200, blank=True)
    field_of_study = models.CharField(max_length=200, blank=True)
    start_year = models.PositiveSmallIntegerField(null=True, blank=True)
    end_year = models.PositiveSmallIntegerField(null=True, blank=True)
    grade = models.CharField(max_length=50, blank=True)
    verified = models.BooleanField(default=False)

    class Meta:
        ordering = ["-end_year"]


class Project(TimeStampedModel):
    profile = models.ForeignKey(CandidateProfile, on_delete=models.CASCADE, related_name="projects")
    name = models.CharField(max_length=200)
    role = models.CharField(max_length=200, blank=True)
    description = models.TextField(blank=True)
    highlights = ArrayField(models.TextField(), default=list, blank=True)
    technologies = short_list(100)
    url = models.URLField(blank=True)
    verified = models.BooleanField(default=False)


class Certification(TimeStampedModel):
    profile = models.ForeignKey(CandidateProfile, on_delete=models.CASCADE, related_name="certifications")
    name = models.CharField(max_length=200)
    issuer = models.CharField(max_length=200, blank=True)
    issued_on = models.DateField(null=True, blank=True)
    expires_on = models.DateField(null=True, blank=True)
    credential_url = models.URLField(blank=True)
    verified = models.BooleanField(default=False)


def resume_upload_path(instance, filename):
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else "bin"
    return f"users/{instance.user_id}/resumes/{uuid.uuid4().hex}.{ext}"


class Resume(TimeStampedModel):
    """An uploaded resume file. The file itself is immutable once stored."""

    class ParseStatus(models.TextChoices):
        PENDING = "pending", "Pending"
        PARSING = "parsing", "Parsing"
        PARSED = "parsed", "Parsed"
        APPLIED = "applied", "Applied to profile"
        FAILED = "failed", "Failed"

    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="resumes")
    title = models.CharField(max_length=200, blank=True)
    file = models.FileField(upload_to=resume_upload_path, storage=get_private_storage, max_length=300)
    original_filename = models.CharField(max_length=255)
    content_type = models.CharField(max_length=100)
    size_bytes = models.PositiveIntegerField()
    sha256 = models.CharField(max_length=64)
    is_master = models.BooleanField(default=False)
    parse_status = models.CharField(max_length=20, choices=ParseStatus.choices, default=ParseStatus.PENDING, db_index=True)
    parse_error = models.CharField(max_length=500, blank=True)
    parser = models.CharField(max_length=40, blank=True)
    # Draft extracted from the file, awaiting user verification. Not used for matching until applied.
    parsed_data = models.JSONField(default=dict, blank=True)
    # Extracted text: sensitive. Never logged; excluded from list serializers.
    raw_text = models.TextField(blank=True)

    class Meta:
        ordering = ["-created_at"]
        constraints = [
            models.UniqueConstraint(fields=["user"], condition=Q(is_master=True), name="one_master_resume_per_user"),
            models.UniqueConstraint(fields=["user", "sha256"], name="uniq_resume_file_per_user"),
        ]

    def __str__(self):
        return self.title or self.original_filename


class ResumeVersion(TimeStampedModel):
    """Structured resume content. Master versions snapshot the verified profile;
    tailored versions are generated per job and need explicit approval before use."""

    class Kind(models.TextChoices):
        MASTER = "master", "Master snapshot"
        TAILORED = "tailored", "Tailored"

    class Status(models.TextChoices):
        DRAFT = "draft", "Draft"
        AWAITING_APPROVAL = "awaiting_approval", "Awaiting approval"
        APPROVED = "approved", "Approved"
        REJECTED = "rejected", "Rejected"
        ARCHIVED = "archived", "Archived"

    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="resume_versions")
    source_resume = models.ForeignKey(Resume, null=True, blank=True, on_delete=models.SET_NULL, related_name="versions")
    base_version = models.ForeignKey("self", null=True, blank=True, on_delete=models.SET_NULL, related_name="derived")
    job = models.ForeignKey("jobs.JobPosting", null=True, blank=True, on_delete=models.SET_NULL, related_name="resume_versions")
    version_number = models.PositiveIntegerField()
    label = models.CharField(max_length=200)
    kind = models.CharField(max_length=20, choices=Kind.choices)
    status = models.CharField(max_length=20, choices=Status.choices, default=Status.DRAFT, db_index=True)
    content = models.JSONField(default=dict)
    # For tailored versions: how each job requirement maps to verified evidence, plus any
    # generator warnings (e.g. claims removed because they were not in the profile).
    tailoring_notes = models.JSONField(default=dict, blank=True)
    generator = models.CharField(max_length=40, blank=True)
    approved_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["-version_number"]
        constraints = [models.UniqueConstraint(fields=["user", "version_number"], name="uniq_resume_version_number")]
        indexes = [models.Index(fields=["user", "kind", "status"])]

    def __str__(self):
        return f"v{self.version_number} {self.label}"
