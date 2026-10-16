from django.conf import settings
from django.contrib.postgres.fields import ArrayField
from django.core.validators import MaxValueValidator
from django.db import models

from apps.core.models import TimeStampedModel


class ScoringConfig(TimeStampedModel):
    """Per-user scoring weights and category thresholds. Weights are relative (normalised at runtime)."""

    user = models.OneToOneField(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="scoring_config")
    w_required_skills = models.PositiveSmallIntegerField(default=30, validators=[MaxValueValidator(100)])
    w_preferred_skills = models.PositiveSmallIntegerField(default=10, validators=[MaxValueValidator(100)])
    w_experience = models.PositiveSmallIntegerField(default=15, validators=[MaxValueValidator(100)])
    w_role_alignment = models.PositiveSmallIntegerField(default=15, validators=[MaxValueValidator(100)])
    w_semantic = models.PositiveSmallIntegerField(default=10, validators=[MaxValueValidator(100)])
    w_location = models.PositiveSmallIntegerField(default=10, validators=[MaxValueValidator(100)])
    w_salary = models.PositiveSmallIntegerField(default=5, validators=[MaxValueValidator(100)])
    w_employment_type = models.PositiveSmallIntegerField(default=5, validators=[MaxValueValidator(100)])
    excellent_threshold = models.PositiveSmallIntegerField(default=80, validators=[MaxValueValidator(100)])
    good_threshold = models.PositiveSmallIntegerField(default=65, validators=[MaxValueValidator(100)])
    possible_threshold = models.PositiveSmallIntegerField(default=45, validators=[MaxValueValidator(100)])

    WEIGHT_FIELDS = ["w_required_skills", "w_preferred_skills", "w_experience", "w_role_alignment", "w_semantic",
                     "w_location", "w_salary", "w_employment_type"]

    def weights(self):
        return {f[2:]: getattr(self, f) for f in self.WEIGHT_FIELDS}


class MatchCategory(models.TextChoices):
    EXCELLENT = "excellent", "Excellent match"
    GOOD = "good", "Good match"
    POSSIBLE = "possible", "Possible match"
    LOW = "low", "Low match"


class JobMatch(TimeStampedModel):
    class UserState(models.TextChoices):
        NEW = "new", "New"
        VIEWED = "viewed", "Viewed"
        SHORTLISTED = "shortlisted", "Shortlisted"
        DISMISSED = "dismissed", "Dismissed"

    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="job_matches")
    job = models.ForeignKey("jobs.JobPosting", on_delete=models.CASCADE, related_name="matches")
    score = models.PositiveSmallIntegerField(db_index=True, validators=[MaxValueValidator(100)])
    category = models.CharField(max_length=10, choices=MatchCategory.choices, db_index=True)
    components = models.JSONField(default=dict, help_text="Per-signal score, weight and details")
    skills_matched = ArrayField(models.CharField(max_length=100), default=list, blank=True)
    skills_missing = ArrayField(models.CharField(max_length=100), default=list, blank=True)
    skills_unverified = ArrayField(models.CharField(max_length=100), default=list, blank=True)
    relevant_experience = models.JSONField(default=list, blank=True)
    gaps = ArrayField(models.CharField(max_length=300), default=list, blank=True)
    location_fit = models.CharField(max_length=300, blank=True)
    salary_fit = models.CharField(max_length=300, blank=True)
    explanation = models.TextField(blank=True)
    recommended_action = models.CharField(max_length=300, blank=True)
    semantic_similarity = models.FloatField(null=True, blank=True)
    information_coverage = models.FloatField(default=1.0, help_text="Share of scoring weight with available data")
    excluded_reason = models.CharField(max_length=300, blank=True)
    input_signature = models.CharField(max_length=64, help_text="Hash of profile+job+config inputs; skip recompute if unchanged")
    user_state = models.CharField(max_length=12, choices=UserState.choices, default=UserState.NEW, db_index=True)
    notified_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["-score", "-created_at"]
        constraints = [models.UniqueConstraint(fields=["user", "job"], name="uniq_match_per_user_job")]
        indexes = [models.Index(fields=["user", "category", "-score"]), models.Index(fields=["user", "user_state"])]
