from rest_framework import serializers

from .adapters.base import AdapterConfigError
from .adapters.registry import ADAPTERS, get_adapter_class
from .models import Company, JobPosting, JobPostingChange, JobSource


class CompanySerializer(serializers.ModelSerializer):
    source_count = serializers.IntegerField(read_only=True, default=0)
    active_jobs = serializers.IntegerField(read_only=True, default=0)
    can_edit = serializers.SerializerMethodField()

    class Meta:
        model = Company
        fields = ["id", "name", "slug", "website", "careers_url", "industry", "company_type", "headquarters", "notes",
                  "source_count", "active_jobs", "can_edit", "created_at", "updated_at"]
        read_only_fields = ["id", "slug", "created_at", "updated_at"]

    def get_can_edit(self, obj) -> bool:
        user = self.context["request"].user
        return user.is_staff or obj.created_by_id == user.id


class JobSourceSerializer(serializers.ModelSerializer):
    company_name = serializers.CharField(source="company.name", read_only=True)
    capabilities = serializers.SerializerMethodField()
    has_submission_key = serializers.BooleanField(read_only=True)
    can_edit = serializers.SerializerMethodField()

    class Meta:
        model = JobSource
        fields = ["id", "company", "company_name", "source_type", "identifier", "config", "enabled", "status",
                  "sync_interval_minutes", "min_request_interval_seconds", "last_synced_at", "last_success_at", "last_error",
                  "last_error_at", "consecutive_failures", "active_job_count", "submission_enabled", "has_submission_key",
                  "submission_terms_confirmed_at", "capabilities", "can_edit", "created_at"]
        read_only_fields = ["id", "status", "last_synced_at", "last_success_at", "last_error", "last_error_at",
                            "consecutive_failures", "active_job_count", "submission_enabled", "submission_terms_confirmed_at",
                            "created_at"]

    def get_capabilities(self, obj) -> dict:
        caps = get_adapter_class(obj.source_type).capabilities.as_dict()
        caps["api_submission_active"] = bool(caps["api_submission"] and obj.submission_enabled and obj.has_submission_key)
        return caps

    def get_can_edit(self, obj) -> bool:
        user = self.context["request"].user
        return user.is_staff or obj.created_by_id == user.id

    def validate_sync_interval_minutes(self, value):
        if value < 60:
            raise serializers.ValidationError("Minimum sync interval is 60 minutes (be polite to job boards).")
        return value

    def validate_min_request_interval_seconds(self, value):
        if value < 0.5:
            raise serializers.ValidationError("Minimum is 0.5 seconds between requests.")
        return value

    def validate(self, attrs):
        source_type = attrs.get("source_type", getattr(self.instance, "source_type", None))
        if source_type not in ADAPTERS:
            raise serializers.ValidationError({"source_type": "Unsupported source type."})
        if "identifier" in attrs:
            try:
                attrs["identifier"] = get_adapter_class(source_type).validate_identifier(attrs["identifier"])
            except AdapterConfigError as exc:
                raise serializers.ValidationError({"identifier": str(exc)})
        config = attrs.get("config")
        if config is not None:
            allowed = {"region"}
            if set(config) - allowed:
                raise serializers.ValidationError({"config": f"Allowed keys: {sorted(allowed)}"})
            if config.get("region") not in (None, "eu", "us"):
                raise serializers.ValidationError({"config": "region must be 'us' or 'eu'."})
        return attrs


class SubmissionCredentialSerializer(serializers.Serializer):
    api_key = serializers.CharField(write_only=True, allow_blank=True, max_length=500)
    submission_enabled = serializers.BooleanField()
    confirm_authorized = serializers.BooleanField(
        help_text="Confirm the key was issued by the employer for this board and that API submission is permitted."
    )

    def validate(self, attrs):
        if attrs["submission_enabled"] and not attrs["confirm_authorized"]:
            raise serializers.ValidationError("You must confirm that API submission is authorized for this board.")
        return attrs


class MatchSummarySerializer(serializers.Serializer):
    id = serializers.IntegerField()
    score = serializers.IntegerField()
    category = serializers.CharField()
    user_state = serializers.CharField()


class JobPostingListSerializer(serializers.ModelSerializer):
    company_name = serializers.CharField(source="company.name", read_only=True)
    source_type = serializers.CharField(source="source.source_type", read_only=True)
    official_url = serializers.CharField(read_only=True)
    my_match = serializers.SerializerMethodField()
    application_id = serializers.SerializerMethodField()

    class Meta:
        model = JobPosting
        fields = ["id", "title", "company", "company_name", "source_type", "department", "location_raw", "locations",
                  "work_model", "employment_type", "seniority", "salary_min", "salary_max", "salary_currency", "salary_text",
                  "published_at", "first_seen_at", "last_seen_at", "status", "expired_at", "official_url", "duplicate_of",
                  "my_match", "application_id"]

    def get_my_match(self, obj) -> dict | None:
        matches = getattr(obj, "my_matches", None)
        if matches is None:
            return None
        return MatchSummarySerializer(matches[0]).data if matches else None

    def get_application_id(self, obj) -> int | None:
        apps = getattr(obj, "my_applications", None)
        return apps[0].id if apps else None


class JobPostingDetailSerializer(JobPostingListSerializer):
    requirements = serializers.JSONField(read_only=True)
    company_detail = serializers.SerializerMethodField()

    class Meta(JobPostingListSerializer.Meta):
        fields = JobPostingListSerializer.Meta.fields + ["description_text", "requirements", "min_years_experience",
                                                         "canonical_url", "apply_url", "description_version", "source_updated_at",
                                                         "company_detail"]

    def get_company_detail(self, obj) -> dict:
        c = obj.company
        return {"id": c.id, "name": c.name, "website": c.website, "careers_url": c.careers_url, "industry": c.industry,
                "company_type": c.company_type}


class JobPostingChangeSerializer(serializers.ModelSerializer):
    class Meta:
        model = JobPostingChange
        fields = ["id", "detected_at", "change_type", "fields_changed"]
