from rest_framework import serializers

from apps.core.skills import canonicalize

from .models import (
    CandidateProfile,
    CandidateSkill,
    CareerPreference,
    Certification,
    Education,
    EmploymentHistory,
    EmploymentType,
    Project,
    Resume,
    ResumeVersion,
    WorkModel,
)
from .services.profile import get_skill


class CareerPreferenceSerializer(serializers.ModelSerializer):
    work_models = serializers.ListField(child=serializers.ChoiceField(choices=WorkModel.choices), required=False)
    employment_types = serializers.ListField(child=serializers.ChoiceField(choices=EmploymentType.choices), required=False)

    class Meta:
        model = CareerPreference
        exclude = ["id", "profile", "created_at"]
        read_only_fields = ["updated_at"]

    def validate(self, attrs):
        lo = attrs.get("salary_min", getattr(self.instance, "salary_min", None))
        hi = attrs.get("salary_max", getattr(self.instance, "salary_max", None))
        if lo is not None and hi is not None and lo > hi:
            raise serializers.ValidationError({"salary_max": "Must be greater than or equal to salary_min."})
        for key in ("target_roles", "preferred_locations", "industries", "company_types", "excluded_companies", "excluded_keywords"):
            if key in attrs:
                attrs[key] = list(dict.fromkeys(v.strip() for v in attrs[key] if v and v.strip()))[:50]
        return attrs


class CandidateProfileSerializer(serializers.ModelSerializer):
    preferences = CareerPreferenceSerializer(read_only=True)
    completeness = serializers.SerializerMethodField()

    class Meta:
        model = CandidateProfile
        exclude = ["user", "embedding", "embedding_hash"]
        read_only_fields = ["id", "created_at", "updated_at", "verified_at"]

    def get_completeness(self, obj) -> dict:
        checks = {
            "contact": bool(obj.full_name and obj.email),
            "summary": bool(obj.summary or obj.headline),
            "experience": obj.employment.filter(verified=True).exists(),
            "skills": obj.skills.filter(verified=True).count() >= 3,
            "education": obj.education.exists(),
            "preferences": bool(getattr(obj, "preferences", None) and obj.preferences.target_roles),
        }
        return {"score": round(100 * sum(checks.values()) / len(checks)), "checks": checks}


class CandidateSkillSerializer(serializers.ModelSerializer):
    name = serializers.CharField(source="skill.name", read_only=True)
    category = serializers.CharField(source="skill.category", read_only=True)
    skill_name = serializers.CharField(write_only=True, required=False)

    class Meta:
        model = CandidateSkill
        fields = ["id", "name", "category", "skill_name", "years", "proficiency", "is_primary", "verified", "source", "updated_at"]
        read_only_fields = ["id", "source", "updated_at"]

    def validate(self, attrs):
        skill_name = attrs.pop("skill_name", None)
        if self.instance is None and not skill_name:
            raise serializers.ValidationError({"skill_name": "This field is required."})
        if skill_name:
            if not canonicalize(skill_name):
                raise serializers.ValidationError({"skill_name": "Invalid skill name."})
            attrs["skill"] = get_skill(skill_name)
            profile = self.context["profile"]
            qs = CandidateSkill.objects.filter(profile=profile, skill=attrs["skill"])
            if self.instance is not None:
                qs = qs.exclude(pk=self.instance.pk)
            if qs.exists():
                raise serializers.ValidationError({"skill_name": "Skill already in profile."})
        return attrs


class _ProfileChildSerializer(serializers.ModelSerializer):
    class Meta:
        exclude = ["profile"]
        read_only_fields = ["id", "created_at", "updated_at"]


class EmploymentHistorySerializer(_ProfileChildSerializer):
    class Meta(_ProfileChildSerializer.Meta):
        model = EmploymentHistory

    def validate(self, attrs):
        start = attrs.get("start_date", getattr(self.instance, "start_date", None))
        end = attrs.get("end_date", getattr(self.instance, "end_date", None))
        if start and end and end < start:
            raise serializers.ValidationError({"end_date": "End date cannot be before start date."})
        if attrs.get("is_current"):
            attrs["end_date"] = None
        if "technologies" in attrs:
            attrs["technologies"] = [canonicalize(t) for t in attrs["technologies"] if t]
        return attrs


class EducationSerializer(_ProfileChildSerializer):
    class Meta(_ProfileChildSerializer.Meta):
        model = Education


class ProjectSerializer(_ProfileChildSerializer):
    class Meta(_ProfileChildSerializer.Meta):
        model = Project


class CertificationSerializer(_ProfileChildSerializer):
    class Meta(_ProfileChildSerializer.Meta):
        model = Certification


class ResumeSerializer(serializers.ModelSerializer):
    class Meta:
        model = Resume
        fields = ["id", "title", "original_filename", "content_type", "size_bytes", "is_master", "parse_status",
                  "parse_error", "parser", "created_at", "updated_at"]
        read_only_fields = fields


class ResumeDetailSerializer(ResumeSerializer):
    class Meta(ResumeSerializer.Meta):
        fields = ResumeSerializer.Meta.fields + ["parsed_data"]
        read_only_fields = fields


class ResumeUploadSerializer(serializers.Serializer):
    file = serializers.FileField()
    title = serializers.CharField(max_length=200, required=False, allow_blank=True)


class ApplyParsedSerializer(serializers.Serializer):
    data = serializers.DictField(required=False, help_text="Reviewed/edited draft. Defaults to the stored parsed draft.")
    replace_sections = serializers.BooleanField(default=True)


class ResumeVersionSerializer(serializers.ModelSerializer):
    job_title = serializers.CharField(source="job.title", read_only=True, default=None)
    company_name = serializers.CharField(source="job.company.name", read_only=True, default=None)

    class Meta:
        model = ResumeVersion
        fields = ["id", "version_number", "label", "kind", "status", "job", "job_title", "company_name", "base_version",
                  "source_resume", "content", "tailoring_notes", "generator", "approved_at", "created_at", "updated_at"]
        read_only_fields = [f for f in fields if f not in ("label",)]


class ResumeVersionEditSerializer(serializers.Serializer):
    """Edits to a tailored draft. Only selection/ordering/wording of existing facts is
    allowed; the tailoring service re-validates content against the verified profile."""

    label = serializers.CharField(max_length=200, required=False)
    content = serializers.DictField(required=False)
