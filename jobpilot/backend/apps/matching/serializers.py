from rest_framework import serializers

from apps.jobs.serializers import JobPostingListSerializer

from .models import JobMatch, ScoringConfig


class JobMatchSerializer(serializers.ModelSerializer):
    job = JobPostingListSerializer(read_only=True)

    class Meta:
        model = JobMatch
        fields = ["id", "job", "score", "category", "components", "skills_matched", "skills_missing", "skills_unverified",
                  "relevant_experience", "gaps", "location_fit", "salary_fit", "explanation", "recommended_action",
                  "semantic_similarity", "information_coverage", "excluded_reason", "user_state", "created_at", "updated_at"]
        read_only_fields = [f for f in fields if f != "user_state"]


class ScoringConfigSerializer(serializers.ModelSerializer):
    class Meta:
        model = ScoringConfig
        exclude = ["id", "user", "created_at"]
        read_only_fields = ["updated_at"]

    def validate(self, attrs):
        merged = {f: attrs.get(f, getattr(self.instance, f, 0)) for f in ScoringConfig.WEIGHT_FIELDS}
        if sum(merged.values()) == 0:
            raise serializers.ValidationError("At least one weight must be greater than zero.")
        t = {k: attrs.get(k, getattr(self.instance, k)) for k in ("excellent_threshold", "good_threshold", "possible_threshold")}
        if not (t["excellent_threshold"] > t["good_threshold"] > t["possible_threshold"]):
            raise serializers.ValidationError("Thresholds must satisfy excellent > good > possible.")
        return attrs
