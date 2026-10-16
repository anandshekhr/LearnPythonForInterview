from django.utils import timezone
from rest_framework import serializers

from .models import ApplicationMode, AutomationRun, AutomationSettings, ScheduledSearch


class AutomationSettingsSerializer(serializers.ModelSerializer):
    automated_submission_consent = serializers.BooleanField(
        write_only=True, required=False,
        help_text="Required true to switch to automated mode. I confirm JobPilot may submit applications on my behalf "
                  "through authorized employer APIs, within my limits, using only approved documents.")
    automated_submission_allowed = serializers.BooleanField(read_only=True)

    class Meta:
        model = AutomationSettings
        exclude = ["id", "user", "created_at"]
        read_only_fields = ["automated_submission_consent_at", "updated_at"]

    def validate(self, attrs):
        consent = attrs.pop("automated_submission_consent", None)
        mode = attrs.get("application_mode", getattr(self.instance, "application_mode", ApplicationMode.DISCOVERY))
        if consent is True:
            attrs["automated_submission_consent_at"] = timezone.now()
        elif consent is False:
            attrs["automated_submission_consent_at"] = None
        has_consent = attrs.get("automated_submission_consent_at", getattr(self.instance, "automated_submission_consent_at", None))
        if mode == ApplicationMode.AUTOMATED and not has_consent:
            raise serializers.ValidationError({"automated_submission_consent": "Explicit consent is required for automated mode."})
        return attrs


class AutomationRunSerializer(serializers.ModelSerializer):
    duration_seconds = serializers.FloatField(read_only=True)

    class Meta:
        model = AutomationRun
        fields = ["id", "kind", "status", "target_type", "target_id", "started_at", "finished_at", "duration_seconds", "stats", "error"]


class ScheduledSearchSerializer(serializers.ModelSerializer):
    class Meta:
        model = ScheduledSearch
        fields = ["id", "name", "enabled", "interval_hours", "sources", "last_run_at", "next_run_at", "created_at"]
        read_only_fields = ["id", "last_run_at", "created_at"]

    def validate_name(self, value):
        qs = ScheduledSearch.objects.filter(user=self.context["request"].user, name=value)
        if self.instance:
            qs = qs.exclude(pk=self.instance.pk)
        if qs.exists():
            raise serializers.ValidationError("You already have a search with this name.")
        return value
