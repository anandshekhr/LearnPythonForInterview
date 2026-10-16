from rest_framework import serializers


from .models import (
    Application,
    ApplicationAnswer,
    ApplicationStatus,
    ApplicationStatusHistory,
    FollowUp,
    Interview,
    SubmissionChannel,
)


class ApplicationStatusHistorySerializer(serializers.ModelSerializer):
    class Meta:
        model = ApplicationStatusHistory
        fields = ["id", "from_status", "to_status", "actor", "note", "created_at"]


class ApplicationAnswerSerializer(serializers.ModelSerializer):
    class Meta:
        model = ApplicationAnswer
        fields = ["id", "question_key", "question_label", "field_type", "required", "options", "answer", "origin", "sensitive",
                  "reviewed_at"]
        read_only_fields = [f for f in fields if f not in ("answer",)]


class AnswerUpdateSerializer(serializers.Serializer):
    id = serializers.IntegerField()
    answer = serializers.CharField(allow_blank=True, max_length=10000)
    reviewed = serializers.BooleanField(default=False, help_text="Required true for sensitive questions you have reviewed")


class InterviewSerializer(serializers.ModelSerializer):
    company_name = serializers.CharField(source="application.company_name", read_only=True)
    job_title = serializers.CharField(source="application.job_title", read_only=True)

    class Meta:
        model = Interview
        fields = ["id", "application", "company_name", "job_title", "round_name", "scheduled_at", "duration_minutes", "mode",
                  "location", "interviewers", "notes", "outcome", "created_at", "updated_at"]
        read_only_fields = ["id", "created_at", "updated_at"]

    def validate_application(self, value):
        if value.user_id != self.context["request"].user.id:
            raise serializers.ValidationError("Application not found.")
        return value


class FollowUpSerializer(serializers.ModelSerializer):
    company_name = serializers.CharField(source="application.company_name", read_only=True)
    job_title = serializers.CharField(source="application.job_title", read_only=True)

    class Meta:
        model = FollowUp
        fields = ["id", "application", "company_name", "job_title", "due_date", "note", "completed_at", "created_at"]
        read_only_fields = ["id", "created_at"]

    def validate_application(self, value):
        if value.user_id != self.context["request"].user.id:
            raise serializers.ValidationError("Application not found.")
        return value


class ApplicationListSerializer(serializers.ModelSerializer):
    resume_version_label = serializers.CharField(source="resume_version.label", read_only=True, default=None)
    next_interview = serializers.SerializerMethodField()

    class Meta:
        model = Application
        fields = ["id", "job", "company_name", "job_title", "job_url", "source_type", "match_score", "status", "mode",
                  "resume_version", "resume_version_label", "cover_letter", "submitted_at", "submission_channel",
                  "submission_reference", "follow_up_date", "last_verified_at", "next_interview", "created_at", "updated_at"]

    def get_next_interview(self, obj) -> str | None:
        upcoming = [i for i in getattr(obj, "upcoming_interviews", [])]
        return upcoming[0].scheduled_at if upcoming else None


class ApplicationDetailSerializer(ApplicationListSerializer):
    history = ApplicationStatusHistorySerializer(many=True, read_only=True)
    answers = ApplicationAnswerSerializer(many=True, read_only=True)
    interviews = InterviewSerializer(many=True, read_only=True)
    follow_ups = FollowUpSerializer(many=True, read_only=True)
    allowed_transitions = serializers.SerializerMethodField()

    class Meta(ApplicationListSerializer.Meta):
        fields = ApplicationListSerializer.Meta.fields + [
            "notes", "approved_at", "submission_response", "submission_attempts", "last_error", "resume_document", "history",
            "answers", "interviews", "follow_ups", "allowed_transitions"]
        read_only_fields = [f for f in fields if f not in ("notes", "follow_up_date")]

    def get_allowed_transitions(self, obj) -> list[str]:
        from .models import MANUAL_ALLOWED, TRANSITIONS

        return sorted(s for s in TRANSITIONS.get(obj.status, set()) if s in MANUAL_ALLOWED)


class ApplicationCreateSerializer(serializers.Serializer):
    job = serializers.IntegerField()
    status = serializers.ChoiceField(choices=[ApplicationStatus.DISCOVERED, ApplicationStatus.SHORTLISTED],
                                     default=ApplicationStatus.SHORTLISTED)


class StatusChangeSerializer(serializers.Serializer):
    status = serializers.ChoiceField(choices=ApplicationStatus.choices)
    note = serializers.CharField(required=False, allow_blank=True, max_length=1000)


class AttachResumeSerializer(serializers.Serializer):
    resume_version = serializers.IntegerField()
    cover_letter = serializers.IntegerField(required=False, allow_null=True)


class SubmitSerializer(serializers.Serializer):
    confirm_retry = serializers.BooleanField(default=False)


class ConfirmSubmittedSerializer(serializers.Serializer):
    channel = serializers.ChoiceField(choices=[SubmissionChannel.MANUAL, SubmissionChannel.BROWSER_ASSISTED],
                                      default=SubmissionChannel.MANUAL)
    reference = serializers.CharField(required=False, allow_blank=True, max_length=200)
    note = serializers.CharField(required=False, allow_blank=True, max_length=1000)


