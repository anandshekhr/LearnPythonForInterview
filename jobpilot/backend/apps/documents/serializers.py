from rest_framework import serializers

from .models import GeneratedDocument
from .services.render import TEMPLATES


class GeneratedDocumentSerializer(serializers.ModelSerializer):
    job_title = serializers.CharField(source="job.title", read_only=True, default=None)
    company_name = serializers.CharField(source="job.company.name", read_only=True, default=None)

    class Meta:
        model = GeneratedDocument
        fields = ["id", "kind", "format", "template", "status", "title", "text_content", "size_bytes", "generator", "warnings",
                  "job", "job_title", "company_name", "resume_version", "approved_at", "created_at", "updated_at"]
        read_only_fields = [f for f in fields if f != "text_content"]


class GenerateRequestSerializer(serializers.Serializer):
    job = serializers.IntegerField()
    include_cover_letter = serializers.BooleanField(default=True)
    template = serializers.ChoiceField(choices=sorted(TEMPLATES), default="classic")
