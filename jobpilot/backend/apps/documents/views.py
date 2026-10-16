from django.http import FileResponse
from drf_spectacular.utils import extend_schema
from rest_framework import mixins, status, viewsets
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.views import APIView

from apps.core import audit
from apps.core.exceptions import DomainError
from apps.jobs.models import JobPosting

from . import tasks
from .models import GeneratedDocument
from .serializers import GeneratedDocumentSerializer, GenerateRequestSerializer
from .services.generation import render_cover_letter


class GenerateDocumentsView(APIView):
    @extend_schema(request=GenerateRequestSerializer, responses={202: dict})
    def post(self, request):
        """Generate a tailored resume (PDF + DOCX) and optional cover letter for a job. Runs asynchronously;
        the new resume version appears under /resume-versions/?job=<id> with status awaiting_approval."""
        serializer = GenerateRequestSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        job = JobPosting.objects.filter(pk=serializer.validated_data["job"]).first()
        if job is None:
            raise DomainError("Job not found.", code="not_found", status_code=404)
        result = tasks.generate_documents.delay(request.user.pk, job.pk, serializer.validated_data["include_cover_letter"],
                                                serializer.validated_data["template"])
        payload = {"queued": True, "task_id": result.id}
        if result.ready():
            payload["result"] = result.result
        return Response(payload, status=status.HTTP_202_ACCEPTED)


class GeneratedDocumentViewSet(mixins.ListModelMixin, mixins.RetrieveModelMixin, mixins.UpdateModelMixin,
                               mixins.DestroyModelMixin, viewsets.GenericViewSet):
    serializer_class = GeneratedDocumentSerializer
    filterset_fields = ["kind", "status", "job", "resume_version", "format"]
    ordering_fields = ["created_at"]
    http_method_names = ["get", "patch", "post", "delete", "head", "options"]

    def get_queryset(self):
        if getattr(self, "swagger_fake_view", False):  # schema generation
            return GeneratedDocument.objects.none()
        return GeneratedDocument.objects.filter(user=self.request.user).select_related("job", "job__company")

    def perform_update(self, serializer):
        doc = serializer.instance
        if doc.kind != GeneratedDocument.Kind.COVER_LETTER or doc.status == GeneratedDocument.Status.APPROVED:
            raise DomainError("Only cover letters awaiting approval can be edited.", code="invalid_state", status_code=409)
        text = serializer.validated_data.get("text_content", doc.text_content)
        render_cover_letter(doc.resume_version, doc.job, text, doc.template, doc=doc)
        audit.record("cover_letter.edited", obj=doc, request=self.request)

    def perform_destroy(self, instance):
        if instance.resume_applications.exists():
            raise DomainError("Document is attached to an application.", code="in_use", status_code=409)
        instance.file.delete(save=False)
        instance.delete()

    @action(detail=True, methods=["get"])
    def download(self, request, pk=None):
        doc = self.get_object()
        if not doc.file:
            raise DomainError("No file for this document.", code="not_found", status_code=404)
        ext = doc.format if doc.format != "text" else "txt"
        audit.record("document.downloaded", obj=doc, request=request)
        return FileResponse(doc.file.open("rb"), as_attachment=request.query_params.get("inline") != "1",
                            filename=f"{doc.kind}-{doc.pk}.{ext}", content_type=doc.content_type)

    @extend_schema(request=None, responses={200: GeneratedDocumentSerializer})
    @action(detail=True, methods=["post"])
    def approve(self, request, pk=None):
        """Approve a cover letter (resumes are approved through their resume version)."""
        doc = self.get_object()
        if doc.kind != GeneratedDocument.Kind.COVER_LETTER:
            raise DomainError("Approve resumes via /resume-versions/{id}/approve/.", code="invalid_kind")
        from django.utils import timezone

        doc.status, doc.approved_at = GeneratedDocument.Status.APPROVED, timezone.now()
        doc.save(update_fields=["status", "approved_at", "updated_at"])
        audit.record("cover_letter.approved", obj=doc, request=request)
        return Response(GeneratedDocumentSerializer(doc).data)

