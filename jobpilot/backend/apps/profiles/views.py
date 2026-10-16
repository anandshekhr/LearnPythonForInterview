from django.db import IntegrityError, transaction
from django.http import FileResponse
from drf_spectacular.utils import extend_schema
from rest_framework import generics, mixins, status, viewsets
from rest_framework.decorators import action
from rest_framework.parsers import FormParser, JSONParser, MultiPartParser
from rest_framework.response import Response

from apps.core import audit
from apps.core.exceptions import DomainError

from . import tasks
from .models import CandidateSkill, Certification, Education, EmploymentHistory, Project, Resume, ResumeVersion
from .serializers import (
    ApplyParsedSerializer,
    CandidateProfileSerializer,
    CandidateSkillSerializer,
    CareerPreferenceSerializer,
    CertificationSerializer,
    EducationSerializer,
    EmploymentHistorySerializer,
    ProjectSerializer,
    ResumeDetailSerializer,
    ResumeSerializer,
    ResumeUploadSerializer,
    ResumeVersionEditSerializer,
    ResumeVersionSerializer,
)
from .services.profile import apply_parsed_resume, create_master_snapshot, get_or_create_profile
from .services.validation import validate_resume_upload


class ProfileView(generics.RetrieveUpdateAPIView):
    serializer_class = CandidateProfileSerializer

    def get_object(self):
        return get_or_create_profile(self.request.user)

    def perform_update(self, serializer):
        serializer.save()
        audit.record("profile.updated", obj=serializer.instance, request=self.request, fields=sorted(serializer.validated_data))


class PreferenceView(generics.RetrieveUpdateAPIView):
    serializer_class = CareerPreferenceSerializer

    def get_object(self):
        return get_or_create_profile(self.request.user).preferences


class _ProfileChildViewSet(viewsets.ModelViewSet):
    """CRUD for profile sections, scoped to the requesting user's profile."""

    def get_profile(self):
        return get_or_create_profile(self.request.user)

    def get_queryset(self):
        if getattr(self, "swagger_fake_view", False):
            return self.queryset.none()
        return self.queryset.filter(profile__user=self.request.user)

    def get_serializer_context(self):
        ctx = super().get_serializer_context()
        if self.request and self.request.user.is_authenticated:
            ctx["profile"] = self.get_profile()
        return ctx

    def perform_create(self, serializer):
        # Entries added by hand in the UI are user-verified facts.
        extra = {"verified": True} if "verified" not in serializer.validated_data else {}
        serializer.save(profile=self.get_profile(), **extra)


class CandidateSkillViewSet(_ProfileChildViewSet):
    queryset = CandidateSkill.objects.select_related("skill")
    serializer_class = CandidateSkillSerializer
    filterset_fields = ["verified", "is_primary"]
    search_fields = ["skill__name"]
    ordering_fields = ["skill__name", "years", "updated_at"]


class EmploymentViewSet(_ProfileChildViewSet):
    queryset = EmploymentHistory.objects.all()
    serializer_class = EmploymentHistorySerializer


class EducationViewSet(_ProfileChildViewSet):
    queryset = Education.objects.all()
    serializer_class = EducationSerializer


class ProjectViewSet(_ProfileChildViewSet):
    queryset = Project.objects.all()
    serializer_class = ProjectSerializer


class CertificationViewSet(_ProfileChildViewSet):
    queryset = Certification.objects.all()
    serializer_class = CertificationSerializer


class ResumeViewSet(mixins.ListModelMixin, mixins.RetrieveModelMixin, mixins.DestroyModelMixin, viewsets.GenericViewSet):
    queryset = Resume.objects.all()
    filterset_fields = ["parse_status", "is_master"]
    parser_classes = [JSONParser, MultiPartParser, FormParser]

    def get_queryset(self):
        if getattr(self, "swagger_fake_view", False):  # schema generation
            return Resume.objects.none()
        qs = super().get_queryset().filter(user=self.request.user)
        return qs.defer("raw_text") if self.action == "list" else qs

    def get_serializer_class(self):
        if self.action == "create":
            return ResumeUploadSerializer
        if self.action == "apply":
            return ApplyParsedSerializer
        return ResumeDetailSerializer if self.action == "retrieve" else ResumeSerializer

    def get_throttles(self):
        self.throttle_scope = "upload" if self.action == "create" else None
        return super().get_throttles()

    @extend_schema(request={"multipart/form-data": ResumeUploadSerializer}, responses={201: ResumeSerializer})
    def create(self, request):
        """Upload a PDF/DOCX resume. Parsing runs asynchronously; poll the resume for `parse_status`."""
        serializer = ResumeUploadSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        uploaded = serializer.validated_data["file"]
        kind, content_type, data, sha = validate_resume_upload(uploaded)
        uploaded.seek(0)
        if Resume.objects.filter(user=request.user, sha256=sha).exists():
            raise DomainError("This resume file has already been uploaded.", code="duplicate_resume", status_code=409)
        try:
            with transaction.atomic():
                resume = Resume(
                    user=request.user, title=serializer.validated_data.get("title") or uploaded.name[:200],
                    original_filename=uploaded.name[:255], content_type=content_type, size_bytes=len(data), sha256=sha,
                )
                resume.file.save(f"resume.{kind}", uploaded, save=False)
                resume.save()
        except IntegrityError:
            raise DomainError("This resume file has already been uploaded.", code="duplicate_resume", status_code=409)
        audit.record("resume.uploaded", obj=resume, request=request, size=len(data), kind=kind)
        transaction.on_commit(lambda: tasks.parse_resume.delay(resume.pk))
        return Response(ResumeSerializer(resume).data, status=status.HTTP_201_CREATED)

    def perform_destroy(self, instance):
        audit.record("resume.deleted", obj=instance, request=self.request)
        instance.file.delete(save=False)
        instance.delete()

    @extend_schema(request=None, responses={202: ResumeSerializer})
    @action(detail=True, methods=["post"])
    def parse(self, request, pk=None):
        """Re-run parsing for this resume."""
        resume = self.get_object()
        if resume.parse_status == Resume.ParseStatus.PARSING:
            raise DomainError("Parsing is already in progress.", code="in_progress", status_code=409)
        Resume.objects.filter(pk=resume.pk).update(parse_status=Resume.ParseStatus.PENDING, parse_error="")
        tasks.parse_resume.delay(resume.pk)
        resume.refresh_from_db()
        return Response(ResumeSerializer(resume).data, status=status.HTTP_202_ACCEPTED)

    @extend_schema(request=ApplyParsedSerializer, responses={200: CandidateProfileSerializer})
    @action(detail=True, methods=["post"])
    def apply(self, request, pk=None):
        """Confirm the (edited) parsed draft and write it to the profile as verified data.
        Makes this the master resume and snapshots a new master version."""
        resume = self.get_object()
        if resume.parse_status not in (Resume.ParseStatus.PARSED, Resume.ParseStatus.APPLIED):
            raise DomainError("Resume has not been parsed yet.", code="not_parsed", status_code=409)
        serializer = ApplyParsedSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        data = serializer.validated_data.get("data") or resume.parsed_data
        profile = apply_parsed_resume(resume, data, serializer.validated_data["replace_sections"])
        audit.record("resume.applied", obj=resume, request=request)
        return Response(CandidateProfileSerializer(profile).data)

    @extend_schema(responses={(200, "application/octet-stream"): bytes})
    @action(detail=True, methods=["get"])
    def download(self, request, pk=None):
        resume = self.get_object()
        audit.record("resume.downloaded", obj=resume, request=request)
        return FileResponse(resume.file.open("rb"), as_attachment=True, filename=resume.original_filename,
                            content_type=resume.content_type)


class ResumeVersionViewSet(mixins.ListModelMixin, mixins.RetrieveModelMixin, mixins.UpdateModelMixin, viewsets.GenericViewSet):
    queryset = ResumeVersion.objects.select_related("job", "job__company")
    serializer_class = ResumeVersionSerializer
    filterset_fields = ["kind", "status", "job"]
    ordering_fields = ["version_number", "created_at"]

    http_method_names = ["get", "patch", "post", "head", "options"]

    def get_queryset(self):
        if getattr(self, "swagger_fake_view", False):
            return ResumeVersion.objects.none()
        return super().get_queryset().filter(user=self.request.user)

    def get_serializer_class(self):
        return ResumeVersionEditSerializer if self.action == "partial_update" else ResumeVersionSerializer

    @extend_schema(request=ResumeVersionEditSerializer, responses={200: ResumeVersionSerializer})
    def partial_update(self, request, *args, **kwargs):
        """Edit a tailored draft. Content is re-validated against your verified profile and re-rendered."""
        from apps.documents.services.generation import render_resume_documents
        from apps.documents.services.tailoring import TailoringError, apply_user_edit

        version = self.get_object()
        serializer = ResumeVersionEditSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        if "label" in serializer.validated_data:
            version.label = serializer.validated_data["label"]
            version.save(update_fields=["label", "updated_at"])
        if "content" in serializer.validated_data:
            try:
                apply_user_edit(version, serializer.validated_data["content"])
            except TailoringError as exc:
                raise DomainError(str(exc), code="unverified_content")
            old = list(version.documents.filter(kind="resume"))
            template = old[0].template if old else "classic"
            formats = [d.format for d in old] or ["pdf", "docx"]
            for doc in old:
                doc.file.delete(save=False)
                doc.delete()
            render_resume_documents(version, template, formats)
            audit.record("resume_version.edited", obj=version, request=request)
        return Response(ResumeVersionSerializer(version).data)

    @extend_schema(request=None, responses={200: ResumeVersionSerializer})
    @action(detail=True, methods=["post"])
    def approve(self, request, pk=None):
        """Approve a tailored version so it can be used in an application."""
        from apps.documents.services.generation import approve_version

        return Response(ResumeVersionSerializer(approve_version(self.get_object(), request.user)).data)

    @extend_schema(request=None, responses={200: ResumeVersionSerializer})
    @action(detail=True, methods=["post"])
    def reject(self, request, pk=None):
        from apps.documents.services.generation import reject_version

        return Response(ResumeVersionSerializer(reject_version(self.get_object(), request.user)).data)

    @extend_schema(responses={200: dict})
    @action(detail=True, methods=["get"])
    def compare(self, request, pk=None):
        """Side-by-side comparison of the master snapshot and this tailored version."""
        from apps.documents.services.tailoring import compare

        return Response(compare(self.get_object()))

    @extend_schema(request=None, responses={201: ResumeVersionSerializer})
    @action(detail=False, methods=["post"])
    def snapshot(self, request):
        """Snapshot the current verified profile as a new approved master version."""
        version = create_master_snapshot(get_or_create_profile(request.user))
        audit.record("resume_version.snapshot", obj=version, request=request)
        return Response(ResumeVersionSerializer(version).data, status=status.HTTP_201_CREATED)
