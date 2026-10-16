from django.db.models import Count, Prefetch, Q
from django.utils import timezone
from drf_spectacular.utils import extend_schema
from rest_framework import mixins, permissions, status, viewsets
from rest_framework.decorators import action
from rest_framework.response import Response

from apps.applications.models import Application
from apps.core import audit
from apps.core.exceptions import DomainError
from apps.matching.models import JobMatch

from . import tasks
from .adapters.registry import ADAPTERS, adapter_for
from .filters import JobPostingFilter
from .http import SourceHTTPError
from .models import Company, JobPosting, JobSource
from .serializers import (
    CompanySerializer,
    JobPostingChangeSerializer,
    JobPostingDetailSerializer,
    JobPostingListSerializer,
    JobSourceSerializer,
    SubmissionCredentialSerializer,
)


class IsCreatorOrStaffForWrites(permissions.BasePermission):
    def has_object_permission(self, request, view, obj):
        if request.method in permissions.SAFE_METHODS:
            return True
        return request.user.is_staff or obj.created_by_id == request.user.id


class CompanyViewSet(viewsets.ModelViewSet):
    """Shared company registry. Anyone can add a company; only its creator or staff can change it."""

    serializer_class = CompanySerializer
    permission_classes = [permissions.IsAuthenticated, IsCreatorOrStaffForWrites]
    search_fields = ["name", "industry", "company_type"]
    filterset_fields = ["company_type", "industry"]
    ordering_fields = ["name", "created_at", "active_jobs"]

    def get_queryset(self):
        return Company.objects.annotate(
            source_count=Count("sources", distinct=True),
            active_jobs=Count("jobs", filter=Q(jobs__status=JobPosting.Status.ACTIVE), distinct=True),
        )

    def perform_create(self, serializer):
        company = serializer.save(created_by=self.request.user)
        audit.record("company.created", obj=company, request=self.request)


class JobSourceViewSet(viewsets.ModelViewSet):
    serializer_class = JobSourceSerializer
    permission_classes = [permissions.IsAuthenticated, IsCreatorOrStaffForWrites]
    filterset_fields = ["source_type", "status", "enabled", "company"]
    search_fields = ["company__name", "identifier"]
    ordering_fields = ["company__name", "last_synced_at", "active_job_count"]

    def get_queryset(self):
        return JobSource.objects.select_related("company")

    def get_throttles(self):
        self.throttle_scope = "sync" if self.action == "sync" else None
        return super().get_throttles()

    def perform_create(self, serializer):
        source = serializer.save(created_by=self.request.user)
        audit.record("source.created", obj=source, request=self.request, source_type=source.source_type)
        tasks.sync_source_task.delay(source.pk, self.request.user.pk)

    @extend_schema(responses={200: dict})
    @action(detail=False, methods=["get"])
    def types(self, request):
        """Supported source types and their documented capabilities and limitations."""
        return Response([{"source_type": key, "label": JobSource.SourceType(key).label, "capabilities": cls.capabilities.as_dict()}
                         for key, cls in ADAPTERS.items()])

    @extend_schema(request=None, responses={202: dict})
    @action(detail=True, methods=["post"])
    def sync(self, request, pk=None):
        """Queue an immediate synchronisation of this source."""
        source = self.get_object()
        if not source.enabled:
            raise DomainError("Source is disabled.", code="source_disabled", status_code=409)
        result = tasks.sync_source_task.delay(source.pk, request.user.pk)
        return Response({"queued": True, "task_id": result.id}, status=status.HTTP_202_ACCEPTED)

    @extend_schema(request=None, responses={200: dict})
    @action(detail=True, methods=["post"])
    def test(self, request, pk=None):
        """Check the source is reachable and count its published jobs (does not store anything)."""
        source = self.get_object()
        adapter = adapter_for(source)
        try:
            count = sum(1 for _ in adapter.fetch_listings())
        except SourceHTTPError as exc:
            return Response({"ok": False, "error": str(exc)})
        finally:
            adapter.http.close()
        return Response({"ok": True, "published_jobs": count})

    @extend_schema(request=SubmissionCredentialSerializer, responses={200: JobSourceSerializer})
    @action(detail=True, methods=["post"], url_path="submission-credentials",
            permission_classes=[permissions.IsAuthenticated, permissions.IsAdminUser])
    def submission_credentials(self, request, pk=None):
        """Staff only: store an employer-issued API key enabling authorized API submission."""
        source = self.get_object()
        if not ADAPTERS[source.source_type].capabilities.api_submission:
            raise DomainError("This source type has no submission API.", code="unsupported")
        serializer = SubmissionCredentialSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        data = serializer.validated_data
        if data["api_key"]:
            source.set_submission_api_key(data["api_key"])
        if data["submission_enabled"] and not source.has_submission_key:
            raise DomainError("An API key is required to enable submission.", code="missing_key")
        source.submission_enabled = data["submission_enabled"]
        source.submission_terms_confirmed_at = timezone.now() if data["confirm_authorized"] else None
        source.save()
        audit.record("source.submission_credentials", obj=source, request=request, enabled=source.submission_enabled)
        return Response(JobSourceSerializer(source, context={"request": request}).data)


class JobPostingViewSet(mixins.ListModelMixin, mixins.RetrieveModelMixin, viewsets.GenericViewSet):
    filterset_class = JobPostingFilter
    ordering_fields = ["published_at", "first_seen_at", "title", "matches__score"]
    ordering = ["-first_seen_at"]

    def get_queryset(self):
        if getattr(self, "swagger_fake_view", False):  # schema generation
            return JobPosting.objects.none()
        user = self.request.user
        qs = JobPosting.objects.select_related("company", "source").prefetch_related(
            Prefetch("matches", queryset=JobMatch.objects.filter(user=user), to_attr="my_matches"),
            Prefetch("applications", queryset=Application.objects.filter(user=user).only("id", "job_id"), to_attr="my_applications"),
        )
        if self.action == "list":
            qs = qs.defer("description_text", "embedding", "raw_metadata", "requirements")
        return qs.distinct() if self.request.query_params.get("skill") else qs

    def get_serializer_class(self):
        return JobPostingDetailSerializer if self.action == "retrieve" else JobPostingListSerializer

    def retrieve(self, request, *args, **kwargs):
        job = self.get_object()
        JobMatch.objects.filter(user=request.user, job=job, user_state=JobMatch.UserState.NEW).update(
            user_state=JobMatch.UserState.VIEWED)
        return Response(self.get_serializer(job).data)

    @extend_schema(responses={200: JobPostingChangeSerializer(many=True)})
    @action(detail=True, methods=["get"])
    def changes(self, request, pk=None):
        job = self.get_object()
        return Response(JobPostingChangeSerializer(job.changes.all()[:50], many=True).data)
