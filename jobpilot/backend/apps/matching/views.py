import django_filters
from drf_spectacular.utils import extend_schema
from rest_framework import generics, mixins, status, viewsets
from rest_framework.decorators import action
from rest_framework.response import Response

from apps.jobs.serializers import JobPostingListSerializer

from . import services, tasks
from .models import JobMatch
from .serializers import JobMatchSerializer, ScoringConfigSerializer


class JobMatchFilter(django_filters.FilterSet):
    min_score = django_filters.NumberFilter(field_name="score", lookup_expr="gte")
    job_status = django_filters.CharFilter(field_name="job__status")
    work_model = django_filters.CharFilter(field_name="job__work_model")
    company = django_filters.NumberFilter(field_name="job__company")
    hide_excluded = django_filters.BooleanFilter(method="excluded")

    class Meta:
        model = JobMatch
        fields = ["category", "user_state", "job"]

    def excluded(self, qs, name, value):
        return qs.filter(excluded_reason="") if value else qs


class JobMatchViewSet(mixins.ListModelMixin, mixins.RetrieveModelMixin, mixins.UpdateModelMixin, viewsets.GenericViewSet):
    """Recommended jobs: explainable match scores for the current user."""

    serializer_class = JobMatchSerializer
    filterset_class = JobMatchFilter
    ordering_fields = ["score", "created_at", "job__published_at"]
    ordering = ["-score"]
    search_fields = ["job__title", "job__company__name"]
    http_method_names = ["get", "patch", "post", "head", "options"]

    def get_queryset(self):
        if getattr(self, "swagger_fake_view", False):  # schema generation
            return JobMatch.objects.none()
        return JobMatch.objects.filter(user=self.request.user, job__duplicate_of__isnull=True).select_related(
            "job", "job__company", "job__source")

    @extend_schema(request=None, responses={202: dict})
    @action(detail=False, methods=["post"])
    def rematch(self, request):
        """Recompute scores for all active jobs (e.g. after editing the profile or weights)."""
        result = tasks.rematch_user.delay(request.user.pk, True)
        return Response({"queued": True, "task_id": result.id}, status=status.HTTP_202_ACCEPTED)

    @extend_schema(responses={200: JobPostingListSerializer(many=True)})
    @action(detail=False, methods=["get"])
    def semantic(self, request):
        """Jobs closest to your profile by embedding similarity (pgvector), regardless of score."""
        jobs = services.semantic_search(request.user, limit=min(int(request.query_params.get("limit", 20)), 50))
        data = JobPostingListSerializer(jobs, many=True, context={"request": request}).data
        for row, job in zip(data, jobs):
            row["similarity"] = round(1 - job.distance, 3)
        return Response(data)


class ScoringConfigView(generics.RetrieveUpdateAPIView):
    serializer_class = ScoringConfigSerializer

    def get_object(self):
        return services.get_config(self.request.user)
