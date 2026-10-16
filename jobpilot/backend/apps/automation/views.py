from django.utils import timezone
from drf_spectacular.utils import extend_schema
from rest_framework import generics, mixins, viewsets
from rest_framework.decorators import action
from rest_framework.response import Response

from apps.core import audit
from apps.core.permissions import OwnedQuerysetMixin

from . import tasks
from .models import AutomationRun, ScheduledSearch
from .serializers import AutomationRunSerializer, AutomationSettingsSerializer, ScheduledSearchSerializer
from .services import get_settings


class AutomationSettingsView(generics.RetrieveUpdateAPIView):
    serializer_class = AutomationSettingsSerializer

    def get_object(self):
        return get_settings(self.request.user)

    def perform_update(self, serializer):
        before = serializer.instance.application_mode
        obj = serializer.save()
        audit.record("automation.settings_updated", obj=obj, request=self.request, mode_from=before, mode_to=obj.application_mode,
                     consent=bool(obj.automated_submission_consent_at))


class AutomationRunViewSet(mixins.ListModelMixin, mixins.RetrieveModelMixin, viewsets.GenericViewSet):
    serializer_class = AutomationRunSerializer
    filterset_fields = ["kind", "status"]
    ordering_fields = ["started_at"]

    def get_queryset(self):
        if getattr(self, "swagger_fake_view", False):  # schema generation
            return AutomationRun.objects.none()
        return AutomationRun.objects.filter(user=self.request.user)


class ScheduledSearchViewSet(OwnedQuerysetMixin, viewsets.ModelViewSet):
    queryset = ScheduledSearch.objects.prefetch_related("sources")
    serializer_class = ScheduledSearchSerializer

    @extend_schema(request=None, responses={202: dict})
    @action(detail=True, methods=["post"], url_path="run-now")
    def run_now(self, request, pk=None):
        search = self.get_object()
        ScheduledSearch.objects.filter(pk=search.pk).update(next_run_at=timezone.now())
        result = tasks.dispatch_due_searches.delay()
        return Response({"queued": True, "task_id": result.id}, status=202)
