from django.utils import timezone
from drf_spectacular.utils import extend_schema
from rest_framework import mixins, viewsets
from rest_framework.decorators import action
from rest_framework.response import Response

from .models import Notification
from .serializers import NotificationSerializer


class NotificationViewSet(mixins.ListModelMixin, mixins.RetrieveModelMixin, mixins.DestroyModelMixin, viewsets.GenericViewSet):
    serializer_class = NotificationSerializer
    filterset_fields = ["kind"]

    def get_queryset(self):
        if getattr(self, "swagger_fake_view", False):  # schema generation
            return Notification.objects.none()
        qs = Notification.objects.filter(user=self.request.user)
        unread = self.request.query_params.get("unread")
        return qs.filter(read_at__isnull=True) if unread in ("1", "true") else qs

    @extend_schema(responses={200: dict})
    @action(detail=False, methods=["get"], url_path="unread-count")
    def unread_count(self, request):
        return Response({"count": Notification.objects.filter(user=request.user, read_at__isnull=True).count()})

    @extend_schema(request=None, responses={200: NotificationSerializer})
    @action(detail=True, methods=["post"], url_path="read")
    def mark_read(self, request, pk=None):
        n = self.get_object()
        if not n.read_at:
            n.read_at = timezone.now()
            n.save(update_fields=["read_at"])
        return Response(NotificationSerializer(n).data)

    @extend_schema(request=None, responses={200: dict})
    @action(detail=False, methods=["post"], url_path="read-all")
    def mark_all_read(self, request):
        count = Notification.objects.filter(user=request.user, read_at__isnull=True).update(read_at=timezone.now())
        return Response({"updated": count})
