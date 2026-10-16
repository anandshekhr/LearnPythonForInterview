from django.db import connection
from django.core.cache import cache
from drf_spectacular.utils import extend_schema
from rest_framework.permissions import AllowAny
from rest_framework.response import Response
from rest_framework.views import APIView


class HealthView(APIView):
    """Liveness/readiness probe used by Docker and load balancers."""

    permission_classes = [AllowAny]
    authentication_classes = []
    throttle_classes = []

    @extend_schema(responses={200: dict, 503: dict})
    def get(self, request):
        checks = {}
        try:
            with connection.cursor() as cursor:
                cursor.execute("SELECT 1")
            checks["database"] = "ok"
        except Exception:
            checks["database"] = "error"
        try:
            cache.set("healthcheck", "1", 5)
            checks["cache"] = "ok" if cache.get("healthcheck") == "1" else "error"
        except Exception:
            checks["cache"] = "error"
        healthy = all(v == "ok" for v in checks.values())
        return Response({"status": "ok" if healthy else "degraded", "checks": checks}, status=200 if healthy else 503)
