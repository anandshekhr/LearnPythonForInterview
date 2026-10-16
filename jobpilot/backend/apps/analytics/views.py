import csv

from django.http import HttpResponse
from drf_spectacular.utils import OpenApiParameter, extend_schema
from rest_framework.response import Response
from rest_framework.views import APIView

from apps.applications.models import Application
from apps.core import audit

from . import services


class OverviewView(APIView):
    @extend_schema(responses={200: dict})
    def get(self, request):
        return Response(services.overview(request.user))


class TimeseriesView(APIView):
    @extend_schema(parameters=[OpenApiParameter("weeks", int)], responses={200: dict})
    def get(self, request):
        weeks = max(1, min(int(request.query_params.get("weeks", 12)), 52))
        return Response(services.timeseries(request.user, weeks))


class BreakdownView(APIView):
    @extend_schema(responses={200: dict})
    def get(self, request):
        return Response(services.breakdowns(request.user))


class ApplicationReportView(APIView):
    """CSV export of your applications and their current status."""

    @extend_schema(responses={(200, "text/csv"): bytes})
    def get(self, request):
        response = HttpResponse(content_type="text/csv")
        response["Content-Disposition"] = 'attachment; filename="applications.csv"'
        writer = csv.writer(response)
        writer.writerow(["company", "job_title", "status", "match_score", "submitted_at", "channel", "reference", "job_url",
                         "follow_up_date", "updated_at"])
        for a in Application.objects.filter(user=request.user).order_by("-updated_at"):
            row = [a.company_name, a.job_title, a.get_status_display(), a.match_score or "",
                   a.submitted_at.isoformat() if a.submitted_at else "", a.submission_channel, a.submission_reference, a.job_url,
                   a.follow_up_date or "", a.updated_at.isoformat()]
            # Neutralise spreadsheet formula injection from untrusted job titles/company names.
            writer.writerow(["'" + str(c) if str(c)[:1] in ("=", "+", "-", "@") else c for c in row])
        audit.record("report.exported", request=request)
        return response
