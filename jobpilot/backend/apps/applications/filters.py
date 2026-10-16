import django_filters

from .models import Application, Interview


class ApplicationFilter(django_filters.FilterSet):
    status__in = django_filters.BaseInFilter(field_name="status")
    submitted_after = django_filters.IsoDateTimeFilter(field_name="submitted_at", lookup_expr="gte")
    submitted_before = django_filters.IsoDateTimeFilter(field_name="submitted_at", lookup_expr="lte")
    company = django_filters.CharFilter(field_name="company_name", lookup_expr="icontains")

    class Meta:
        model = Application
        fields = ["status", "job", "source_type", "submission_channel", "mode"]


class InterviewFilter(django_filters.FilterSet):
    start = django_filters.IsoDateTimeFilter(field_name="scheduled_at", lookup_expr="gte")
    end = django_filters.IsoDateTimeFilter(field_name="scheduled_at", lookup_expr="lte")

    class Meta:
        model = Interview
        fields = ["application", "outcome", "mode"]
