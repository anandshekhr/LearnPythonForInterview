import django_filters
from django.db.models import Q

from .models import JobPosting


class JobPostingFilter(django_filters.FilterSet):
    q = django_filters.CharFilter(method="search", label="Keyword search in title, company and description")
    published_after = django_filters.IsoDateTimeFilter(field_name="published_at", lookup_expr="gte")
    seen_after = django_filters.IsoDateTimeFilter(field_name="first_seen_at", lookup_expr="gte")
    location = django_filters.CharFilter(method="by_location")
    min_score = django_filters.NumberFilter(method="by_min_score")
    category = django_filters.CharFilter(method="by_category")
    skill = django_filters.CharFilter(field_name="skills__skill__name", lookup_expr="iexact")
    include_duplicates = django_filters.BooleanFilter(method="dupes")

    class Meta:
        model = JobPosting
        fields = ["status", "company", "source", "work_model", "employment_type", "seniority"]

    def search(self, qs, name, value):
        terms = [t for t in value.split() if t][:6]
        for term in terms:
            qs = qs.filter(Q(title__icontains=term) | Q(company__name__icontains=term) | Q(description_text__icontains=term))
        return qs

    def by_location(self, qs, name, value):
        return qs.filter(Q(location_raw__icontains=value) | Q(locations__icontains=[value]))

    def by_min_score(self, qs, name, value):
        return qs.filter(matches__user=self.request.user, matches__score__gte=value)

    def by_category(self, qs, name, value):
        return qs.filter(matches__user=self.request.user, matches__category=value)

    def dupes(self, qs, name, value):
        return qs if value else qs.filter(duplicate_of__isnull=True)
