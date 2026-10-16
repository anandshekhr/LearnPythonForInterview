"""Analytics computed from real database rows only (no placeholder numbers)."""
from __future__ import annotations

from collections import Counter
from datetime import timedelta

from django.db.models import Count, Q
from django.db.models.functions import TruncWeek
from django.utils import timezone

from apps.applications.models import Application, ApplicationStatus, ApplicationStatusHistory, Interview
from apps.jobs.models import JobPosting
from apps.matching.models import JobMatch
from apps.profiles.models import ResumeVersion

S = ApplicationStatus
REACHED_INTERVIEW = [S.ASSESSMENT, S.INTERVIEW_SCHEDULED, S.INTERVIEW_COMPLETED, S.OFFER_RECEIVED]


def _rate(numerator, denominator):
    return round(100 * numerator / denominator, 1) if denominator else None


def overview(user) -> dict:
    now = timezone.now()
    week_ago = now - timedelta(days=7)
    matches = JobMatch.objects.filter(user=user, job__duplicate_of__isnull=True)
    active_matches = matches.filter(job__status=JobPosting.Status.ACTIVE)
    apps = Application.objects.filter(user=user)
    submitted = apps.filter(submitted_at__isnull=False)
    submitted_count = submitted.count()
    ever_interview = ApplicationStatusHistory.objects.filter(application__user=user, to_status__in=REACHED_INTERVIEW) \
        .values("application").distinct().count()
    offers = ApplicationStatusHistory.objects.filter(application__user=user, to_status=S.OFFER_RECEIVED) \
        .values("application").distinct().count()
    responded = ApplicationStatusHistory.objects.filter(application__user=user,
                                                        to_status__in=REACHED_INTERVIEW + [S.REJECTED]) \
        .values("application").distinct().count()
    return {
        "jobs_discovered": matches.count(),
        "active_jobs": active_matches.count(),
        "new_jobs_this_week": matches.filter(job__first_seen_at__gte=week_ago).count(),
        "high_match_jobs": active_matches.filter(category__in=["excellent", "good"], excluded_reason="").count(),
        "shortlisted": active_matches.filter(user_state=JobMatch.UserState.SHORTLISTED).count(),
        "applications_total": apps.count(),
        "applications_submitted": submitted_count,
        "submitted_this_week": submitted.filter(submitted_at__gte=week_ago).count(),
        "awaiting_review": ResumeVersion.objects.filter(user=user, kind="tailored", status="awaiting_approval").count(),
        "ready_to_apply": apps.filter(status=S.READY_TO_APPLY).count(),
        "submission_failures": apps.filter(status=S.SUBMISSION_FAILED).count(),
        "interviews_scheduled": Interview.objects.filter(application__user=user, scheduled_at__gte=now,
                                                         outcome=Interview.Outcome.PENDING).count(),
        "offers_received": offers,
        "rejections": apps.filter(status=S.REJECTED).count(),
        "conversion": {
            "response_rate": _rate(responded, submitted_count),
            "interview_rate": _rate(ever_interview, submitted_count),
            "offer_rate": _rate(offers, submitted_count),
            "interview_to_offer": _rate(offers, ever_interview),
        },
    }


def timeseries(user, weeks=12) -> dict:
    start = timezone.now() - timedelta(weeks=weeks)

    def weekly(qs, field):
        rows = qs.filter(**{f"{field}__gte": start}).annotate(week=TruncWeek(field)).values("week").annotate(n=Count("id"))
        return {r["week"].date().isoformat(): r["n"] for r in rows}

    discovered = weekly(JobPosting.objects.filter(matches__user=user), "first_seen_at")
    submitted = weekly(Application.objects.filter(user=user), "submitted_at")
    interviews = weekly(Interview.objects.filter(application__user=user), "scheduled_at")
    labels = sorted(set(discovered) | set(submitted) | set(interviews))
    return {"weeks": labels, "jobs_discovered": [discovered.get(w, 0) for w in labels],
            "applications_submitted": [submitted.get(w, 0) for w in labels], "interviews": [interviews.get(w, 0) for w in labels]}


def breakdowns(user) -> dict:
    apps = Application.objects.filter(user=user)
    matches = JobMatch.objects.filter(user=user, job__status=JobPosting.Status.ACTIVE, job__duplicate_of__isnull=True)
    status_counts = dict(apps.values_list("status").annotate(n=Count("id")))
    categories = dict(matches.values_list("category").annotate(n=Count("id")))
    buckets = Counter(min(m // 10 * 10, 90) for m in matches.values_list("score", flat=True))
    missing = Counter()
    for skills in matches.filter(category__in=["excellent", "good", "possible"]).values_list("skills_missing", flat=True):
        missing.update(skills)
    by_source = list(apps.values("source_type").annotate(total=Count("id"), submitted=Count("id", filter=Q(submitted_at__isnull=False))))
    return {
        "status_funnel": [{"status": s, "label": S(s).label, "count": status_counts.get(s, 0)} for s in S.values if s != S.SUBMITTING],
        "match_categories": {c: categories.get(c, 0) for c in ["excellent", "good", "possible", "low"]},
        "score_distribution": [{"bucket": f"{b}-{b + 9}", "count": buckets.get(b, 0)} for b in range(0, 100, 10)],
        "top_missing_skills": [{"skill": k, "jobs": v} for k, v in missing.most_common(10)],
        "applications_by_source": by_source,
    }
