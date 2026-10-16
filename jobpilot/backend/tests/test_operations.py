"""Automation settings, scheduling, task retries/recovery, notifications, analytics, health, retention."""
from datetime import timedelta

import pytest
from celery.exceptions import Retry
from django.utils import timezone

from apps.applications.models import Application
from apps.automation.models import AutomationRun, ScheduledSearch
from apps.automation.tasks import dispatch_due_searches, purge_expired_data, recover_interrupted_runs
from apps.core.testing import auth_client
from apps.core.testing_profiles import verified_profile
from apps.jobs.models import Company, JobPosting, JobSource
from apps.jobs.tests.helpers import mock_greenhouse
from apps.notifications.models import Notification

pytestmark = pytest.mark.django_db


@pytest.fixture
def setup(respx_mock):
    user = verified_profile()
    api = auth_client(user)
    company = Company.objects.create(name="Example SaaS")
    source = JobSource.objects.create(company=company, source_type="greenhouse", identifier="examplesaas", created_by=user)
    route = mock_greenhouse(respx_mock)
    return user, api, source, route


def test_health(client):
    r = client.get("/health/")
    assert r.status_code == 200 and r.json()["checks"] == {"database": "ok", "cache": "ok"}
    assert r["X-Request-ID"]


def test_openapi_schema_served(setup):
    _, api, *_ = setup
    r = api.get("/api/schema/")
    assert r.status_code == 200 and b"/api/v1/applications/{id}/submit/" in r.content


def test_automation_mode_requires_explicit_consent(setup):
    _, api, *_ = setup
    r = api.patch("/api/v1/automation/settings/", {"application_mode": "automated"}, format="json")
    assert r.status_code == 400
    r = api.patch("/api/v1/automation/settings/", {"application_mode": "automated", "automated_submission_consent": True},
                  format="json")
    assert r.status_code == 200 and r.data["automated_submission_allowed"] is True
    r = api.patch("/api/v1/automation/settings/", {"automated_submission_consent": False}, format="json")
    assert r.status_code == 400  # still automated without consent
    r = api.patch("/api/v1/automation/settings/", {"application_mode": "assisted", "automated_submission_consent": False},
                  format="json")
    assert r.status_code == 200 and r.data["automated_submission_allowed"] is False
    assert api.patch("/api/v1/automation/settings/", {"max_submissions_per_day": 500}, format="json").status_code == 400


def test_scheduled_search_dispatches_sync_once_per_source(setup):
    user, api, source, route = setup
    r = api.post("/api/v1/scheduled-searches/", {"name": "Daily", "interval_hours": 12, "sources": [source.pk]}, format="json")
    assert r.status_code == 201
    api.post("/api/v1/scheduled-searches/", {"name": "All", "interval_hours": 6}, format="json")
    result = dispatch_due_searches()
    assert result == {"searches": 2, "sources_queued": 1}
    assert route.call_count == 1 and JobPosting.objects.count() == 3
    search = ScheduledSearch.objects.get(name="Daily")
    assert search.next_run_at > timezone.now() + timedelta(hours=11)
    assert dispatch_due_searches()["searches"] == 0  # not due again yet
    assert AutomationRun.objects.filter(kind="source_sync", user=user, status="succeeded").exists()


def test_sync_task_retries_on_transient_errors(setup, respx_mock):
    _, _, source, route = setup
    route.respond(503)
    from apps.jobs.tasks import sync_source_task

    with pytest.raises(Retry):
        sync_source_task.apply(args=[source.pk], throw=True)
    source.refresh_from_db()
    assert source.consecutive_failures == 1 and source.status == "degraded"
    assert AutomationRun.objects.filter(kind="source_sync", status="failed").count() == 1


def test_source_failure_notifies_owner_after_repeated_errors(setup):
    user, _, source, route = setup
    route.respond(404)
    from apps.jobs.tasks import sync_source_task

    for _ in range(3):
        sync_source_task.delay(source.pk)
    source.refresh_from_db()
    assert source.status == "error"
    assert Notification.objects.filter(user=user, kind="source_error").count() == 1


def test_interrupted_runs_recovered():
    AutomationRun.objects.create(kind="source_sync", heartbeat_at=timezone.now() - timedelta(hours=2))
    assert recover_interrupted_runs()["interrupted_runs"] == 1
    assert AutomationRun.objects.get().status == "interrupted"


def test_notifications_api(setup):
    user, api, source, _ = setup
    from apps.jobs.tasks import sync_source_task

    sync_source_task.delay(source.pk)
    count = api.get("/api/v1/notifications/unread-count/").data["count"]
    assert count >= 1
    first = api.get("/api/v1/notifications/", {"unread": "1"}).data["results"][0]
    assert first["kind"] == "high_match" and first["link"].startswith("/jobs/")
    api.post(f"/api/v1/notifications/{first['id']}/read/")
    assert api.get("/api/v1/notifications/unread-count/").data["count"] == count - 1
    api.post("/api/v1/notifications/read-all/")
    assert api.get("/api/v1/notifications/unread-count/").data["count"] == 0


def test_notification_preferences_respected(setup):
    user, api, source, _ = setup
    api.patch("/api/v1/automation/settings/", {"notify_new_matches": False}, format="json")
    from apps.jobs.tasks import sync_source_task

    sync_source_task.delay(source.pk)
    assert not Notification.objects.filter(user=user, kind="high_match").exists()


def test_expired_shortlisted_job_notifies(setup):
    user, api, source, route = setup
    from apps.jobs.tasks import sync_source_task

    sync_source_task.delay(source.pk)
    job = JobPosting.objects.get(external_id="4012002")
    api.post("/api/v1/applications/", {"job": job.pk}, format="json")
    payload = route.calls[0].response.json()
    payload["jobs"] = [j for j in payload["jobs"] if j["id"] != 4012002]
    route.respond(200, json=payload)
    JobSource.objects.filter(pk=source.pk).update(last_synced_at=None)
    sync_source_task.delay(source.pk)
    assert Notification.objects.filter(user=user, kind="job_expired", object_id=str(job.pk)).exists()


def test_analytics_reflect_real_data(setup):
    user, api, source, _ = setup
    empty = api.get("/api/v1/analytics/overview/").data
    assert empty["jobs_discovered"] == 0 and empty["conversion"]["interview_rate"] is None
    from apps.jobs.tasks import sync_source_task

    sync_source_task.delay(source.pk)
    job = JobPosting.objects.get(external_id="4012001")
    app_id = api.post("/api/v1/applications/", {"job": job.pk}, format="json").data["id"]
    Application.objects.filter(pk=app_id).update(status="ready_to_apply")
    api.post(f"/api/v1/applications/{app_id}/confirm-submitted/", {}, format="json")
    api.post(f"/api/v1/applications/{app_id}/status/", {"status": "interview_scheduled"}, format="json")
    data = api.get("/api/v1/analytics/overview/").data
    assert data["jobs_discovered"] == 3 and data["new_jobs_this_week"] == 3 and data["high_match_jobs"] >= 1
    assert data["applications_submitted"] == 1 and data["conversion"]["interview_rate"] == 100.0
    series = api.get("/api/v1/analytics/timeseries/", {"weeks": 4}).data
    assert sum(series["applications_submitted"]) == 1 and sum(series["jobs_discovered"]) == 3
    breakdown = api.get("/api/v1/analytics/breakdowns/").data
    assert sum(b["count"] for b in breakdown["score_distribution"]) == 3
    assert {"status": "interview_scheduled", "label": "Interview scheduled", "count": 1} in breakdown["status_funnel"]


def test_csv_report_neutralises_formulas(setup):
    user, api, source, _ = setup
    company = Company.objects.create(name="=HYPERLINK(\"http://evil\")")
    Application.objects.create(user=user, company_name=company.name, job_title="Dev", job_url="https://x.example/1")
    r = api.get("/api/v1/analytics/applications.csv")
    body = r.content.decode()
    assert r["Content-Type"] == "text/csv" and "'=HYPERLINK" in body


def test_retention_purges_old_closed_applications(setup):
    user, *_ = setup
    user.profile.data_retention_days = 30
    user.profile.save()
    old = Application.objects.create(user=user, company_name="A", job_title="B", job_url="https://x.example/2", status="closed")
    Application.objects.filter(pk=old.pk).update(updated_at=timezone.now() - timedelta(days=60))
    keep = Application.objects.create(user=user, company_name="A", job_title="C", job_url="https://x.example/3", status="submitted")
    Application.objects.filter(pk=keep.pk).update(updated_at=timezone.now() - timedelta(days=60))
    assert purge_expired_data()["applications"] == 1
    assert list(Application.objects.values_list("pk", flat=True)) == [keep.pk]


def test_audit_log_has_no_secrets(setup):
    from apps.core.audit import record
    from apps.core.models import AuditLog

    user, *_ = setup
    entry = record("test.event", user=user, password="hunter2", nested={"token": "abc", "ok": 1})
    assert entry.metadata == {"password": "[redacted]", "nested": {"token": "[redacted]", "ok": 1}}
    assert AuditLog.objects.filter(action="test.event").exists()


def test_admin_changelists_load(setup, client):
    from django.contrib import admin as dj_admin
    from django.contrib.auth import get_user_model

    staff = get_user_model().objects.create_superuser(username="root@example.com", email="root@example.com", password="x-Long-pass-1")
    client.force_login(staff)
    for model in dj_admin.site._registry:
        url = f"/admin/{model._meta.app_label}/{model._meta.model_name}/"
        assert client.get(url).status_code == 200, url
