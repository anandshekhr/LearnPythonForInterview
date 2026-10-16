import httpx
import pytest

from apps.automation.models import AutomationRun
from apps.core.testing_profiles import verified_profile
from apps.jobs.http import SourceHTTPError
from apps.jobs.models import Company, JobPosting, JobPostingChange, JobSkill, JobSource
from apps.jobs.services.sync import sync_source
from apps.jobs.tasks import sync_source_task
from apps.matching.models import JobMatch
from apps.notifications.models import Notification

from .helpers import load, mock_greenhouse, mock_lever

pytestmark = pytest.mark.django_db


@pytest.fixture
def gh_source():
    company = Company.objects.create(name="Example SaaS", company_type="SaaS")
    return JobSource.objects.create(company=company, source_type="greenhouse", identifier="examplesaas",
                                    min_request_interval_seconds=0)


def test_full_sync_pipeline_creates_jobs_requirements_and_matches(respx_mock, gh_source):
    user = verified_profile()
    mock_greenhouse(respx_mock)
    result = sync_source_task.delay(gh_source.pk).get()
    assert result["status"] == "ok" and result["created"] == 3
    gh_source.refresh_from_db()
    assert gh_source.status == "active" and gh_source.active_job_count == 3 and gh_source.last_success_at
    run = AutomationRun.objects.get(kind="source_sync")
    assert run.status == "succeeded" and run.stats["created"] == 3

    python_job = JobPosting.objects.get(external_id="4012001")
    assert python_job.requirements["min_years"] == 6
    assert {"Python", "Django", "PostgreSQL", "Redis", "Docker"} <= set(python_job.requirements["required_skills"])
    assert {"Angular", "Kubernetes", "AWS"} <= set(python_job.requirements["preferred_skills"])
    assert JobSkill.objects.filter(job=python_job, importance="required", skill__name="Django").exists()
    assert python_job.embedding is not None

    matches = {m.job.external_id: m for m in JobMatch.objects.filter(user=user).select_related("job")}
    assert set(matches) == {"4012001", "4012002", "4012003"}
    best, sales = matches["4012001"], matches["4012003"]
    assert best.category in {"excellent", "good"} and best.score >= 70, best.explanation
    assert sales.category == "low" and sales.score < best.score
    assert "Python" in best.skills_matched and "Django" in best.skills_matched
    assert best.components["salary"]["score"] == 1.0  # ₹32–42 LPA meets ₹30 LPA minimum
    assert "preferred area" in best.location_fit
    assert any(r["label"].startswith("Senior Software Engineer") for r in best.relevant_experience)
    # Prompt injection inside the description has no effect on the deterministic score.
    assert "IGNORE ALL PREVIOUS INSTRUCTIONS" in python_job.description_text and best.score < 100
    # High matches notify once.
    assert Notification.objects.filter(user=user, kind="high_match", object_id=str(python_job.pk)).count() == 1


def test_resync_detects_unchanged_updated_and_removed(respx_mock, gh_source):
    route = mock_greenhouse(respx_mock)
    sync_source(gh_source)
    payload = load("greenhouse_jobs.json")
    payload["jobs"][0]["content"] += "&lt;p&gt;Now also requires Kafka.&lt;/p&gt;"
    removed = payload["jobs"].pop(1)
    route.respond(200, json=payload)
    gh_source.refresh_from_db()
    result = sync_source(gh_source)
    assert len(result.updated) == 1 and result.unchanged == 1 and len(result.expired) == 1
    changed = JobPosting.objects.get(external_id="4012001")
    assert changed.description_version == 2
    assert JobPostingChange.objects.filter(job=changed, change_type="updated", fields_changed__contains=["description_text"]).exists()
    gone = JobPosting.objects.get(external_id=str(removed["id"]))
    assert gone.status == "expired" and gone.expired_at
    # It comes back -> reactivated, not duplicated.
    payload["jobs"].append(removed)
    route.respond(200, json=payload)
    result = sync_source(gh_source)
    assert result.reactivated == [gone.pk]
    assert JobPosting.objects.filter(source=gh_source).count() == 3


def test_mass_removal_guard(respx_mock, gh_source):
    payload = load("greenhouse_jobs.json")
    base = payload["jobs"][0]
    payload["jobs"] = [{**base, "id": 5000 + i, "title": f"Engineer {i}"} for i in range(6)]
    route = mock_greenhouse(respx_mock, payload=payload)
    sync_source(gh_source)
    route.respond(200, json={"jobs": []})
    gh_source.refresh_from_db()
    result = sync_source(gh_source)
    assert result.expired == [] and result.warnings
    assert JobPosting.objects.filter(source=gh_source, status="active").count() == 6


def test_cross_source_duplicate_detection(respx_mock, gh_source):
    mock_greenhouse(respx_mock)
    sync_source(gh_source)
    original = JobPosting.objects.get(external_id="4012001")
    other = JobSource.objects.create(company=gh_source.company, source_type="greenhouse", identifier="examplesaas-mirror",
                                     min_request_interval_seconds=0)
    payload = load("greenhouse_jobs.json")
    payload["jobs"] = [payload["jobs"][0] | {"id": 999}]
    mock_greenhouse(respx_mock, token="examplesaas-mirror", payload=payload)
    sync_source(other)
    dup = JobPosting.objects.get(source=other, external_id="999")
    assert dup.duplicate_of_id == original.pk


def test_sync_failure_marks_source_and_run(respx_mock, gh_source):
    respx_mock.get("https://boards-api.greenhouse.io/v1/boards/examplesaas/jobs").respond(404)
    result = sync_source_task.delay(gh_source.pk).get()
    assert result["status"] == "failed"
    gh_source.refresh_from_db()
    assert gh_source.status == "degraded" and gh_source.consecutive_failures == 1 and "404" in gh_source.last_error
    assert AutomationRun.objects.get(kind="source_sync").status == "failed"


def test_network_failure_in_sync_service_raises(respx_mock, gh_source):
    respx_mock.get("https://boards-api.greenhouse.io/v1/boards/examplesaas/jobs").mock(side_effect=httpx.ConnectTimeout("t"))
    from apps.jobs.adapters.registry import adapter_for  # noqa: F401
    from apps.jobs.http import SourceHttpClient

    http = SourceHttpClient("x", min_interval=0, max_retries=1, sleep=lambda s: None)
    with pytest.raises(SourceHTTPError):
        sync_source(gh_source, http=http)


def test_lever_source_sync(respx_mock):
    company = Company.objects.create(name="Example Lever")
    source = JobSource.objects.create(company=company, source_type="lever", identifier="examplelever", min_request_interval_seconds=0)
    mock_lever(respx_mock)
    result = sync_source(source)
    assert len(result.created) == 2
    job = JobPosting.objects.get(external_id__startswith="8f0c1a2b-1111")
    assert job.work_model == "hybrid" and job.salary_min == 2_800_000 and job.seniority == "mid"
