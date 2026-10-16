import pytest

from apps.core.testing import auth_client, make_user
from apps.core.testing_profiles import verified_profile
from apps.jobs.models import JobSource

from .helpers import mock_greenhouse

pytestmark = pytest.mark.django_db


@pytest.fixture
def synced(respx_mock):
    user = verified_profile()
    api = auth_client(user)
    mock_greenhouse(respx_mock)
    company = api.post("/api/v1/companies/", {"name": "Example SaaS", "company_type": "SaaS"}, format="json").data
    r = api.post("/api/v1/sources/", {"company": company["id"], "source_type": "greenhouse", "identifier": "examplesaas",
                                      "min_request_interval_seconds": 1}, format="json")
    assert r.status_code == 201, r.data
    return user, api, company, r.data


def test_create_source_triggers_sync_and_lists_jobs(synced):
    user, api, company, source = synced
    r = api.get(f"/api/v1/sources/{source['id']}/")
    assert r.data["status"] == "active" and r.data["active_job_count"] == 3
    caps = r.data["capabilities"]
    assert caps["api_submission"] is True and caps["api_submission_active"] is False
    r = api.get("/api/v1/jobs/", {"ordering": "-matches__score"})
    assert r.data["count"] == 3
    assert r.data["results"][0]["my_match"]["score"] >= r.data["results"][-1]["my_match"]["score"]
    r = api.get("/api/v1/jobs/", {"q": "python", "work_model": "hybrid"})
    assert r.data["count"] == 1
    job_id = r.data["results"][0]["id"]
    r = api.get(f"/api/v1/jobs/{job_id}/")
    assert r.data["requirements"]["min_years"] == 6 and r.data["official_url"].startswith("https://boards.greenhouse.io/")
    assert api.get("/api/v1/jobs/", {"min_score": 70}).data["count"] >= 1
    assert api.get("/api/v1/jobs/", {"skill": "django"}).data["count"] == 1


def test_matches_endpoint_and_state(synced):
    user, api, *_ = synced
    r = api.get("/api/v1/matches/", {"category": "excellent"})
    assert r.data["count"] >= 1
    match = r.data["results"][0]
    assert match["explanation"] and match["recommended_action"] and "components" in match
    r = api.patch(f"/api/v1/matches/{match['id']}/", {"user_state": "shortlisted"}, format="json")
    assert r.status_code == 200 and r.data["user_state"] == "shortlisted"
    r = api.patch(f"/api/v1/matches/{match['id']}/", {"score": 100}, format="json")
    assert r.data["score"] == match["score"]  # read-only
    r = api.get("/api/v1/matches/semantic/")
    assert r.status_code == 200 and r.data[0]["similarity"] > 0
    assert api.post("/api/v1/matches/rematch/").status_code == 202


def test_scoring_config_validation(synced):
    _, api, *_ = synced
    assert api.patch("/api/v1/scoring-config/", {"good_threshold": 90}, format="json").status_code == 400
    r = api.patch("/api/v1/scoring-config/", {"w_salary": 20}, format="json")
    assert r.status_code == 200 and r.data["w_salary"] == 20


def test_other_users_cannot_edit_registry_or_see_matches(synced):
    user, api, company, source = synced
    other = auth_client(make_user("other@example.com"))
    assert other.get(f"/api/v1/companies/{company['id']}/").status_code == 200  # shared registry is readable
    assert other.patch(f"/api/v1/companies/{company['id']}/", {"name": "x"}, format="json").status_code == 403
    assert other.delete(f"/api/v1/sources/{source['id']}/").status_code == 403
    assert other.get("/api/v1/matches/").data["count"] == 0
    match_id = api.get("/api/v1/matches/").data["results"][0]["id"]
    assert other.get(f"/api/v1/matches/{match_id}/").status_code == 404


def test_submission_credentials_staff_only_and_encrypted(synced):
    user, api, company, source = synced
    payload = {"api_key": "employer-key-123", "submission_enabled": True, "confirm_authorized": True}
    assert api.post(f"/api/v1/sources/{source['id']}/submission-credentials/", payload, format="json").status_code == 403
    staff = make_user("admin@example.com", is_staff=True)
    admin = auth_client(staff)
    r = admin.post(f"/api/v1/sources/{source['id']}/submission-credentials/", {**payload, "confirm_authorized": False}, format="json")
    assert r.status_code == 400
    r = admin.post(f"/api/v1/sources/{source['id']}/submission-credentials/", payload, format="json")
    assert r.status_code == 200 and r.data["capabilities"]["api_submission_active"] is True
    assert "api_key" not in r.data and "submission_api_key_encrypted" not in r.data
    obj = JobSource.objects.get(pk=source["id"])
    assert "employer-key-123" not in obj.submission_api_key_encrypted
    assert obj.get_submission_api_key() == "employer-key-123"


def test_source_validation(synced):
    _, api, company, _ = synced
    bad = api.post("/api/v1/sources/", {"company": company["id"], "source_type": "greenhouse", "identifier": "../x"}, format="json")
    assert bad.status_code == 400
    dup = api.post("/api/v1/sources/", {"company": company["id"], "source_type": "greenhouse", "identifier": "examplesaas"}, format="json")
    assert dup.status_code == 400
    fast = api.post("/api/v1/sources/", {"company": company["id"], "source_type": "lever", "identifier": "z",
                                         "sync_interval_minutes": 5}, format="json")
    assert fast.status_code == 400
    types = api.get("/api/v1/sources/types/").data
    assert {t["source_type"] for t in types} >= {"greenhouse", "lever"}
