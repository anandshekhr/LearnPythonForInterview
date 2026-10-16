"""Discovery -> matching -> tailoring -> approval -> application -> submission, with mocked sources."""
import json

import httpx
import pytest
from django.utils import timezone

from apps.applications.models import Application
from apps.applications.services.submission import recover_stuck_submissions
from apps.automation.models import AutomationSettings
from apps.core.testing import auth_client, make_user
from apps.core.testing_profiles import verified_profile
from apps.documents.models import GeneratedDocument
from apps.jobs.models import JobPosting, JobSource
from apps.jobs.tests.helpers import load, mock_greenhouse, mock_lever
from apps.notifications.models import Notification
from apps.profiles.models import ResumeVersion

pytestmark = pytest.mark.django_db
GH_JOB = "https://boards-api.greenhouse.io/v1/boards/examplesaas/jobs/4012001"


@pytest.fixture(autouse=True)
def _media(tmp_path, settings):
    settings.PRIVATE_MEDIA_ROOT = tmp_path


@pytest.fixture
def world(respx_mock):
    user = verified_profile()
    api = auth_client(user)
    mock_greenhouse(respx_mock)
    company = api.post("/api/v1/companies/", {"name": "Example SaaS"}, format="json").data
    source = api.post("/api/v1/sources/", {"company": company["id"], "source_type": "greenhouse", "identifier": "examplesaas",
                                           "min_request_interval_seconds": 1}, format="json").data
    job = JobPosting.objects.get(external_id="4012001")
    return user, api, source, job


def generate_and_approve(api, job):
    r = api.post("/api/v1/documents/generate/", {"job": job.pk, "include_cover_letter": True}, format="json")
    assert r.status_code == 202 and r.data["result"]["status"] == "ok", r.data
    version_id = r.data["result"]["resume_version_id"]
    return version_id


def test_tailoring_is_factual_and_requires_approval(world):
    user, api, _, job = world
    version_id = generate_and_approve(api, job)
    version = ResumeVersion.objects.get(pk=version_id)
    master = version.base_version
    assert version.status == "awaiting_approval" and version.kind == "tailored" and version.job == job
    # Facts preserved: same employers/titles/dates; skills subset of verified; required skills first.
    assert [(e["company"], e["title"], e["start_date"]) for e in version.content["employment"]] == \
        [(e["company"], e["title"], e["start_date"]) for e in master.content["employment"]]
    assert set(version.content["skills"]) == set(master.content["skills"])
    assert version.content["skills"][0] == "Python"
    # React is preferred by the job but absent from the profile -> listed as missing, never claimed.
    rows = {r["requirement"]: r for r in version.tailoring_notes["requirement_map"]}
    assert rows["React"]["status"] == "missing" and "React" not in version.content["skills"]
    assert rows["Django"]["status"] == "matched" and rows["Django"]["evidence"]
    docs = GeneratedDocument.objects.filter(resume_version=version)
    assert {(d.kind, d.format) for d in docs} == {("resume", "pdf"), ("resume", "docx"), ("cover_letter", "pdf")}
    assert all(d.status == "awaiting_approval" for d in docs)
    letter = docs.get(kind="cover_letter")
    assert "Example SaaS" in letter.text_content and "Senior Python Developer" in letter.text_content
    pdf = docs.get(kind="resume", format="pdf")
    body = b"".join(api.get(f"/api/v1/documents/{pdf.pk}/download/").streaming_content)
    assert body.startswith(b"%PDF")

    r = api.get(f"/api/v1/resume-versions/{version_id}/compare/")
    assert r.status_code == 200 and any(s["section"] == "skills" and s["changed"] for s in r.data["sections"])

    # Editing in a fabricated skill or employer is rejected; reordering is fine.
    content = version.content
    bad = json.loads(json.dumps(content))
    bad["skills"].append("React")
    r = api.patch(f"/api/v1/resume-versions/{version_id}/", {"content": bad}, format="json")
    assert r.status_code == 400 and "React" in r.data["error"]["message"]
    bad = json.loads(json.dumps(content))
    bad["employment"][0]["company"] = "Google"
    assert api.patch(f"/api/v1/resume-versions/{version_id}/", {"content": bad}, format="json").status_code == 400
    good = json.loads(json.dumps(content))
    good["skills"] = list(reversed(good["skills"]))
    good["employment"][0]["achievements"] = list(reversed(good["employment"][0]["achievements"]))
    assert api.patch(f"/api/v1/resume-versions/{version_id}/", {"content": good}, format="json").status_code == 200

    # The application waits for approval; then becomes ready to apply.
    app = Application.objects.get(user=user, job=job)
    assert app.status == "awaiting_approval"
    assert Notification.objects.filter(user=user, kind="awaiting_review").exists()
    r = api.post(f"/api/v1/resume-versions/{version_id}/approve/")
    assert r.status_code == 200 and r.data["status"] == "approved"
    app.refresh_from_db()
    assert app.status == "ready_to_apply" and app.resume_version_id == version_id
    assert app.resume_document.format == "pdf" and app.resume_document.status == "approved"
    assert api.post(f"/api/v1/resume-versions/{version_id}/approve/").status_code == 409


def test_llm_rephrase_rejects_fabrications(world, monkeypatch, settings):
    from apps.core import llm

    user, api, _, job = world
    settings.LLM_PROVIDER = "anthropic"

    def fake(task, *a, **k):
        trusted = k["trusted_context"]
        if task != "resume_tailoring":
            raise llm.LLMUnavailable("x")
        b0, b1 = trusted["bullets"][0], trusted["bullets"][1]
        return llm.LLMResult({"summary": "Seasoned engineer with 15+ years at Google.", "bullets": [
            {**b0, "text": b0["text"].rstrip(".") + ", improving revenue by 45%."},  # new metric -> rejected
            {**b1, "text": "Delivered " + b1["text"][0].lower() + b1["text"][1:]},  # pure rephrase -> accepted
        ]}, "anthropic", "test")

    monkeypatch.setattr(llm, "generate_json", fake)
    version = ResumeVersion.objects.get(pk=generate_and_approve(api, job))
    notes = version.tailoring_notes
    assert any("new figures" in w for w in notes["warnings"])
    assert any("Kept original summary" in w for w in notes["warnings"])
    assert len(notes["rewrites"]) == 1 and notes["rewrites"][0]["rewritten"].startswith("Delivered ")
    assert "45%" not in json.dumps(version.content)


def test_assisted_flow_manual_confirmation_and_tracking(world):
    user, api, _, job = world
    r = api.post("/api/v1/applications/", {"job": job.pk}, format="json")
    assert r.status_code == 201 and r.data["status"] == "shortlisted"
    app_id = r.data["id"]
    dup = api.post("/api/v1/applications/", {"job": job.pk}, format="json")
    assert dup.status_code == 409 and dup.data["error"]["details"]["application_id"] == app_id
    # Not ready yet.
    assert api.post(f"/api/v1/applications/{app_id}/confirm-submitted/", {}, format="json").status_code == 409
    version_id = generate_and_approve(api, job)
    api.post(f"/api/v1/resume-versions/{version_id}/approve/")
    kit = api.get(f"/api/v1/applications/{app_id}/kit/").data
    assert kit["official_url"].startswith("https://boards.greenhouse.io/") and kit["status"] == "ready_to_apply"
    assert kit["automated_submission"]["available"] is False and "employer-issued" in kit["automated_submission"]["reason"]
    r = api.post(f"/api/v1/applications/{app_id}/confirm-submitted/", {"reference": "GH-123"}, format="json")
    assert r.status_code == 200 and r.data["status"] == "submitted" and r.data["submission_channel"] == "manual"
    # Lifecycle continues with validation and history.
    assert api.post(f"/api/v1/applications/{app_id}/status/", {"status": "offer_received"}, format="json").status_code == 200
    assert api.post(f"/api/v1/applications/{app_id}/status/", {"status": "shortlisted"}, format="json").status_code == 409
    detail = api.get(f"/api/v1/applications/{app_id}/").data
    assert [h["to_status"] for h in detail["history"]] == [
        "shortlisted", "resume_prepared", "awaiting_approval", "ready_to_apply", "submitted", "offer_received"]
    assert api.delete(f"/api/v1/applications/{app_id}/").status_code == 409
    other = auth_client(make_user("eve@example.com"))
    assert other.get(f"/api/v1/applications/{app_id}/").status_code == 404
    assert other.post(f"/api/v1/applications/{app_id}/status/", {"status": "withdrawn"}, format="json").status_code == 404


def test_interviews_and_reminders(world):
    user, api, _, job = world
    app = api.post("/api/v1/applications/", {"job": job.pk}, format="json").data
    version_id = generate_and_approve(api, job)
    api.post(f"/api/v1/resume-versions/{version_id}/approve/")
    api.post(f"/api/v1/applications/{app['id']}/confirm-submitted/", {}, format="json")
    when = (timezone.now() + timezone.timedelta(hours=3)).isoformat()
    r = api.post("/api/v1/interviews/", {"application": app["id"], "round_name": "Technical round", "scheduled_at": when}, format="json")
    assert r.status_code == 201
    assert Application.objects.get(pk=app["id"]).status == "interview_scheduled"
    api.post("/api/v1/follow-ups/", {"application": app["id"], "due_date": str(timezone.localdate()), "note": "Ping recruiter"},
             format="json")
    from apps.notifications.tasks import send_due_reminders

    assert send_due_reminders() == 2
    assert send_due_reminders() == 0
    other = auth_client(make_user("mallory@example.com"))
    r = other.post("/api/v1/interviews/", {"application": app["id"], "round_name": "x", "scheduled_at": when}, format="json")
    assert r.status_code == 400


@pytest.fixture
def automated(world, respx_mock):
    user, api, source, job = world
    src = JobSource.objects.get(pk=source["id"])
    src.set_submission_api_key("employer-key")
    src.submission_enabled = True
    src.save()
    respx_mock.get(GH_JOB).respond(200, json=load("greenhouse_job_questions.json"))
    app = api.post("/api/v1/applications/", {"job": job.pk}, format="json").data
    version_id = generate_and_approve(api, job)
    api.post(f"/api/v1/resume-versions/{version_id}/approve/")
    api.post(f"/api/v1/applications/{app['id']}/prepare-answers/")
    return user, api, app["id"], respx_mock


def enable_automation(user, consent=True):
    AutomationSettings.objects.update_or_create(user=user, defaults={
        "application_mode": "automated", "automated_submission_consent_at": timezone.now() if consent else None})


def answer_sensitive(api, app_id, reviewed=True):
    answers = api.get(f"/api/v1/applications/{app_id}/").data["answers"]
    visa = next(a for a in answers if a["question_key"] == "question_1002")
    assert visa["sensitive"] and visa["answer"] == ""  # never pre-filled
    return api.patch(f"/api/v1/applications/{app_id}/answers/", [{"id": visa["id"], "answer": "0", "reviewed": reviewed}],
                     format="json")


def test_automated_submission_requires_consent_and_review(automated):
    user, api, app_id, respx = automated
    answers = {a["question_key"]: a for a in api.get(f"/api/v1/applications/{app_id}/").data["answers"]}
    assert answers["first_name"]["answer"] == "Asha" and answers["email"]["answer"] == "asha.verma@example.com"
    assert answers["gender"]["sensitive"] and answers["gender"]["answer"] == ""
    headers = {"HTTP_IDEMPOTENCY_KEY": "k1"}
    r = api.post(f"/api/v1/applications/{app_id}/submit/", {}, format="json", **headers)
    assert r.status_code == 403 and r.data["error"]["code"] == "automation_not_enabled"
    enable_automation(user)
    r = api.post(f"/api/v1/applications/{app_id}/submit/", {}, format="json", **headers)
    assert r.status_code == 400 and r.data["error"]["code"] == "answers_incomplete"
    answer_sensitive(api, app_id, reviewed=False)
    r = api.post(f"/api/v1/applications/{app_id}/submit/", {}, format="json", **headers)
    assert any("explicit review" in p for p in r.data["error"]["details"]["problems"])
    assert api.post(f"/api/v1/applications/{app_id}/submit/", {}, format="json").status_code == 400  # no idempotency key


def test_automated_submission_success_and_idempotency(automated):
    user, api, app_id, respx = automated
    enable_automation(user)
    answer_sensitive(api, app_id)
    route = respx.post(GH_JOB).respond(200, json={"success": "Candidate saved successfully"})
    headers = {"HTTP_IDEMPOTENCY_KEY": "submit-abc"}
    r = api.post(f"/api/v1/applications/{app_id}/submit/", {}, format="json", **headers)
    assert r.status_code == 202, r.data
    app = Application.objects.get(pk=app_id)
    assert app.status == "submitted" and app.submission_channel == "api" and app.submitted_at
    sent = json.loads(route.calls[0].request.content)
    assert route.calls[0].request.headers["Authorization"].startswith("Basic ")
    assert sent["first_name"] == "Asha" and sent["question_1002"] == "0" and sent["resume_content_filename"].endswith(".pdf")
    assert "gender" not in sent  # unanswered sensitive question is never sent
    # Replay with the same key: no second POST.
    r = api.post(f"/api/v1/applications/{app_id}/submit/", {}, format="json", **headers)
    assert r.status_code == 200 and route.call_count == 1
    r = api.post(f"/api/v1/applications/{app_id}/submit/", {}, format="json", HTTP_IDEMPOTENCY_KEY="other")
    assert r.status_code == 409 and route.call_count == 1
    assert Notification.objects.filter(user=user, kind="submitted").count() == 1


def test_ambiguous_failure_requires_confirmed_retry(automated):
    user, api, app_id, respx = automated
    enable_automation(user)
    answer_sensitive(api, app_id)
    route = respx.post(GH_JOB).mock(side_effect=httpx.ReadTimeout("timeout"))
    r = api.post(f"/api/v1/applications/{app_id}/submit/", {}, format="json", HTTP_IDEMPOTENCY_KEY="a1")
    app = Application.objects.get(pk=app_id)
    assert app.status == "submission_failed" and app.submission_response["unconfirmed"] is True
    assert Notification.objects.filter(user=user, kind="submission_failed").exists()
    r = api.post(f"/api/v1/applications/{app_id}/submit/", {}, format="json", HTTP_IDEMPOTENCY_KEY="a2")
    assert r.status_code == 409 and r.data["error"]["code"] == "unconfirmed_previous_attempt"
    route.mock(side_effect=None, return_value=httpx.Response(200, json={"success": "Candidate saved successfully"}))
    r = api.post(f"/api/v1/applications/{app_id}/submit/", {"confirm_retry": True}, format="json", HTTP_IDEMPOTENCY_KEY="a3")
    assert r.status_code == 202 and Application.objects.get(pk=app_id).status == "submitted"


def test_definite_rejection_and_daily_limit(automated):
    user, api, app_id, respx = automated
    enable_automation(user)
    AutomationSettings.objects.filter(user=user).update(max_submissions_per_day=0)
    answer_sensitive(api, app_id)
    r = api.post(f"/api/v1/applications/{app_id}/submit/", {}, format="json", HTTP_IDEMPOTENCY_KEY="z")
    assert r.status_code == 429
    AutomationSettings.objects.filter(user=user).update(max_submissions_per_day=5)
    respx.post(GH_JOB).respond(422, json={"error": "Invalid email"})
    api.post(f"/api/v1/applications/{app_id}/submit/", {}, format="json", HTTP_IDEMPOTENCY_KEY="z")
    app = Application.objects.get(pk=app_id)
    assert app.status == "submission_failed" and app.submission_response["unconfirmed"] is False and "422" in app.last_error


def test_interrupted_submission_recovered(automated):
    user, api, app_id, respx = automated
    Application.objects.filter(pk=app_id).update(status="submitting",
                                                 submission_started_at=timezone.now() - timezone.timedelta(hours=1))
    assert recover_stuck_submissions() == 1
    app = Application.objects.get(pk=app_id)
    assert app.status == "submission_failed" and app.submission_response["unconfirmed"] is True


def test_lever_api_submission(respx_mock, settings, tmp_path):
    from apps.jobs.models import Company

    user = verified_profile("lever@example.com")
    api = auth_client(user)
    company = Company.objects.create(name="Example Lever")
    source = JobSource.objects.create(company=company, source_type="lever", identifier="examplelever", min_request_interval_seconds=0,
                                      submission_enabled=True)
    source.set_submission_api_key("lever-key")
    source.save()
    mock_lever(respx_mock)
    from apps.jobs.tasks import sync_source_task

    sync_source_task.delay(source.pk)
    job = JobPosting.objects.get(external_id__startswith="8f0c1a2b-1111")
    app = api.post("/api/v1/applications/", {"job": job.pk}, format="json").data
    version_id = generate_and_approve(api, job)
    api.post(f"/api/v1/resume-versions/{version_id}/approve/")
    api.post(f"/api/v1/applications/{app['id']}/prepare-answers/")
    enable_automation(user)
    route = respx_mock.post(f"https://api.lever.co/v0/postings/examplelever/{job.external_id}").respond(
        200, json={"ok": True, "applicationId": "lev-789"})
    r = api.post(f"/api/v1/applications/{app['id']}/submit/", {}, format="json", HTTP_IDEMPOTENCY_KEY="lv1")
    assert r.status_code == 202
    req = route.calls[0].request
    assert req.url.params["key"] == "lever-key" and b'name="resume"' in req.content
    obj = Application.objects.get(pk=app["id"])
    assert obj.status == "submitted" and obj.submission_reference == "lev-789"
