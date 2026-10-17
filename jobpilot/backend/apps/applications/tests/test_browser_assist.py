"""Browser automation against controlled local fixtures (never real job sites)."""
import functools
import glob
import http.server
import threading
from pathlib import Path

import pytest
from django.utils import timezone

from apps.applications.browser.filler import AssistNotPermitted, check_host_allowed
from apps.applications.browser.runner import run_assist_for_application
from apps.applications.models import Application, ApplicationAnswer
from apps.applications.services.answers import prepare_answers
from apps.applications.services.lifecycle import attach_resume, create_application
from apps.core.testing_profiles import verified_profile
from apps.documents.services.generation import approve_version, generate_for_job
from apps.jobs.adapters.base import ApplicationQuestion
from apps.jobs.models import Company, JobPosting, JobSource

pytestmark = [pytest.mark.django_db]
FIXTURES = Path(__file__).parent / "fixtures"
CHROME = (glob.glob("/opt/pw-browsers/chromium-*/chrome-linux/chrome") or [""])[0]


@pytest.fixture(scope="module")
def server():
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(FIXTURES))
    handler.log_message = lambda *a, **k: None
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


@pytest.fixture(autouse=True)
def _browser(settings, tmp_path):
    try:
        import playwright  # noqa: F401
    except ImportError:
        pytest.skip("playwright not installed")
    settings.BROWSER_AUTOMATION_ENABLED = True
    settings.BROWSER_EXECUTABLE_PATH = CHROME
    settings.PRIVATE_MEDIA_ROOT = tmp_path


def ready_application(source_type="greenhouse"):
    user = verified_profile(f"{source_type}@example.com")
    company = Company.objects.create(name=f"Fixture {source_type}")
    source = JobSource.objects.create(company=company, source_type=source_type, identifier=f"fx-{source_type}")
    job = JobPosting.objects.create(
        source=source, company=company, external_id="1", canonical_url="https://example.invalid/job", title="Senior Python Developer",
        normalized_title="senior python developer", last_seen_at=timezone.now(), content_hash="x", fingerprint="f",
        description_text="Python Django PostgreSQL", requirements={"required_skills": ["Python", "Django"]},
    )
    app = create_application(user, job)
    version, docs = generate_for_job(user, job, include_cover_letter=True)
    for d in docs:
        if d.kind == "cover_letter":
            d.status = "approved"
            d.save()
    approve_version(version, user)
    app.refresh_from_db()
    if app.status != "ready_to_apply":
        app = attach_resume(app, version, user)
    questions = [
        ApplicationQuestion("first_name", "First Name", "text", True), ApplicationQuestion("last_name", "Last Name", "text", True),
        ApplicationQuestion("email", "Email", "text", True), ApplicationQuestion("resume", "Resume", "file", True),
        ApplicationQuestion("question_1001", "LinkedIn Profile", "text"),
        ApplicationQuestion("question_1002", "Will you require visa sponsorship?", "select", True, sensitive=True),
        ApplicationQuestion("gender", "Gender", "select", False, sensitive=True),
    ]
    prepare_answers(app, questions)
    return Application.objects.get(pk=app.pk)


def capture(store):
    def wait(page, report):
        store["values"] = page.evaluate("""() => Object.fromEntries([...document.querySelectorAll('input,textarea,select')]
            .map(e => [e.name || e.id, e.type === 'file' ? (e.files.length ? e.files[0].name : '') : e.value]))""")
        store["submitted"] = page.evaluate("window.submitted")
    return wait


def test_greenhouse_form_prefilled_never_submitted(server):
    app = ready_application("greenhouse")
    ApplicationAnswer.objects.filter(application=app, question_key="question_1001").update(
        answer="https://linkedin.com/in/asha", origin="user", reviewed_at=timezone.now())
    store = {}
    report = run_assist_for_application(app.pk, headless=True, keep_open=True, wait_for_user=capture(store),
                                        url=f"{server}/greenhouse_form.html", allowed_hosts=["127.0.0.1"])
    v = store["values"]
    assert v["job_application[first_name]"] == "Asha" and v["job_application[email]"] == "asha.verma@example.com"
    assert v["resume"].endswith(".pdf")
    assert v["cover_letter_text"].startswith("Dear Hiring Team at Fixture greenhouse")
    assert v["question_1001"] == "https://linkedin.com/in/asha"
    assert v["question_1002"] == "" and v["gender"] == ""  # sensitive, unreviewed -> untouched
    assert store["submitted"] is False  # never submits
    assert any("sponsorship" in s for s in report["skipped"])
    assert not report["human_verification_required"]


def test_reviewed_sensitive_answer_is_filled(server):
    app = ready_application("greenhouse")
    ApplicationAnswer.objects.filter(application=app, question_key="question_1002").update(answer="0", reviewed_at=timezone.now())
    store = {}
    run_assist_for_application(app.pk, headless=True, keep_open=True, wait_for_user=capture(store),
                               url=f"{server}/greenhouse_form.html", allowed_hosts=["127.0.0.1"])
    assert store["values"]["question_1002"] == "0" and store["submitted"] is False


def test_lever_form(server):
    app = ready_application("lever")
    store = {}
    report = run_assist_for_application(app.pk, headless=True, keep_open=True, wait_for_user=capture(store),
                                        url=f"{server}/lever_form.html", allowed_hosts=["127.0.0.1"])
    v = store["values"]
    assert v["name"] == "Asha Verma" and v["org"] == "Nimbus Analytics Pvt Ltd"
    assert v["urls[LinkedIn]"].startswith("https://linkedin.com/in/") and v["resume"].endswith(".pdf")
    assert store["submitted"] is False and "full_name" in report["filled"]


def test_captcha_detected_and_reported(server):
    app = ready_application("greenhouse")
    report = run_assist_for_application(app.pk, headless=True, url=f"{server}/captcha_form.html", allowed_hosts=["127.0.0.1"])
    assert report["human_verification_required"] is True and "CAPTCHA" in report["blocked_reason"]


def test_login_wall_stops_automation(server):
    app = ready_application("greenhouse")
    report = run_assist_for_application(app.pk, headless=True, url=f"{server}/login.html", allowed_hosts=["127.0.0.1"])
    assert report["login_required"] and report["filled"] == []


def test_disallowed_hosts_rejected(settings):
    with pytest.raises(AssistNotPermitted):
        check_host_allowed("https://www.linkedin.com/jobs/view/1")
    with pytest.raises(AssistNotPermitted):
        check_host_allowed("http://jobs.lever.co/acme/1")  # https only
    check_host_allowed("https://jobs.lever.co/acme/1")


def test_disabled_by_default(settings):
    app = ready_application("greenhouse")
    settings.BROWSER_AUTOMATION_ENABLED = False
    with pytest.raises(AssistNotPermitted):
        run_assist_for_application(app.pk)


def test_career_page_js_rendering(server, respx_mock, settings, monkeypatch):
    """render_js discovers links created by JavaScript; non-public requests from the page are aborted."""
    import json as _json

    from apps.jobs import crawl
    from apps.jobs.services.sync import sync_source

    blocked = []
    original = crawl.check_public_url

    def guard(url):
        if "169.254" in url:
            blocked.append(url)
            raise crawl.UnsafeURL("non-public")
        return original(url)

    monkeypatch.setattr(crawl, "check_public_url", guard)
    monkeypatch.setattr("apps.jobs.adapters.career_page.check_public_url", guard)
    respx_mock.get(f"{server}/robots.txt").respond(404)
    posting = {"@context": "https://schema.org", "@type": "JobPosting", "title": "Platform Engineer",
               "description": "Python, Kubernetes", "datePosted": "2026-10-01"}
    respx_mock.get(f"{server}/jobs/42-platform-engineer").respond(
        200, html=f'<script type="application/ld+json">{_json.dumps(posting)}</script>')
    source = JobSource.objects.create(company=Company.objects.create(name="SPA Co"), source_type="career_page",
                                      identifier=f"{server}/careers_spa.html", config={"render_js": True})
    result = sync_source(source)
    assert [j.title for j in JobPosting.objects.filter(pk__in=result.created)] == ["Platform Engineer"]
    assert blocked, "request to the metadata address should have been intercepted"
