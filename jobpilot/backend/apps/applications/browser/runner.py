import os

from django.conf import settings

from ..models import Application
from .filler import AssistNotPermitted, build_plan, materialize_resume, run_assist


def run_assist_for_application(application_id: int, headless: bool | None = None, keep_open=False, wait_for_user=None,
                               url: str | None = None, allowed_hosts=None, screenshot_dir=None) -> dict:
    """Pre-fill the official form. Never submits. Returns a report dict."""
    if not settings.BROWSER_AUTOMATION_ENABLED:
        raise AssistNotPermitted("Browser automation is disabled (BROWSER_AUTOMATION_ENABLED=0).")
    app = Application.objects.select_related("job", "job__source", "user__profile", "resume_document", "cover_letter").get(
        pk=application_id)
    if app.status not in ("ready_to_apply", "submission_failed"):
        raise AssistNotPermitted("Approve a resume for this application first.")
    resume_path = materialize_resume(app)
    try:
        plan = build_plan(app, resume_path, url=url)
        report = run_assist(plan, headless=settings.BROWSER_HEADLESS if headless is None else headless, keep_open=keep_open,
                            allowed_hosts=allowed_hosts, screenshot_dir=screenshot_dir, wait_for_user=wait_for_user)
    finally:
        os.remove(resume_path)
    return report.as_dict()
