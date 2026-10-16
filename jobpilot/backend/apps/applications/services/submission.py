"""Submission engine for the three application modes.

* discovery  – nothing is submitted; the official link is shown.
* assisted   – documents and answers are prepared; the user submits on the official page
               (optionally with the local browser-assist helper) and confirms here.
* automated  – only through an adapter's documented API with an employer-issued key,
               explicit user consent, daily limits and duplicate checks.

A submission is claimed only when the source confirms it. Network failures after the
request was sent are recorded as *unconfirmed*: retrying then requires explicit
confirmation from the user so an application is never sent twice by accident.
"""
from __future__ import annotations

import logging
from datetime import timedelta

from django.db import transaction
from django.utils import timezone

from apps.automation.models import ApplicationMode
from apps.automation.services import get_settings
from apps.core import audit
from apps.core.exceptions import DomainError
from apps.jobs.adapters.base import SubmissionNotSupported
from apps.jobs.adapters.registry import adapter_for
from apps.jobs.http import SourceHTTPError
from apps.notifications.models import Notification
from apps.notifications.service import notify
from apps.profiles.models import ResumeVersion

from ..models import Application, ApplicationAnswer, ApplicationStatus, SubmissionChannel
from .answers import profile_values, readiness_problems
from .lifecycle import record_history, transition

logger = logging.getLogger(__name__)
S = ApplicationStatus
STUCK_AFTER = timedelta(minutes=15)


def api_submission_available(app: Application) -> tuple[bool, str]:
    job = app.job
    if job is None:
        return False, "The job posting no longer exists."
    source = job.source
    caps = adapter_for(source).capabilities
    if not caps.api_submission:
        return False, f"{source.get_source_type_display()} has no submission API."
    if not (source.submission_enabled and source.has_submission_key):
        return False, "No employer-issued API key is configured for this board; apply on the official page."
    return True, ""


def check_ready(app: Application, mode: str) -> None:
    if app.is_submitted:
        raise DomainError("This application has already been submitted.", code="already_submitted", status_code=409)
    if app.status not in (S.READY_TO_APPLY, S.SUBMISSION_FAILED):
        raise DomainError("Approve a resume for this application first.", code="not_ready", status_code=409)
    if app.resume_version is None or app.resume_version.status != ResumeVersion.Status.APPROVED:
        raise DomainError("An approved resume version is required.", code="not_approved", status_code=409)
    if app.job is None or app.job.status != app.job.Status.ACTIVE:
        raise DomainError("The job posting is no longer active.", code="job_expired", status_code=409)
    problems = readiness_problems(app)
    if problems:
        raise DomainError("Some answers need your attention.", code="answers_incomplete", details={"problems": problems})


def start_automated_submission(app: Application, user, idempotency_key: str, confirm_retry_after_unconfirmed=False):
    """Validate, lock (status SUBMITTING) and return the app. The network call happens in a task."""
    if not idempotency_key or len(idempotency_key) > 80:
        raise DomainError("An Idempotency-Key header (max 80 chars) is required.", code="idempotency_key_required")
    if app.idempotency_key == idempotency_key and app.status in (S.SUBMITTING, *Application.SUBMITTED_STATES):
        return app, False  # replay of the same request: no second submission
    settings_ = get_settings(user)
    if settings_.application_mode != ApplicationMode.AUTOMATED or not settings_.automated_submission_consent_at:
        raise DomainError("Automated submission is off. Enable it and give consent in Automation settings.",
                          code="automation_not_enabled", status_code=403)
    ok, reason = api_submission_available(app)
    if not ok:
        raise DomainError(reason, code="submission_not_supported", status_code=409)
    check_ready(app, "automated")
    if app.submission_response.get("unconfirmed") and not confirm_retry_after_unconfirmed:
        raise DomainError("The previous attempt may have reached the employer. Check your email/ATS, then retry with "
                          "confirm_retry=true if it did not.", code="unconfirmed_previous_attempt", status_code=409)
    day_start = timezone.localtime().replace(hour=0, minute=0, second=0, microsecond=0)
    sent_today = Application.objects.filter(user=user, submission_channel=SubmissionChannel.API, submitted_at__gte=day_start).count()
    if sent_today >= settings_.max_submissions_per_day:
        raise DomainError(f"Daily automated submission limit reached ({settings_.max_submissions_per_day}).",
                          code="daily_limit", status_code=429)
    with transaction.atomic():
        updated = Application.objects.filter(pk=app.pk, status__in=[S.READY_TO_APPLY, S.SUBMISSION_FAILED]).update(
            status=S.SUBMITTING, submission_started_at=timezone.now(), idempotency_key=idempotency_key,
            submission_attempts=app.submission_attempts + 1, last_error="", mode=ApplicationMode.AUTOMATED,
        )
        if not updated:
            raise DomainError("A submission for this application is already in progress.", code="in_progress", status_code=409)
        record_history(app, app.status, S.SUBMITTING, user, "Automated submission started", "system")
    audit.record("application.submission_started", user=user, obj=app)
    app.refresh_from_db()
    return app, True


def _payload(app: Application) -> dict:
    values = profile_values(app.user.profile)
    doc = app.resume_document
    with doc.file.open("rb") as fh:
        resume_bytes = fh.read()
    answers = {a.question_key: a.answer for a in app.answers.all()
               if a.answer and a.field_type != "file" and a.question_key not in {"first_name", "last_name", "email", "phone"}
               and (not a.sensitive or a.reviewed_at)}
    urls = [{"label": label, "url": values[key]} for label, key in (("LinkedIn", "linkedin"), ("GitHub", "github"),
                                                                     ("Portfolio", "website")) if values.get(key)]
    return {
        "first_name": values["first_name"], "last_name": values["last_name"], "email": values["email"], "phone": values["phone"],
        "resume_bytes": resume_bytes, "resume_filename": f"{values['first_name']}_{values['last_name']}_Resume.{doc.format}".replace(" ", "_"),
        "resume_content_type": doc.content_type,
        "cover_letter_text": app.cover_letter.text_content if app.cover_letter else "",
        "answers": answers, "urls": urls,
    }


def perform_submission(app_id: int) -> Application:
    app = Application.objects.select_related("job", "job__source", "resume_document", "cover_letter", "user").get(pk=app_id)
    if app.status != S.SUBMITTING:
        return app
    adapter = adapter_for(app.job.source, with_credentials=True)
    try:
        result = adapter.submit_application(app.job.external_id, _payload(app))
    except SubmissionNotSupported as exc:
        return _fail(app, str(exc), unconfirmed=False)
    except SourceHTTPError as exc:
        definite = exc.status_code is not None and 400 <= exc.status_code < 500 and exc.status_code not in (408, 429)
        return _fail(app, f"Submission failed: {exc}", unconfirmed=not definite, status_code=exc.status_code)
    except Exception as exc:  # network drop after the request may have been received
        logger.warning("submission_error", extra={"application_id": app.pk, "error": type(exc).__name__})
        return _fail(app, f"Submission outcome unknown ({type(exc).__name__}).", unconfirmed=True)
    finally:
        adapter.http.close()
    response = {"status_code": result.status_code, "message": result.message, "excerpt": result.response_excerpt}
    if not result.confirmed:
        return _fail(app, f"The source did not confirm the submission ({result.message or result.status_code}).",
                     unconfirmed=bool(result.status_code and result.status_code < 300), response=response)
    app = transition(app, S.SUBMITTED, note="Submitted through the authorized API", system=True,
                     submitted_at=timezone.now(), submission_channel=SubmissionChannel.API,
                     submission_reference=result.reference[:200], submission_response=response, last_error="")
    notify(app.user, Notification.Kind.SUBMITTED, f"Application submitted: {app.job_title} at {app.company_name}",
           body=f"Reference: {result.reference}" if result.reference else "The employer's system confirmed receipt.",
           link=f"/applications/{app.pk}", obj=app, dedupe_key=f"submitted:{app.pk}")
    audit.record("application.submitted", user=app.user, obj=app, channel="api")
    return app


def _fail(app, message, unconfirmed, status_code=None, response=None):
    response = {**(response or {}), "unconfirmed": unconfirmed, "status_code": status_code}
    app = transition(app, S.SUBMISSION_FAILED, note=message, system=True, last_error=message[:1000], submission_response=response)
    notify(app.user, Notification.Kind.SUBMISSION_FAILED, f"Submission failed: {app.job_title} at {app.company_name}",
           body=message + (" It may still have reached the employer – check before retrying." if unconfirmed else ""),
           link=f"/applications/{app.pk}", obj=app, dedupe_key=f"submission_failed:{app.pk}:{app.submission_attempts}")
    audit.record("application.submission_failed", user=app.user, obj=app, unconfirmed=unconfirmed)
    return app


def confirm_manual_submission(app: Application, user, channel=SubmissionChannel.MANUAL, reference="", note=""):
    """The user applied on the official page and confirms it here."""
    if app.is_submitted:
        raise DomainError("This application is already marked as submitted.", code="already_submitted", status_code=409)
    if app.status not in (S.READY_TO_APPLY, S.SUBMISSION_FAILED):
        raise DomainError("Approve a resume for this application first (status must be ready to apply).", code="not_ready",
                          status_code=409)
    return transition(app, S.SUBMITTED, user=user, note=note or "Submitted by you on the official application page",
                      system=False, submitted_at=timezone.now(), submission_channel=channel, submission_reference=reference[:200],
                      last_error="")


def recover_stuck_submissions() -> int:
    cutoff = timezone.now() - STUCK_AFTER
    count = 0
    for app in Application.objects.filter(status=S.SUBMITTING, submission_started_at__lt=cutoff):
        _fail(app, "Submission was interrupted before a response was received.", unconfirmed=True)
        count += 1
    return count


def mark_sensitive_reviewed(answer: ApplicationAnswer):
    answer.reviewed_at = timezone.now()
    answer.save(update_fields=["reviewed_at", "updated_at"])
