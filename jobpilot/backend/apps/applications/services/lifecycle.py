"""Application lifecycle: creation with duplicate prevention, validated transitions, history."""
from __future__ import annotations

from django.db import IntegrityError, transaction
from django.db.models import Q
from django.utils import timezone

from apps.core import audit
from apps.core.exceptions import DomainError
from apps.documents.models import GeneratedDocument
from apps.jobs.models import JobPosting
from apps.matching.models import JobMatch
from apps.notifications.models import Notification
from apps.notifications.service import notify
from apps.profiles.models import ResumeVersion

from ..models import MANUAL_ALLOWED, TRANSITIONS, Application, ApplicationStatus, ApplicationStatusHistory

S = ApplicationStatus


def find_duplicate(user, job: JobPosting) -> Application | None:
    """An application to the same posting, or to the same role mirrored on another source."""
    related_ids = {job.pk}
    if job.duplicate_of_id:
        related_ids.add(job.duplicate_of_id)
        related_ids |= set(JobPosting.objects.filter(duplicate_of_id=job.duplicate_of_id).values_list("pk", flat=True))
    related_ids |= set(JobPosting.objects.filter(duplicate_of_id=job.pk).values_list("pk", flat=True))
    return Application.objects.filter(user=user).filter(
        Q(job_id__in=related_ids) | Q(job_fingerprint=job.fingerprint, job__isnull=True) | Q(job_url=job.official_url)
    ).first()


def record_history(app: Application, from_status: str, to_status: str, user=None, note="", actor="user"):
    ApplicationStatusHistory.objects.create(application=app, from_status=from_status, to_status=to_status,
                                            changed_by=user, actor=actor, note=note[:1000])


@transaction.atomic
def create_application(user, job: JobPosting, status=S.DISCOVERED, actor_user=None) -> Application:
    existing = find_duplicate(user, job)
    if existing:
        raise DomainError("You already have an application for this job.", code="duplicate_application", status_code=409,
                          details={"application_id": existing.pk, "status": existing.status})
    match = JobMatch.objects.filter(user=user, job=job).first()
    from apps.automation.services import get_settings

    try:
        with transaction.atomic():
            app = Application.objects.create(
                user=user, job=job, match=match, company_name=job.company.name, job_title=job.title, job_url=job.official_url,
                source_type=job.source.source_type, job_fingerprint=job.fingerprint, match_score=match.score if match else None,
                status=status, mode=get_settings(user).application_mode,
            )
    except IntegrityError:
        raise DomainError("You already have an application for this job.", code="duplicate_application", status_code=409)
    record_history(app, "", status, actor_user or user, "Application created")
    if match and match.user_state in (JobMatch.UserState.NEW, JobMatch.UserState.VIEWED) and status == S.SHORTLISTED:
        JobMatch.objects.filter(pk=match.pk).update(user_state=JobMatch.UserState.SHORTLISTED)
    audit.record("application.created", user=user, obj=app, job_id=job.pk)
    return app


def transition(app: Application, to_status: str, user=None, note="", system=False, **fields) -> Application:
    """Validated state change with history. `system=False` restricts to user-settable states."""
    with transaction.atomic():
        locked = Application.objects.select_for_update().get(pk=app.pk)
        current = locked.status
        if to_status == current:
            return locked
        if to_status not in TRANSITIONS.get(current, set()):
            raise DomainError(f"Cannot move an application from '{current}' to '{to_status}'.", code="invalid_transition",
                              status_code=409, details={"allowed": sorted(TRANSITIONS.get(current, set()))})
        if not system and to_status not in MANUAL_ALLOWED:
            raise DomainError(f"'{to_status}' is set by the application workflow, not manually.", code="invalid_transition",
                              status_code=409)
        locked.status = to_status
        now = timezone.now()
        if to_status in Application.SUBMITTED_STATES or to_status in Application.TERMINAL:
            locked.last_verified_at = now
        for key, value in fields.items():
            setattr(locked, key, value)
        locked.save()
        record_history(locked, current, to_status, user, note, "system" if system else "user")
    audit.record("application.status_changed", user=user or app.user, obj=locked, from_status=current, to_status=to_status)
    return locked


def on_documents_generated(user, job, version: ResumeVersion):
    app = Application.objects.filter(user=user, job=job).first()
    if app is None:
        app = create_application(user, job, status=S.SHORTLISTED)
    if app.status in (S.DISCOVERED, S.SHORTLISTED):
        app = transition(app, S.RESUME_PREPARED, note=f"Tailored resume v{version.version_number} generated", system=True)
    if app.status == S.RESUME_PREPARED:
        app = transition(app, S.AWAITING_APPROVAL, note="Waiting for your review", system=True)
        notify(user, Notification.Kind.AWAITING_REVIEW, f"Review your tailored resume for {job.title}",
               body="Check the side-by-side comparison and approve it before applying.", link=f"/resume-versions/{version.pk}",
               obj=version, dedupe_key=f"awaiting_review:{version.pk}")
    return app


def on_version_approved(user, version: ResumeVersion):
    if not version.job_id:
        return None
    app = Application.objects.filter(user=user, job_id=version.job_id).first()
    if app is None or app.status not in (S.AWAITING_APPROVAL, S.RESUME_PREPARED, S.SHORTLISTED, S.DISCOVERED, S.READY_TO_APPLY):
        return app
    return attach_resume(app, version, user)


def attach_resume(app: Application, version: ResumeVersion, user, cover_letter: GeneratedDocument | None = None):
    if version.user_id != app.user_id:
        raise DomainError("Resume version not found.", code="not_found", status_code=404)
    if version.status != ResumeVersion.Status.APPROVED:
        raise DomainError("Only approved resume versions can be used for an application.", code="not_approved", status_code=409)
    if app.is_submitted:
        raise DomainError("The application has already been submitted.", code="already_submitted", status_code=409)
    resume_doc = (version.documents.filter(kind=GeneratedDocument.Kind.RESUME, status=GeneratedDocument.Status.APPROVED)
                  .order_by("format").first())  # "docx" < "pdf": prefer PDF below
    pdf = version.documents.filter(kind=GeneratedDocument.Kind.RESUME, format="pdf", status=GeneratedDocument.Status.APPROVED).first()
    resume_doc = pdf or resume_doc
    if version.kind == ResumeVersion.Kind.TAILORED and resume_doc is None:
        raise DomainError("The approved version has no rendered resume document.", code="missing_document", status_code=409)
    if cover_letter is not None and (cover_letter.user_id != app.user_id or cover_letter.status != GeneratedDocument.Status.APPROVED):
        raise DomainError("Cover letter must be approved before use.", code="not_approved", status_code=409)
    if cover_letter is None and version.job_id:
        cover_letter = version.documents.filter(kind=GeneratedDocument.Kind.COVER_LETTER,
                                                status=GeneratedDocument.Status.APPROVED).first()
    fields = dict(resume_version=version, resume_document=resume_doc, cover_letter=cover_letter, approved_at=timezone.now())
    path = {
        S.DISCOVERED: [S.RESUME_PREPARED, S.AWAITING_APPROVAL, S.READY_TO_APPLY],
        S.SHORTLISTED: [S.AWAITING_APPROVAL, S.READY_TO_APPLY],
        S.RESUME_PREPARED: [S.AWAITING_APPROVAL, S.READY_TO_APPLY],
        S.AWAITING_APPROVAL: [S.READY_TO_APPLY],
        S.SUBMISSION_FAILED: [S.READY_TO_APPLY],
        S.READY_TO_APPLY: [],
    }.get(app.status)
    if path is None:
        raise DomainError(f"Cannot attach a resume while the application is '{app.status}'.", code="invalid_transition",
                          status_code=409)
    if not path:
        for key, value in fields.items():
            setattr(app, key, value)
        app.save()
        return app
    for step in path:
        app = transition(app, step, user=user, note="Resume approved" if step == S.READY_TO_APPLY else "", system=True,
                         **(fields if step == S.READY_TO_APPLY else {}))
    return app
