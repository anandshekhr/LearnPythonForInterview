"""Prepare answers to an application's questions from the verified profile.

Only straightforward profile facts are pre-filled. Sensitive questions (legal
declarations, demographics, disability, sponsorship/work authorisation, consent) are
never pre-filled and must be answered and reviewed by the user before any submission.
"""
from __future__ import annotations

import logging

from django.utils import timezone

from apps.jobs.adapters.registry import adapter_for
from apps.jobs.http import SourceHTTPError
from apps.profiles.services.profile import get_or_create_profile

from ..models import Application, ApplicationAnswer

logger = logging.getLogger(__name__)

STANDARD_FIELDS = {"first_name", "last_name", "email", "phone"}
FILE_FIELDS = {"resume", "cover_letter", "resume_text", "cover_letter_text"}


def profile_values(profile) -> dict:
    full = (profile.full_name or profile.user.full_name or "").strip()
    first, _, last = full.partition(" ")
    return {
        "first_name": first, "last_name": last, "email": profile.email or profile.user.email, "phone": profile.phone,
        "linkedin": profile.linkedin_url, "github": profile.github_url, "website": profile.portfolio_url,
        "location": profile.location, "current_title": profile.current_title,
        "current_company": next((e.company for e in profile.employment.filter(is_current=True, verified=True)), ""),
    }


def _guess_profile_value(label: str, values: dict) -> str:
    lowered = label.lower()
    for keyword, key in (("linkedin", "linkedin"), ("github", "github"), ("portfolio", "website"), ("website", "website"),
                         ("current company", "current_company"), ("current employer", "current_company"),
                         ("current title", "current_title"), ("location", "location"), ("city", "location")):
        if keyword in lowered and values.get(key):
            return values[key]
    return ""


def prepare_answers(app: Application, questions=None) -> list[ApplicationAnswer]:
    """Create/refresh ApplicationAnswer rows. Existing user answers are never overwritten."""
    profile = get_or_create_profile(app.user)
    values = profile_values(profile)
    if questions is None:
        questions = []
        if app.job and app.job.source:
            adapter = adapter_for(app.job.source)
            try:
                if adapter.capabilities.application_questions:
                    questions = adapter.fetch_application_questions(app.job.external_id)
            except SourceHTTPError as exc:
                logger.warning("questions_fetch_failed", extra={"application_id": app.pk, "error": str(exc)})
            finally:
                adapter.http.close()
        if not questions:
            from apps.jobs.adapters.base import ApplicationQuestion

            questions = [ApplicationQuestion(k, k.replace("_", " ").title(), "text", True) for k in ("first_name", "last_name", "email")]
            questions += [ApplicationQuestion("phone", "Phone", "text", False), ApplicationQuestion("resume", "Resume/CV", "file", True)]
    out = []
    for q in questions:
        answer = ApplicationAnswer.objects.filter(application=app, question_key=q.key).first()
        if answer and answer.origin == ApplicationAnswer.Origin.USER:
            out.append(answer)
            continue
        if q.sensitive:
            value, origin = "", ApplicationAnswer.Origin.USER
        elif q.key in FILE_FIELDS or q.field_type == "file":
            value, origin = "[attached document]", ApplicationAnswer.Origin.PROFILE
        elif q.key in STANDARD_FIELDS:
            value, origin = values.get(q.key, ""), ApplicationAnswer.Origin.PROFILE
        else:
            value = _guess_profile_value(q.label, values)
            origin = ApplicationAnswer.Origin.PROFILE if value else ApplicationAnswer.Origin.USER
        answer, _ = ApplicationAnswer.objects.update_or_create(
            application=app, question_key=q.key,
            defaults=dict(question_label=q.label[:1000], field_type=q.field_type, required=q.required, options=q.options,
                          answer=value, origin=origin, sensitive=q.sensitive,
                          reviewed_at=timezone.now() if origin == ApplicationAnswer.Origin.PROFILE and value else None),
        )
        out.append(answer)
    return out


def readiness_problems(app: Application) -> list[str]:
    problems = []
    for a in app.answers.all():
        if a.field_type == "file":
            continue
        if a.required and not a.answer.strip():
            problems.append(f"Required question unanswered: {a.question_label[:120]}")
        if a.sensitive and a.answer.strip() and a.reviewed_at is None:
            problems.append(f"Sensitive answer needs your explicit review: {a.question_label[:120]}")
    return problems
