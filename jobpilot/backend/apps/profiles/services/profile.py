"""Profile operations: applying verified resume data, snapshots, and the evidence view
used by matching and tailoring (verified data only)."""
from __future__ import annotations

from datetime import date
from decimal import Decimal

from django.db import transaction
from django.db.models import Max
from django.utils import timezone

from apps.core.skills import canonicalize, category_of

from ..models import (
    CandidateProfile,
    CandidateSkill,
    CareerPreference,
    Certification,
    Education,
    EmploymentHistory,
    Project,
    Resume,
    ResumeVersion,
    Skill,
)

DEFAULT_PREFERENCES = {
    "target_roles": [
        "Python Backend Developer", "Senior Python Developer", "Full Stack Engineer", "Python + Angular Lead",
        "AI Engineer", "Machine Learning Engineer", "Data Engineer", "Technical Lead",
    ],
    "preferred_locations": ["Noida", "Delhi NCR", "India"],
    "work_models": ["hybrid", "remote"],
    "employment_types": ["full_time"],
    "company_types": ["Product", "SaaS", "Technology", "AI startup", "IT services"],
    "salary_currency": "INR",
    "salary_min": 3_000_000,
    "salary_max": 4_000_000,
    "min_match_score": 60,
}


def get_or_create_profile(user) -> CandidateProfile:
    profile, created = CandidateProfile.objects.get_or_create(
        user=user, defaults={"full_name": user.full_name, "email": user.email}
    )
    if created or not hasattr(profile, "preferences"):
        CareerPreference.objects.get_or_create(profile=profile, defaults=DEFAULT_PREFERENCES)
    return profile


def get_skill(name: str) -> Skill | None:
    canonical = canonicalize(name)
    if not canonical:
        return None
    skill = Skill.objects.filter(name__iexact=canonical).first()
    if skill is None:
        skill, _ = Skill.objects.get_or_create(name=canonical, defaults={"category": category_of(canonical)})
    return skill


def _d(value):
    if not value:
        return None
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


@transaction.atomic
def apply_parsed_resume(resume: Resume, data: dict, replace_sections: bool = True) -> CandidateProfile:
    """Write user-reviewed resume data into the profile, marking it verified.

    `data` is the (possibly edited) draft the user confirmed in the UI. Existing
    employment/education/projects/certifications are replaced when `replace_sections`,
    skills are merged.
    """
    profile = get_or_create_profile(resume.user)
    contact = data.get("contact") or {}
    for field in ("full_name", "email", "phone", "linkedin_url", "github_url", "portfolio_url", "location"):
        if contact.get(field):
            setattr(profile, field, contact[field])
    for field in ("headline", "summary"):
        if data.get(field):
            setattr(profile, field, data[field])
    if data.get("total_experience_years") is not None:
        profile.total_experience_years = Decimal(str(data["total_experience_years"]))
    employment = data.get("employment") or []
    if employment:
        current = next((e for e in employment if e.get("is_current")), employment[0])
        profile.current_title = current.get("title", "")[:200] or profile.current_title
    profile.verified_at = timezone.now()
    profile.save()

    if replace_sections:
        profile.employment.all().delete()
        profile.education.all().delete()
        profile.projects.all().delete()
        profile.certifications.all().delete()
    for i, e in enumerate(employment):
        if not (e.get("company") or e.get("title")):
            continue
        EmploymentHistory.objects.create(
            profile=profile, company=(e.get("company") or "")[:200], title=(e.get("title") or "")[:200],
            location=(e.get("location") or "")[:200], start_date=_d(e.get("start_date")), end_date=_d(e.get("end_date")),
            is_current=bool(e.get("is_current")), description=e.get("description") or "",
            achievements=[a for a in e.get("achievements") or [] if a],
            technologies=[canonicalize(t) for t in e.get("technologies") or []], sort_order=i, verified=True,
        )
    for ed in data.get("education") or []:
        if ed.get("institution") or ed.get("degree"):
            Education.objects.create(
                profile=profile, institution=(ed.get("institution") or ed.get("degree"))[:200], degree=(ed.get("degree") or "")[:200],
                field_of_study=(ed.get("field_of_study") or "")[:200], start_year=ed.get("start_year"),
                end_year=ed.get("end_year"), grade=(ed.get("grade") or "")[:50], verified=True,
            )
    for p in data.get("projects") or []:
        if p.get("name"):
            Project.objects.create(
                profile=profile, name=p["name"][:200], role=(p.get("role") or "")[:200], description=p.get("description") or "",
                highlights=[h for h in p.get("highlights") or [] if h], technologies=[canonicalize(t) for t in p.get("technologies") or []],
                url=p.get("url") or "", verified=True,
            )
    for c in data.get("certifications") or []:
        if c.get("name"):
            Certification.objects.create(
                profile=profile, name=c["name"][:200], issuer=(c.get("issuer") or "")[:200], issued_on=_d(c.get("issued_on")),
                credential_url=c.get("credential_url") or "", verified=True,
            )
    for s in data.get("skills") or []:
        name = s.get("name") if isinstance(s, dict) else s
        skill = get_skill(name)
        if skill is None:
            continue
        cs, created = CandidateSkill.objects.get_or_create(
            profile=profile, skill=skill, defaults={"verified": True, "source": "resume"}
        )
        if not created and not cs.verified:
            cs.verified = True
            cs.save(update_fields=["verified", "updated_at"])

    Resume.objects.filter(user=resume.user, is_master=True).exclude(pk=resume.pk).update(is_master=False)
    resume.is_master = True
    resume.parse_status = Resume.ParseStatus.APPLIED
    resume.parsed_data = data
    resume.save(update_fields=["is_master", "parse_status", "parsed_data", "updated_at"])
    create_master_snapshot(profile, source_resume=resume)
    return profile


def next_version_number(user) -> int:
    current = ResumeVersion.objects.select_for_update().filter(user=user).aggregate(m=Max("version_number"))["m"]
    return (current or 0) + 1


def profile_content(profile: CandidateProfile, verified_only: bool = True) -> dict:
    """Structured resume content built from the profile. This is the single source of
    facts that tailoring may select from and reorder, never extend."""
    def vf(qs):
        return qs.filter(verified=True) if verified_only else qs

    return {
        "contact": {
            "full_name": profile.full_name, "email": profile.email, "phone": profile.phone, "location": profile.location,
            "linkedin_url": profile.linkedin_url, "github_url": profile.github_url, "portfolio_url": profile.portfolio_url,
        },
        "headline": profile.headline or profile.current_title,
        "summary": profile.summary,
        "total_experience_years": float(profile.total_experience_years),
        "skills": [cs.skill.name for cs in vf(profile.skills.select_related("skill"))],
        "employment": [
            {
                "id": e.id, "company": e.company, "title": e.title, "location": e.location,
                "start_date": e.start_date.isoformat() if e.start_date else None,
                "end_date": e.end_date.isoformat() if e.end_date else None,
                "is_current": e.is_current, "description": e.description,
                "achievements": list(e.achievements), "technologies": list(e.technologies),
            }
            for e in vf(profile.employment.all())
        ],
        "education": [
            {"id": ed.id, "institution": ed.institution, "degree": ed.degree, "field_of_study": ed.field_of_study,
             "start_year": ed.start_year, "end_year": ed.end_year, "grade": ed.grade}
            for ed in vf(profile.education.all())
        ],
        "projects": [
            {"id": p.id, "name": p.name, "role": p.role, "description": p.description, "highlights": list(p.highlights),
             "technologies": list(p.technologies), "url": p.url}
            for p in vf(profile.projects.all())
        ],
        "certifications": [
            {"id": c.id, "name": c.name, "issuer": c.issuer, "issued_on": c.issued_on.isoformat() if c.issued_on else None}
            for c in vf(profile.certifications.all())
        ],
    }


@transaction.atomic
def create_master_snapshot(profile: CandidateProfile, source_resume: Resume | None = None) -> ResumeVersion:
    user = profile.user
    ResumeVersion.objects.filter(user=user, kind=ResumeVersion.Kind.MASTER, status=ResumeVersion.Status.APPROVED).update(
        status=ResumeVersion.Status.ARCHIVED
    )
    return ResumeVersion.objects.create(
        user=user, source_resume=source_resume, version_number=next_version_number(user),
        label=f"Master snapshot {timezone.now():%Y-%m-%d}", kind=ResumeVersion.Kind.MASTER,
        status=ResumeVersion.Status.APPROVED, content=profile_content(profile), generator="profile", approved_at=timezone.now(),
    )


def current_master(user) -> ResumeVersion | None:
    return (
        ResumeVersion.objects.filter(user=user, kind=ResumeVersion.Kind.MASTER, status=ResumeVersion.Status.APPROVED)
        .order_by("-version_number")
        .first()
    )
