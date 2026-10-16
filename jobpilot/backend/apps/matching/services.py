"""Matching orchestration: build facts from the database, score, persist, notify."""
from __future__ import annotations

import hashlib
import json
import logging

from django.conf import settings
from django.db import transaction
from pgvector.django import CosineDistance

from apps.core import embeddings
from apps.jobs.models import JobPosting
from apps.notifications.models import Notification
from apps.notifications.service import notify
from apps.profiles.models import CandidateProfile

from .models import JobMatch, ScoringConfig
from .scoring import CandidateFacts, JobFacts, score_match

logger = logging.getLogger(__name__)

SEMANTIC_RANGES = {"hashing": (0.05, 0.5), "openai": (0.2, 0.65)}


def get_config(user) -> ScoringConfig:
    config, _ = ScoringConfig.objects.get_or_create(user=user)
    return config


def thresholds(config: ScoringConfig) -> dict:
    return {"excellent": config.excellent_threshold, "good": config.good_threshold, "possible": config.possible_threshold}


def candidate_facts(profile: CandidateProfile) -> CandidateFacts:
    prefs = profile.preferences
    skills = list(profile.skills.select_related("skill"))
    experiences = []
    for e in profile.employment.filter(verified=True):
        text = " ".join([e.title, e.description, *e.achievements])
        from apps.core.skills import extract_skills

        experiences.append({"kind": "employment", "label": f"{e.title} at {e.company}", "title": e.title,
                            "skills": set(e.technologies) | set(extract_skills(text)), "text": text})
    for p in profile.projects.filter(verified=True):
        text = " ".join([p.name, p.description, *p.highlights])
        experiences.append({"kind": "project", "label": p.name, "title": p.name, "skills": set(p.technologies), "text": text})
    return CandidateFacts(
        verified_skills={cs.skill.name for cs in skills if cs.verified},
        unverified_skills={cs.skill.name for cs in skills if not cs.verified},
        years_experience=float(profile.total_experience_years),
        target_roles=list(prefs.target_roles), preferred_locations=list(prefs.preferred_locations),
        work_models=list(prefs.work_models), employment_types=list(prefs.employment_types),
        salary_min=prefs.salary_min, salary_max=prefs.salary_max, salary_currency=prefs.salary_currency,
        open_to_relocation=prefs.open_to_relocation, excluded_companies=list(prefs.excluded_companies),
        excluded_keywords=list(prefs.excluded_keywords), experiences=experiences,
    )


def job_facts(job: JobPosting) -> JobFacts:
    req = job.requirements or {}
    return JobFacts(
        title=job.title, company=job.company.name, seniority=job.seniority,
        required_skills=req.get("required_skills", []), preferred_skills=req.get("preferred_skills", []),
        min_years=req.get("min_years"), max_years=req.get("max_years"), work_model=job.work_model,
        locations=list(job.locations), location_raw=job.location_raw, employment_type=job.employment_type,
        salary_min=job.salary_min, salary_max=job.salary_max, salary_currency=job.salary_currency,
        description=job.description_text[:20000], is_expired=job.status == JobPosting.Status.EXPIRED,
    )


def profile_embedding_text(profile: CandidateProfile) -> str:
    parts = [profile.headline, profile.current_title, profile.summary, " ".join(profile.preferences.target_roles)]
    parts += [cs.skill.name for cs in profile.skills.select_related("skill").filter(verified=True)]
    for e in profile.employment.filter(verified=True):
        parts += [e.title, e.description, *e.achievements, " ".join(e.technologies)]
    for p in profile.projects.filter(verified=True):
        parts += [p.name, p.description, *p.highlights]
    return "\n".join(x for x in parts if x)


def ensure_profile_embedding(profile: CandidateProfile) -> bool:
    text = profile_embedding_text(profile)
    digest = hashlib.sha256(f"{settings.EMBEDDING_PROVIDER}:{text}".encode()).hexdigest()
    if profile.embedding is not None and profile.embedding_hash == digest:
        return False
    profile.embedding = embeddings.embed(text) if text.strip() else None
    profile.embedding_hash = digest
    CandidateProfile.objects.filter(pk=profile.pk).update(embedding=profile.embedding, embedding_hash=digest)
    return True


def _signature(profile, job, config, facts: CandidateFacts) -> str:
    material = {
        "job": job.content_hash, "req": job.requirements_hash, "status": job.status,
        "profile": profile.embedding_hash, "years": facts.years_experience, "skills": sorted(facts.verified_skills),
        "unverified": sorted(facts.unverified_skills), "roles": facts.target_roles, "locs": facts.preferred_locations,
        "wm": facts.work_models, "et": facts.employment_types, "sal": [facts.salary_min, facts.salary_max, facts.salary_currency],
        "reloc": facts.open_to_relocation, "excl": [facts.excluded_companies, facts.excluded_keywords],
        "cfg": [config.weights(), thresholds(config)], "exp": [e["label"] for e in facts.experiences],
    }
    return hashlib.sha256(json.dumps(material, sort_keys=True, default=str).encode()).hexdigest()


def eligible_profiles():
    return CandidateProfile.objects.filter(verified_at__isnull=False).select_related("user", "preferences")


def compute_match(profile: CandidateProfile, job: JobPosting, config=None, facts=None, similarity=None, force=False):
    config = config or get_config(profile.user)
    facts = facts or candidate_facts(profile)
    signature = _signature(profile, job, config, facts)
    existing = JobMatch.objects.filter(user=profile.user, job=job).first()
    if existing and existing.input_signature == signature and not force:
        return existing, False
    if similarity is None and job.embedding is not None and profile.embedding is not None:
        similarity = embeddings.cosine(job.embedding, profile.embedding)
    result = score_match(job_facts(job), facts, config.weights(), thresholds(config), similarity,
                         SEMANTIC_RANGES.get(settings.EMBEDDING_PROVIDER, (0.05, 0.5)))
    values = dict(
        score=result.score, category=result.category, components=result.signals, skills_matched=result.skills_matched,
        skills_missing=result.skills_missing, skills_unverified=result.skills_unverified,
        relevant_experience=result.relevant_experience, gaps=[g[:300] for g in result.gaps], location_fit=result.location_fit[:300],
        salary_fit=result.salary_fit[:300], explanation=result.explanation, recommended_action=result.recommended_action[:300],
        semantic_similarity=similarity, information_coverage=result.coverage, excluded_reason=result.excluded_reason[:300],
        input_signature=signature,
    )
    with transaction.atomic():
        match, created = JobMatch.objects.update_or_create(user=profile.user, job=job, defaults=values)
    _maybe_notify(match)
    _maybe_auto_actions(match)
    return match, True


def _maybe_notify(match: JobMatch):
    from apps.automation.services import get_settings

    if match.notified_at or match.excluded_reason or match.job.status != JobPosting.Status.ACTIVE or match.job.duplicate_of_id:
        return
    if match.score < get_settings(match.user).notify_min_score:
        return
    created = notify(
        match.user, Notification.Kind.HIGH_MATCH, f"{match.score}/100 · {match.job.title} at {match.job.company.name}",
        body=match.explanation[:500], link=f"/jobs/{match.job_id}", obj=match.job, dedupe_key=f"high_match:{match.job_id}",
    )
    if created:
        JobMatch.objects.filter(pk=match.pk).update(notified_at=created.created_at)


def _maybe_auto_actions(match: JobMatch):
    from apps.automation.services import get_settings

    auto = get_settings(match.user)
    if (auto.auto_shortlist_min_score and match.score >= auto.auto_shortlist_min_score and match.user_state == JobMatch.UserState.NEW
            and not match.excluded_reason and match.job.status == JobPosting.Status.ACTIVE):
        JobMatch.objects.filter(pk=match.pk).update(user_state=JobMatch.UserState.SHORTLISTED)


def match_job_for_all_users(job: JobPosting) -> int:
    if job.duplicate_of_id or job.status != JobPosting.Status.ACTIVE:
        return 0
    count = 0
    for profile in eligible_profiles():
        ensure_profile_embedding(profile)
        _, changed = compute_match(profile, job)
        count += int(changed)
    return count


def match_all_jobs_for_user(user, force=False) -> int:
    profile = CandidateProfile.objects.filter(user=user).select_related("preferences").first()
    if profile is None or profile.verified_at is None:
        return 0
    ensure_profile_embedding(profile)
    config, facts = get_config(user), candidate_facts(profile)
    jobs = JobPosting.objects.filter(status=JobPosting.Status.ACTIVE, duplicate_of__isnull=True).select_related("company")
    if profile.embedding is not None:
        # pgvector computes similarity in the database (cosine distance -> similarity).
        jobs = jobs.annotate(distance=CosineDistance("embedding", profile.embedding))
    count = 0
    for job in jobs.iterator(chunk_size=200):
        distance = getattr(job, "distance", None)
        similarity = None if distance is None else float(1 - distance)
        _, changed = compute_match(profile, job, config, facts, similarity, force)
        count += int(changed)
    return count


def semantic_search(user, limit=50):
    """Active jobs nearest to the profile embedding (pgvector HNSW index)."""
    profile = CandidateProfile.objects.filter(user=user).first()
    if profile is None or profile.embedding is None:
        return JobPosting.objects.none()
    return (JobPosting.objects.filter(status=JobPosting.Status.ACTIVE, duplicate_of__isnull=True, embedding__isnull=False)
            .annotate(distance=CosineDistance("embedding", profile.embedding)).order_by("distance")[:limit])
