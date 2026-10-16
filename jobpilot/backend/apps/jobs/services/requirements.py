"""Extract structured requirements from a job description (untrusted text)."""
from __future__ import annotations

import hashlib
import re
from decimal import Decimal

from django.db import transaction

from apps.core import embeddings, llm
from apps.core.skills import canonicalize, extract_skills
from apps.core.text import normalize_token

from ..models import JobPosting, JobSkill

REQUIRED_HEAD = re.compile(
    r"^(requirements|qualifications|minimum qualifications|basic qualifications|required( skills| qualifications)?|"
    r"must[- ]haves?|what (you'll|you will) (need|bring)|what we('re| are) looking for|who you are|you have|"
    r"skills (and|&) experience|about you|your profile|key skills)\b",
    re.I,
)
PREFERRED_HEAD = re.compile(
    r"^(preferred( qualifications| skills)?|nice[- ]to[- ]haves?|bonus( points)?|good to have|it('s| is| would be) (a )?(plus|great)|"
    r"additional (qualifications|skills)|desired skills|pluses)\b",
    re.I,
)
RESP_HEAD = re.compile(r"^(responsibilities|key responsibilities|what (you'll|you will) do|the role|your role|in this role|day to day)\b", re.I)
YEARS_RE = re.compile(
    r"(?P<lo>\d{1,2})\s*(?:\+|plus)?\s*(?:(?:-|–|to)\s*(?P<hi>\d{1,2})\s*)?(?:\+\s*)?years?(?:'|’)?\s*(?:of\s+)?"
    r"(?:(?:professional|relevant|industry|hands[- ]on|software|work|total)\s+)*(?:experience|exp\b)",
    re.I,
)


def _sections(text: str):
    current = "general"
    out = {"general": [], "required": [], "preferred": [], "responsibilities": []}
    for line in text.splitlines():
        stripped = line.strip().strip(":").strip()
        if not stripped:
            continue
        if len(stripped) < 80:
            if REQUIRED_HEAD.match(stripped):
                current = "required"
                continue
            if PREFERRED_HEAD.match(stripped):
                current = "preferred"
                continue
            if RESP_HEAD.match(stripped):
                current = "responsibilities"
                continue
        out[current].append(stripped)
    return out


def extract_requirements_rules(title: str, text: str) -> dict:
    sections = _sections(text)
    req_text = "\n".join(sections["required"])
    pref_text = "\n".join(sections["preferred"])
    other_text = "\n".join(sections["general"] + sections["responsibilities"])
    title_skills = extract_skills(title)
    required = extract_skills(req_text)
    preferred = [s for s in extract_skills(pref_text) if s not in required]
    other = [s for s in extract_skills(other_text) if s not in required and s not in preferred]
    if not sections["required"] and not sections["preferred"]:
        required, other = other, []  # no structure: everything mentioned counts as a requirement
    # Skills named in the title are always required (e.g. "Senior Python Developer").
    for s in title_skills:
        if s not in required:
            required.insert(0, s)
        if s in preferred:
            preferred.remove(s)
    preferred += [s for s in other if s not in preferred]
    lows, highs = [], []
    for m in YEARS_RE.finditer(text):
        lo = int(m.group("lo"))
        if 0 < lo <= 25:
            lows.append(lo)
            if m.group("hi"):
                highs.append(int(m.group("hi")))
    return {
        "required_skills": required[:40],
        "preferred_skills": preferred[:40],
        "min_years": min(lows) if lows else None,
        "max_years": max(highs) if highs else None,
        "responsibilities": [r.lstrip("•-* ") for r in sections["responsibilities"]][:20],
        "qualifications": [q.lstrip("•-* ") for q in sections["required"]][:25],
        "extractor": "rules",
    }


_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["required_skills", "preferred_skills", "min_years", "responsibilities"],
    "properties": {
        "required_skills": {"type": "array", "items": {"type": "string"}},
        "preferred_skills": {"type": "array", "items": {"type": "string"}},
        "min_years": {"type": ["integer", "null"]},
        "responsibilities": {"type": "array", "items": {"type": "string"}},
    },
}


def extract_requirements(title: str, text: str) -> dict:
    result = extract_requirements_rules(title, text)
    if not llm.is_enabled():
        return result
    try:
        out = llm.generate_json(
            "job_requirements",
            "Extract the job's required skills, preferred skills, minimum years of experience and key responsibilities. "
            "Use short canonical technology names. Only include items explicitly stated in the posting.",
            _SCHEMA,
            untrusted_blocks={"job_title": title, "job_description": text},
        ).data
    except llm.LLMUnavailable:
        return result
    norm = normalize_token(text + " " + title)

    def grounded(skills):
        return [canonicalize(s) for s in skills if s and normalize_token(s) in norm]

    req = list(dict.fromkeys(result["required_skills"] + grounded(out.get("required_skills", []))))
    pref = [s for s in dict.fromkeys(result["preferred_skills"] + grounded(out.get("preferred_skills", []))) if s not in req]
    result.update(required_skills=req[:40], preferred_skills=pref[:40], extractor=f"rules+{settings_provider()}")
    if out.get("min_years") and str(out["min_years"]) in text:
        result["min_years"] = out["min_years"]
    if out.get("responsibilities"):
        result["responsibilities"] = [r for r in out["responsibilities"] if r][:20]
    return result


def settings_provider():
    from django.conf import settings

    return settings.LLM_PROVIDER


def job_embedding_text(job: JobPosting) -> str:
    req = job.requirements or {}
    return "\n".join([
        job.title, job.title, " ".join(req.get("required_skills", [])), " ".join(req.get("preferred_skills", [])),
        " ".join(req.get("responsibilities", [])), job.description_text[:6000],
    ])


def analyze_job(job: JobPosting, force: bool = False) -> bool:
    """(Re)extract requirements, JobSkill rows and the embedding when content changed."""
    from apps.profiles.services.profile import get_skill

    if not force and job.requirements_hash == job.content_hash and job.embedding is not None:
        return False
    requirements = extract_requirements(job.title, job.description_text)
    vector = embeddings.embed(job_embedding_text(_with_req(job, requirements)))
    with transaction.atomic():
        job.requirements = requirements
        job.requirements_hash = job.content_hash
        job.min_years_experience = Decimal(requirements["min_years"]) if requirements.get("min_years") is not None else None
        job.embedding = vector
        job.embedding_hash = hashlib.sha256(job.content_hash.encode()).hexdigest()
        job.save(update_fields=["requirements", "requirements_hash", "min_years_experience", "embedding", "embedding_hash", "updated_at"])
        JobSkill.objects.filter(job=job).delete()
        rows = []
        for name in requirements["required_skills"]:
            skill = get_skill(name)
            if skill:
                rows.append(JobSkill(job=job, skill=skill, importance=JobSkill.Importance.REQUIRED))
        seen = {r.skill_id for r in rows}
        for name in requirements["preferred_skills"]:
            skill = get_skill(name)
            if skill and skill.id not in seen:
                seen.add(skill.id)
                rows.append(JobSkill(job=job, skill=skill, importance=JobSkill.Importance.PREFERRED))
        JobSkill.objects.bulk_create(rows, ignore_conflicts=True)
    return True


def _with_req(job, requirements):
    job.requirements = requirements
    return job
