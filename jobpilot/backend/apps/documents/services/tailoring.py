"""Factual resume tailoring.

Tailoring may *select, reorder and lightly rephrase* verified profile facts. It may
never add employers, titles, dates, degrees, certifications, skills or metrics.
Every tailored version is validated against the verified profile snapshot
(`validate_tailored_content`) — for AI output and for user edits alike — and is
created in `awaiting_approval` status; it cannot be used for an application until the
user approves it.
"""
from __future__ import annotations

import copy
import re

from django.db import transaction
from django.utils import timezone

from apps.core import llm
from apps.core.skills import extract_skills
from apps.core.text import normalize_token
from apps.profiles.models import ResumeVersion
from apps.profiles.services.profile import (
    create_master_snapshot,
    current_master,
    get_or_create_profile,
    next_version_number,
    profile_content,
)

NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)?%?")


class TailoringError(Exception):
    pass


# ----------------------------------------------------------------- relevance


def _keywords(job) -> set[str]:
    req = job.requirements or {}
    words = set()
    for resp in req.get("responsibilities", []) + [job.title]:
        words |= {w for w in re.findall(r"[a-z][a-z+#.\-]{3,}", resp.lower())}
    return words


def _relevance(text: str, skills: set[str], keywords: set[str]) -> float:
    found = set(extract_skills(text))
    lowered = text.lower()
    return 2.0 * len(found & skills) + 0.3 * sum(1 for k in keywords if k in lowered)


def requirement_map(job, content: dict) -> list[dict]:
    req = job.requirements or {}
    verified = set(content["skills"])
    rows = []
    for importance, names in (("required", req.get("required_skills", [])), ("preferred", req.get("preferred_skills", []))):
        for name in names:
            evidence = []
            for e in content["employment"]:
                for i, a in enumerate(e["achievements"]):
                    if name in extract_skills(a):
                        evidence.append({"type": "achievement", "employment_id": e["id"], "index": i,
                                         "label": f"{e['title']} at {e['company']}", "text": a})
                if name in e.get("technologies", []) and not any(ev.get("employment_id") == e["id"] for ev in evidence):
                    evidence.append({"type": "technology", "employment_id": e["id"], "label": f"{e['title']} at {e['company']}"})
            for p in content["projects"]:
                if name in p.get("technologies", []) or name in extract_skills(" ".join([p["description"], *p["highlights"]])):
                    evidence.append({"type": "project", "project_id": p["id"], "label": p["name"]})
            status = "matched" if name in verified else ("evidence_only" if evidence else "missing")
            rows.append({"requirement": name, "importance": importance, "status": status, "evidence": evidence[:4]})
    return rows


# ------------------------------------------------------------- deterministic


def tailor_content(job, base: dict) -> tuple[dict, dict]:
    req = job.requirements or {}
    wanted = set(req.get("required_skills", [])) | set(req.get("preferred_skills", []))
    keywords = _keywords(job)
    content = copy.deepcopy(base)
    verified = list(base["skills"])
    required = [s for s in req.get("required_skills", []) if s in verified]
    preferred = [s for s in req.get("preferred_skills", []) if s in verified and s not in required]
    content["skills"] = required + preferred + [s for s in verified if s not in required and s not in preferred]
    for e in content["employment"]:
        indexed = list(enumerate(e["achievements"]))
        indexed.sort(key=lambda pair: -_relevance(pair[1], wanted, keywords))
        e["achievements"] = [a for _, a in indexed]
        e["achievement_sources"] = [i for i, _ in indexed]
    content["projects"].sort(key=lambda p: -_relevance(" ".join([p["name"], p["description"], *p["highlights"]]), wanted, keywords))
    content["projects"] = content["projects"][:4]
    notes = {
        "requirement_map": requirement_map(job, base),
        "warnings": [],
        "changes": ["Skills reordered to put the job's required skills first.",
                    "Achievements reordered by relevance to the job's requirements."],
        "rewrites": [],
    }
    missing = [r["requirement"] for r in notes["requirement_map"] if r["status"] == "missing" and r["importance"] == "required"]
    if missing:
        notes["warnings"].append(
            "Not claimed because they are not in your verified profile: " + ", ".join(missing[:10])
            + ". Add them to your profile only if they are genuinely part of your experience."
        )
    return content, notes


# ------------------------------------------------------------- AI rephrasing

_REWRITE_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["summary", "bullets"],
    "properties": {
        "summary": {"type": "string"},
        "bullets": {"type": "array", "items": {
            "type": "object", "additionalProperties": False, "required": ["employment_id", "index", "text"],
            "properties": {"employment_id": {"type": "integer"}, "index": {"type": "integer"}, "text": {"type": "string"}},
        }},
    },
}


def _numbers(text: str) -> set[str]:
    return {n.replace(",", "") for n in NUMBER_RE.findall(text or "")}


def check_claims(new_text: str, source_text: str, verified_skills: set[str], known_entities: set[str]) -> list[str]:
    """Return reasons a rephrased sentence introduces facts not present in its source."""
    problems = []
    extra_numbers = _numbers(new_text) - _numbers(source_text)
    if extra_numbers:
        problems.append(f"new figures {sorted(extra_numbers)}")
    new_skills = set(extract_skills(new_text)) - set(extract_skills(source_text)) - verified_skills
    if new_skills:
        problems.append(f"unverified skills {sorted(new_skills)}")
    capitalised = set(re.findall(r"\b[A-Z][a-zA-Z]{2,}(?:\s[A-Z][a-zA-Z]{2,})*\b", new_text))
    source_norm = normalize_token(source_text)
    unknown = {c for c in capitalised if normalize_token(c) not in source_norm and normalize_token(c) not in known_entities
               and c.lower() not in {s.lower() for s in verified_skills} and len(c) > 3}
    # Sentence-initial words are commonly capitalised; only flag multi-word or mid-sentence proper nouns.
    unknown = {c for c in unknown if " " in c or not new_text.startswith(c)}
    if unknown:
        problems.append(f"unknown names {sorted(unknown)[:3]}")
    return problems


def rephrase_with_llm(job, content: dict, notes: dict) -> None:
    if not llm.is_enabled():
        return
    bullets = [{"employment_id": e["id"], "index": i, "text": a}
               for e in content["employment"] for i, a in enumerate(e["achievements"][:5])]
    trusted = {"summary": content["summary"], "skills": content["skills"], "bullets": bullets,
               "employment": [{"id": e["id"], "company": e["company"], "title": e["title"]} for e in content["employment"]]}
    try:
        result = llm.generate_json(
            "resume_tailoring",
            "Rephrase the candidate's summary and the given achievement bullets so they emphasise relevance to the job. "
            "Rules: keep every fact exactly as given; do not add numbers, percentages, technologies, employers, titles, "
            "certifications or outcomes that are not in that same bullet; keep each bullet to one sentence; if a bullet "
            "cannot be improved, return it unchanged. Return the same employment_id/index for each bullet.",
            _REWRITE_SCHEMA, trusted_context=trusted,
            untrusted_blocks={"job_title": job.title, "job_requirements": "\n".join(
                (job.requirements or {}).get("required_skills", []) + (job.requirements or {}).get("responsibilities", []))},
        )
    except llm.LLMUnavailable:
        notes["warnings"].append("AI rephrasing unavailable; achievements kept in your original wording.")
        return
    verified = set(content["skills"])
    entities = {normalize_token(x) for e in content["employment"] for x in (e["company"], e["title"], e["location"])}
    by_key = {(e["id"], i): (e, i) for e in content["employment"] for i in range(len(e["achievements"]))}
    for b in result.data.get("bullets", []):
        target = by_key.get((b.get("employment_id"), b.get("index")))
        if not target or not b.get("text"):
            continue
        e, i = target
        original = e["achievements"][i]
        problems = check_claims(b["text"], original, verified, entities)
        if problems:
            notes["warnings"].append(f"Kept original wording for one bullet at {e['company']} (AI draft had {'; '.join(problems)}).")
            continue
        if b["text"].strip() != original:
            e["achievements"][i] = b["text"].strip()[:600]
            notes["rewrites"].append({"employment_id": e["id"], "original": original, "rewritten": e["achievements"][i]})
    summary = (result.data.get("summary") or "").strip()
    if summary and summary != content["summary"]:
        source = " ".join([content["summary"], *[a for e in content["employment"] for a in e["achievements"]]])
        problems = check_claims(summary, source, verified, entities)
        years = re.findall(r"(\d+)\+?\s*years", summary)
        if any(float(y) > content["total_experience_years"] + 0.5 for y in years):
            problems.append("overstated years of experience")
        if problems:
            notes["warnings"].append(f"Kept original summary (AI draft had {'; '.join(problems)}).")
        else:
            notes["rewrites"].append({"section": "summary", "original": content["summary"], "rewritten": summary})
            content["summary"] = summary[:1500]


# ---------------------------------------------------------------- validation


def validate_tailored_content(content: dict, base: dict) -> list[str]:
    """Structural fact check against the verified base. Returns a list of violations."""
    errors = []
    base_emp = {e["id"]: e for e in base["employment"]}
    for e in content.get("employment", []):
        src = base_emp.get(e.get("id"))
        if src is None:
            errors.append(f"Unknown employment entry '{e.get('company')}'")
            continue
        for key in ("company", "title", "start_date", "end_date", "is_current"):
            if e.get(key) != src.get(key):
                errors.append(f"{key} of '{src['company']}' cannot be changed in a tailored resume")
    if len(content.get("employment", [])) != len(base["employment"]):
        errors.append("Employment history must list every verified role (roles cannot be added or hidden).")
    unknown_skills = set(content.get("skills", [])) - set(base["skills"])
    if unknown_skills:
        errors.append(f"Skills not in your verified profile: {', '.join(sorted(unknown_skills))}")
    base_edu = {(x["institution"], x["degree"], x["end_year"]) for x in base["education"]}
    for x in content.get("education", []):
        if (x.get("institution"), x.get("degree"), x.get("end_year")) not in base_edu:
            errors.append(f"Education entry '{x.get('institution')}' does not match your profile")
    base_certs = {c["name"] for c in base["certifications"]}
    for c in content.get("certifications", []):
        if c.get("name") not in base_certs:
            errors.append(f"Certification '{c.get('name')}' is not in your profile")
    base_projects = {p["id"] for p in base["projects"]}
    for p in content.get("projects", []):
        if p.get("id") not in base_projects:
            errors.append(f"Project '{p.get('name')}' is not in your profile")
    if content.get("contact") != base.get("contact"):
        errors.append("Contact details must match your profile")
    if content.get("total_experience_years") != base.get("total_experience_years"):
        errors.append("Total experience cannot be changed in a tailored resume")
    return errors


# -------------------------------------------------------------------- entry


@transaction.atomic
def create_tailored_version(user, job) -> ResumeVersion:
    profile = get_or_create_profile(user)
    if profile.verified_at is None:
        raise TailoringError("Verify your profile (apply a parsed resume or confirm your details) before tailoring.")
    base_version = current_master(user)
    base = profile_content(profile)
    if base_version is None or base_version.content != base:
        base_version = create_master_snapshot(profile)
    if not base["employment"]:
        raise TailoringError("Your verified profile has no employment history to tailor.")
    content, notes = tailor_content(job, base)
    rephrase_with_llm(job, content, notes)
    errors = validate_tailored_content(content, base)
    if errors:  # defensive: deterministic path should never violate
        raise TailoringError("; ".join(errors))
    return ResumeVersion.objects.create(
        user=user, base_version=base_version, job=job, version_number=next_version_number(user),
        label=f"{job.title} – {job.company.name}"[:200], kind=ResumeVersion.Kind.TAILORED,
        status=ResumeVersion.Status.AWAITING_APPROVAL, content=content, tailoring_notes=notes,
        generator="rules" + ("+" + llm_provider() if notes["rewrites"] else ""),
    )


def llm_provider():
    from django.conf import settings

    return settings.LLM_PROVIDER


def apply_user_edit(version: ResumeVersion, new_content: dict) -> ResumeVersion:
    if version.kind != ResumeVersion.Kind.TAILORED:
        raise TailoringError("Only tailored versions can be edited.")
    if version.status not in (ResumeVersion.Status.AWAITING_APPROVAL, ResumeVersion.Status.DRAFT):
        raise TailoringError("Approved or rejected versions cannot be edited; generate a new version instead.")
    base = version.base_version.content if version.base_version else profile_content(get_or_create_profile(version.user))
    errors = validate_tailored_content(new_content, base)
    if errors:
        raise TailoringError("; ".join(errors))
    version.content = new_content
    version.status = ResumeVersion.Status.AWAITING_APPROVAL
    version.tailoring_notes = {**version.tailoring_notes, "edited_by_user_at": timezone.now().isoformat()}
    version.save(update_fields=["content", "status", "tailoring_notes", "updated_at"])
    return version


def compare(version: ResumeVersion) -> dict:
    """Side-by-side data: master (base) vs tailored, with per-section change markers."""
    base = version.base_version.content if version.base_version else {}
    tailored = version.content
    sections = []
    sections.append({"section": "summary", "master": base.get("summary", ""), "tailored": tailored.get("summary", ""),
                     "changed": base.get("summary") != tailored.get("summary")})
    sections.append({"section": "skills", "master": base.get("skills", []), "tailored": tailored.get("skills", []),
                     "changed": base.get("skills") != tailored.get("skills")})
    base_emp = {e["id"]: e for e in base.get("employment", [])}
    for e in tailored.get("employment", []):
        b = base_emp.get(e["id"], {})
        sections.append({"section": f"employment:{e['id']}", "heading": f"{e['title']} – {e['company']}",
                         "master": b.get("achievements", []), "tailored": e["achievements"],
                         "changed": b.get("achievements") != e["achievements"]})
    sections.append({"section": "projects", "master": [p["name"] for p in base.get("projects", [])],
                     "tailored": [p["name"] for p in tailored.get("projects", [])],
                     "changed": [p["id"] for p in base.get("projects", [])] != [p["id"] for p in tailored.get("projects", [])]})
    return {"version_id": version.id, "base_version_id": version.base_version_id, "sections": sections,
            "requirement_map": version.tailoring_notes.get("requirement_map", []),
            "warnings": version.tailoring_notes.get("warnings", []), "rewrites": version.tailoring_notes.get("rewrites", [])}
