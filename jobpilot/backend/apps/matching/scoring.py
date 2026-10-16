"""Explainable job-match scoring.

Every signal returns a value in [0, 1] or ``None`` when the information needed to
judge it is unavailable. Unknown signals are excluded and the remaining weights are
renormalised, so missing optional information (e.g. salary) never auto-rejects a job.
Only *verified* profile facts count as evidence; nothing is inferred.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from apps.jobs.normalization import location_tags, normalize_title

SENIORITY_ORDER = ["intern", "junior", "mid", "senior", "lead", "principal"]
ROLE_SYNONYMS = {
    "backend": {"backend", "server", "api", "platform"},
    "full stack": {"full stack", "fullstack"},
    "machine learning": {"machine learning", "ml", "ai"},
    "ai": {"ai", "machine learning", "llm", "genai", "applied ai"},
    "data": {"data"},
    "lead": {"lead", "staff", "principal", "head"},
    "developer": {"developer", "engineer", "programmer", "sde"},
    "engineer": {"engineer", "developer", "programmer", "sde"},
}
_STOP = {"of", "and", "the", "a", "an", "for", "with", "in", "at", "+", "-", "&", "senior", "junior", "sr", "jr", "ii", "iii"}


@dataclass
class CandidateFacts:
    verified_skills: set[str]
    unverified_skills: set[str]
    years_experience: float
    target_roles: list[str]
    preferred_locations: list[str]
    work_models: list[str]
    employment_types: list[str]
    salary_min: int | None
    salary_max: int | None
    salary_currency: str
    open_to_relocation: bool
    excluded_companies: list[str]
    excluded_keywords: list[str]
    experiences: list[dict] = field(default_factory=list)  # {kind, label, skills:set, text}


@dataclass
class JobFacts:
    title: str
    company: str
    seniority: str
    required_skills: list[str]
    preferred_skills: list[str]
    min_years: float | None
    max_years: float | None
    work_model: str
    locations: list[str]
    location_raw: str
    employment_type: str
    salary_min: int | None
    salary_max: int | None
    salary_currency: str
    description: str = ""
    is_expired: bool = False


@dataclass
class Signal:
    name: str
    value: float | None
    weight: float
    detail: str

    def as_dict(self):
        return {"score": None if self.value is None else round(self.value, 3), "weight": self.weight, "detail": self.detail}


@dataclass
class MatchResult:
    score: int
    category: str
    signals: dict
    skills_matched: list[str]
    skills_missing: list[str]
    skills_unverified: list[str]
    relevant_experience: list[dict]
    gaps: list[str]
    location_fit: str
    salary_fit: str
    explanation: str
    recommended_action: str
    coverage: float
    excluded_reason: str = ""


def _clamp(x):
    return max(0.0, min(1.0, x))


def _tokens(text: str) -> set[str]:
    out = set()
    norm = normalize_title(text)
    for tok in re.split(r"[\s/,]+", norm):
        if tok and tok not in _STOP:
            out.add(tok)
    for key, syns in ROLE_SYNONYMS.items():
        if key in norm:
            out |= syns
    return out


def expected_seniority(years: float) -> str:
    if years < 1:
        return "intern"
    if years < 2.5:
        return "junior"
    if years < 5:
        return "mid"
    if years < 9:
        return "senior"
    return "lead"


# ------------------------------------------------------------------ signals


def skills_signal(job_skills, candidate: CandidateFacts, name, weight, label):
    if not job_skills:
        return Signal(name, None, weight, f"The posting lists no {label} skills."), [], [], []
    matched = [s for s in job_skills if s in candidate.verified_skills]
    unverified = [s for s in job_skills if s not in candidate.verified_skills and s in candidate.unverified_skills]
    missing = [s for s in job_skills if s not in candidate.verified_skills and s not in candidate.unverified_skills]
    value = len(matched) / len(job_skills)
    detail = f"{len(matched)} of {len(job_skills)} {label} skills verified in your profile"
    if unverified:
        detail += f"; {len(unverified)} listed but not yet verified"
    return Signal(name, value, weight, detail), matched, missing, unverified


def experience_signal(job: JobFacts, candidate: CandidateFacts, weight):
    if job.min_years is None:
        return Signal("experience", None, weight, "The posting does not state required years of experience."), None
    years, need = candidate.years_experience, float(job.min_years)
    if years >= need:
        value = 1.0
        if job.max_years and years > float(job.max_years) + 3:
            value = 0.75
            return Signal("experience", value, weight,
                          f"You have {years:g} years; the role asks for {need:g}–{float(job.max_years):g} (you may be overqualified)."), None
        return Signal("experience", value, weight, f"You have {years:g} years; the role asks for {need:g}+."), None
    shortfall = need - years
    value = _clamp(1 - 1.5 * shortfall / max(need, 1))
    return Signal("experience", value, weight, f"You have {years:g} years; the role asks for {need:g}+."), \
        f"Experience gap: {shortfall:g} year(s) below the stated minimum"


def role_signal(job: JobFacts, candidate: CandidateFacts, weight):
    job_tokens = _tokens(job.title)
    best_role, best = "", 0.0
    for role in candidate.target_roles:
        role_tokens = _tokens(role)
        if not role_tokens or not job_tokens:
            continue
        overlap = len(job_tokens & role_tokens) / len(role_tokens)
        if overlap > best:
            best_role, best = role, overlap
    title_score = _clamp(best)
    exp = expected_seniority(candidate.years_experience)
    if job.seniority in SENIORITY_ORDER:
        distance = abs(SENIORITY_ORDER.index(job.seniority) - SENIORITY_ORDER.index(exp))
        seniority_score = {0: 1.0, 1: 0.7, 2: 0.3}.get(distance, 0.0)
        seniority_note = f"seniority '{job.seniority}' vs your expected level '{exp}'"
    elif job.seniority == "manager":
        seniority_score = 0.5 if exp in {"senior", "lead"} else 0.2
        seniority_note = "people-management role"
    else:
        seniority_score, seniority_note = 0.6, "seniority not stated"
    if not candidate.target_roles:
        return Signal("role_alignment", seniority_score, weight, f"No target roles configured; {seniority_note}.")
    value = 0.6 * title_score + 0.4 * seniority_score
    match_note = f"closest target role '{best_role}'" if best_role else "title does not resemble your target roles"
    return Signal("role_alignment", value, weight, f"{match_note}; {seniority_note}.")


def semantic_signal(similarity: float | None, weight, floor=0.05, ceiling=0.5):
    if similarity is None:
        return Signal("semantic", None, weight, "Semantic similarity unavailable (profile or job not embedded yet).")
    value = _clamp((similarity - floor) / (ceiling - floor))
    return Signal("semantic", value, weight, f"Description similarity to your profile: {similarity:.2f} (cosine).")


def location_signal(job: JobFacts, candidate: CandidateFacts, weight):
    pref_tags = location_tags(candidate.preferred_locations)
    job_tags = location_tags(job.locations or ([job.location_raw] if job.location_raw else []))
    prefers_remote = "remote" in candidate.work_models
    if job.work_model == "remote":
        restricted = [t for t in job_tags if t not in {"remote", "remote india"} and t not in pref_tags]
        if prefers_remote and ("remote india" in job_tags or not restricted or job_tags & pref_tags):
            return Signal("location", 1.0, weight, "Remote role, and you accept remote work."), "Remote – matches your preference"
        if prefers_remote:
            return Signal("location", 0.5, weight, f"Remote, but possibly restricted to: {job.location_raw}"), \
                f"Remote with location restriction ({job.location_raw}) – check eligibility"
        return Signal("location", 0.6, weight, "Remote role; remote is not in your preferred work models."), "Remote (not a stated preference)"
    if not job_tags and job.work_model == "unknown":
        return Signal("location", None, weight, "Location not stated."), "Location not stated"
    in_area = bool(job_tags & pref_tags)
    model_ok = job.work_model in candidate.work_models or job.work_model == "unknown"
    if in_area:
        value = 1.0 if model_ok else 0.7
        fit = f"{job.location_raw or ', '.join(job.locations)} is in your preferred area"
        if not model_ok:
            fit += f" ({job.work_model}, not a preferred work model)"
        return Signal("location", value, weight, fit + "."), fit
    value = 0.4 if candidate.open_to_relocation else 0.1
    fit = f"{job.location_raw or ', '.join(job.locations)} is outside your preferred locations"
    if candidate.open_to_relocation:
        fit += " (you are open to relocation)"
    return Signal("location", value, weight, fit + "."), fit


def salary_signal(job: JobFacts, candidate: CandidateFacts, weight):
    if not (job.salary_min or job.salary_max):
        return Signal("salary", None, weight, "Salary not disclosed."), "Salary not disclosed"
    if not candidate.salary_min and not candidate.salary_max:
        return Signal("salary", None, weight, "No salary expectation configured."), "No expectation configured"
    if job.salary_currency and candidate.salary_currency and job.salary_currency != candidate.salary_currency:
        return Signal("salary", None, weight, f"Salary in {job.salary_currency}; your expectation is in {candidate.salary_currency}."), \
            f"Posted in {job.salary_currency} – compare manually"
    job_hi = job.salary_max or job.salary_min
    want = candidate.salary_min or candidate.salary_max
    fmt = _money(candidate.salary_currency)
    if job_hi >= want:
        return Signal("salary", 1.0, weight, f"Range up to {fmt(job_hi)} meets your minimum of {fmt(want)}."), \
            f"Meets expectation (up to {fmt(job_hi)})"
    ratio = job_hi / want
    value = _clamp((ratio - 0.6) / 0.4)
    return Signal("salary", value, weight, f"Range up to {fmt(job_hi)} is below your minimum of {fmt(want)}."), \
        f"Below expectation (up to {fmt(job_hi)} vs {fmt(want)})"


def _money(currency):
    def fmt(amount):
        if currency == "INR" and amount >= 100_000:
            return f"₹{amount / 100_000:.1f} LPA"
        return f"{currency} {amount:,.0f}".strip()
    return fmt


def employment_signal(job: JobFacts, candidate: CandidateFacts, weight):
    if not job.employment_type or not candidate.employment_types:
        return Signal("employment_type", None, weight, "Employment type not stated.")
    ok = job.employment_type in candidate.employment_types
    return Signal("employment_type", 1.0 if ok else 0.2, weight,
                  f"{job.employment_type.replace('_', ' ')} {'matches' if ok else 'is not in'} your preferences.")


# ----------------------------------------------------------------- compose


def relevant_experience(job: JobFacts, candidate: CandidateFacts, limit=3):
    wanted = set(job.required_skills) | set(job.preferred_skills)
    title_tokens = _tokens(job.title)
    ranked = []
    for exp in candidate.experiences:
        overlap = sorted(wanted & exp["skills"])
        title_overlap = len(title_tokens & _tokens(exp.get("title", ""))) if exp["kind"] == "employment" else 0
        score = len(overlap) + 0.5 * title_overlap
        if score > 0:
            ranked.append((score, {"kind": exp["kind"], "label": exp["label"], "skills": overlap[:8]}))
    ranked.sort(key=lambda x: -x[0])
    return [r[1] for r in ranked[:limit]]


def categorize(score: int, thresholds: dict) -> str:
    if score >= thresholds["excellent"]:
        return "excellent"
    if score >= thresholds["good"]:
        return "good"
    if score >= thresholds["possible"]:
        return "possible"
    return "low"


ACTIONS = {
    "excellent": "Strong fit: generate a tailored resume, review it, and apply.",
    "good": "Good fit: review the gaps below, then tailor your resume and apply.",
    "possible": "Partial fit: read the posting carefully and apply only if the gaps are acceptable.",
    "low": "Weak fit: skip unless you have context the profile does not capture.",
}


def score_match(job: JobFacts, candidate: CandidateFacts, weights: dict, thresholds: dict,
                similarity: float | None = None, semantic_range=(0.05, 0.5)) -> MatchResult:
    req_sig, matched, missing, unverified = skills_signal(job.required_skills, candidate, "required_skills",
                                                          weights["required_skills"], "required")
    pref_sig, p_matched, p_missing, p_unverified = skills_signal(job.preferred_skills, candidate, "preferred_skills",
                                                                 weights["preferred_skills"], "preferred")
    exp_sig, exp_gap = experience_signal(job, candidate, weights["experience"])
    loc_sig, location_fit = location_signal(job, candidate, weights["location"])
    sal_sig, salary_fit = salary_signal(job, candidate, weights["salary"])
    signals = [
        req_sig, pref_sig, exp_sig, role_signal(job, candidate, weights["role_alignment"]),
        semantic_signal(similarity, weights["semantic"], *semantic_range), loc_sig, sal_sig,
        employment_signal(job, candidate, weights["employment_type"]),
    ]
    known = [s for s in signals if s.value is not None and s.weight > 0]
    total_weight = sum(s.weight for s in signals if s.weight > 0) or 1
    known_weight = sum(s.weight for s in known)
    raw = sum(s.value * s.weight for s in known) / known_weight if known_weight else 0.0
    coverage = known_weight / total_weight
    score = round(100 * raw)

    gaps = []
    if missing:
        gaps.append(f"Required skills not in your profile: {', '.join(missing[:8])}")
    if unverified:
        gaps.append(f"Required skills listed but unverified: {', '.join(unverified[:8])} – verify them if accurate")
    if exp_gap:
        gaps.append(exp_gap)
    if p_missing:
        gaps.append(f"Nice-to-have skills you lack: {', '.join(p_missing[:6])}")
    if loc_sig.value is not None and loc_sig.value < 0.5:
        gaps.append(location_fit)
    if sal_sig.value is not None and sal_sig.value < 1:
        gaps.append(salary_fit)

    capped_reason = ""
    if req_sig.value is not None and req_sig.value < 0.5 and score >= thresholds["good"]:
        score = thresholds["good"] - 1
        capped_reason = "Capped below 'good' because fewer than half of the required skills are verified in your profile."

    excluded_reason = ""
    lowered_company = job.company.lower()
    if any(c.strip().lower() == lowered_company for c in candidate.excluded_companies if c.strip()):
        excluded_reason = f"{job.company} is in your excluded companies"
    else:
        haystack = f"{job.title}\n{job.description}".lower()
        hit = next((k for k in candidate.excluded_keywords if k.strip() and re.search(rf"\b{re.escape(k.strip().lower())}\b", haystack)), None)
        if hit:
            excluded_reason = f"Posting mentions excluded keyword '{hit}'"
    if excluded_reason:
        score = min(score, thresholds["possible"] - 1)

    category = categorize(score, thresholds)
    action = ACTIONS[category]
    if job.is_expired:
        action = "This posting has expired; it is kept for your records only."
    elif excluded_reason:
        action = f"Excluded by your preferences ({excluded_reason})."

    strongest = sorted((s for s in known if s.weight and s.value >= 0.7), key=lambda s: -s.value * s.weight)[:2]
    weakest = sorted((s for s in known if s.weight), key=lambda s: s.value)[:2]
    parts = [f"Overall {score}/100 ({category} match)."]
    if strongest:
        parts.append("Strengths: " + "; ".join(s.detail for s in strongest))
    if weakest and weakest[0].value < 0.7:
        parts.append("Weaknesses: " + "; ".join(s.detail for s in weakest if s.value < 0.7))
    if capped_reason:
        parts.append(capped_reason)
    unknown = [s.name.replace("_", " ") for s in signals if s.value is None and s.weight]
    if unknown:
        parts.append(f"Not scored (information unavailable): {', '.join(unknown)}.")
    if coverage < 0.6:
        parts.append("Limited information in the posting: treat this score as indicative only.")

    return MatchResult(
        score=score, category=category, signals={s.name: s.as_dict() for s in signals},
        skills_matched=list(dict.fromkeys(matched + p_matched)),
        skills_missing=list(dict.fromkeys(missing + p_missing)),
        skills_unverified=list(dict.fromkeys(unverified + p_unverified)),
        relevant_experience=relevant_experience(job, candidate), gaps=gaps, location_fit=location_fit,
        salary_fit=salary_fit, explanation=" ".join(parts), recommended_action=action, coverage=round(coverage, 3),
        excluded_reason=excluded_reason,
    )
