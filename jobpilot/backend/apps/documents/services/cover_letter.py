"""Role-specific cover letters built only from verified evidence."""
from __future__ import annotations

from apps.core import llm
from apps.core.text import normalize_token

from .tailoring import check_claims


def _evidence_sentences(version) -> list[str]:
    seen, out = set(), []
    for row in version.tailoring_notes.get("requirement_map", []):
        if row["status"] != "matched":
            continue
        for ev in row["evidence"]:
            if ev["type"] == "achievement" and ev["text"] not in seen:
                seen.add(ev["text"])
                out.append(f"As {ev['label']}, {ev['text'][0].lower()}{ev['text'][1:].rstrip('.')}.")
                break
        if len(out) >= 3:
            break
    return out


def draft_cover_letter(version, job) -> tuple[str, list[str], str]:
    content = version.content
    contact = content.get("contact", {})
    name = contact.get("full_name") or ""
    matched = [r["requirement"] for r in version.tailoring_notes.get("requirement_map", []) if r["status"] == "matched"]
    years = content.get("total_experience_years") or 0
    current = next((e for e in content.get("employment", []) if e.get("is_current")), (content.get("employment") or [{}])[0])
    intro = (f"I am writing to apply for the {job.title} position at {job.company.name}. "
             f"I have {years:g} years of professional experience" + (f", currently as {current['title']} at {current['company']}" if current.get("title") else "")
             + ".")
    skills = f"My verified experience covers {', '.join(matched[:6])}, which the role lists as important." if matched else ""
    evidence = _evidence_sentences(version)
    body = [f"Dear Hiring Team at {job.company.name},", intro + (" " + skills if skills else "")]
    if evidence:
        body.append("Relevant examples from my work:\n" + "\n".join(f"• {e}" for e in evidence))
    body.append(f"I would welcome the opportunity to discuss how I can contribute to {job.company.name}. Thank you for your time and consideration.")
    body.append(f"Sincerely,\n{name}")
    text = "\n\n".join(body)
    warnings: list[str] = []
    generator = "template"

    if llm.is_enabled():
        try:
            result = llm.generate_json(
                "cover_letter",
                "Polish this cover letter draft for clarity and tone. Keep it under 250 words. Use only the facts in the "
                "draft: do not add numbers, technologies, employers, titles or achievements that are not already present.",
                {"type": "object", "additionalProperties": False, "required": ["letter"], "properties": {"letter": {"type": "string"}}},
                trusted_context={"draft": text}, untrusted_blocks={"job_title": job.title},
            )
            polished = result.data.get("letter", "").strip()
            entities = {normalize_token(x) for x in [job.company.name, name, job.title]}
            problems = check_claims(polished, text, set(content.get("skills", [])), entities)
            if polished and not problems:
                text, generator = polished, f"template+{result.provider}"
            elif problems:
                warnings.append(f"AI polish discarded ({'; '.join(problems)}); using the factual template.")
        except llm.LLMUnavailable:
            warnings.append("AI polish unavailable; using the factual template.")
    return text, warnings, generator
