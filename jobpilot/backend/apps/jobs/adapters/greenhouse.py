"""Greenhouse Job Board API (https://developers.greenhouse.io/job-board.html).

Listing published jobs is public and needs no credentials. Submitting an
application (POST /v1/boards/{token}/jobs/{id}) requires a Job Board API key that the
employer creates in Greenhouse, sent as HTTP Basic auth (key as username).
"""
from __future__ import annotations

import base64
import re
from datetime import datetime

from apps.core.text import html_to_text

from ..normalization import NormalizedJob, Salary, normalize_employment_type
from .base import ApplicationQuestion, JobSourceAdapter, SourceCapabilities, SubmissionResult, SubmissionNotSupported

DEFAULT_BASE = "https://boards-api.greenhouse.io/v1/boards"


def base_url():
    from django.conf import settings

    return settings.SOURCE_API_BASE_OVERRIDES.get("greenhouse") or DEFAULT_BASE
SENSITIVE_RE = re.compile(
    r"(gender|race|ethnic|veteran|disab|sponsor|visa|authori[sz]ed to work|work permit|legal|criminal|convict|"
    r"pronoun|sexual|demographic|eeo|age|date of birth|religion|consent|declar|terms|privacy|relocat)",
    re.I,
)


def _dt(value):
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


class GreenhouseAdapter(JobSourceAdapter):
    source_type = "greenhouse"
    capabilities = SourceCapabilities(
        salary_data=True,
        application_questions=True,
        api_submission=True,
        api_submission_requires="Job Board API key issued by the employer (Greenhouse Dev Center > API Credential Management)",
        browser_assist=True,
        documentation_url="https://developers.greenhouse.io/job-board.html",
        limitations=(
            "Only published jobs are visible.",
            "Application submission requires the employer's Job Board API key; without it, use the hosted application form.",
            "Some boards protect the hosted form with CAPTCHA; JobPilot never bypasses it.",
        ),
    )

    def _job(self, item: dict) -> NormalizedJob:
        metadata = {m.get("name"): m.get("value") for m in item.get("metadata") or [] if m.get("name")}
        offices = [o.get("name") for o in item.get("offices") or [] if o.get("name")]
        location_raw = (item.get("location") or {}).get("name") or ", ".join(offices)
        employment_raw = next((str(v) for k, v in metadata.items() if k and "employment" in k.lower() and v), "")
        salary = Salary()
        for rng in item.get("pay_input_ranges") or []:
            if rng.get("min_cents") and rng.get("max_cents"):
                salary = Salary(int(rng["min_cents"]) // 100, int(rng["max_cents"]) // 100, rng.get("currency_type", ""),
                                "year", rng.get("title", "")[:200])
                break
        url = item.get("absolute_url") or ""
        return NormalizedJob(
            external_id=str(item["id"]),
            title=item.get("title") or "",
            canonical_url=url,
            apply_url=url,
            description_text=html_to_text(item.get("content") or ""),
            department=", ".join(d.get("name", "") for d in item.get("departments") or [] if d.get("name")),
            location_raw=location_raw,
            employment_type=normalize_employment_type(employment_raw, item.get("title", "")),
            salary=salary,
            published_at=_dt(item.get("first_published")) or _dt(item.get("updated_at")),
            source_updated_at=_dt(item.get("updated_at")),
            raw_metadata={"internal_job_id": item.get("internal_job_id"), "requisition_id": item.get("requisition_id"),
                          "offices": offices},
        ).finalize()

    def fetch_listings(self):
        data = self.http.get_json(f"{base_url()}/{self.identifier}/jobs", params={"content": "true", "pay_transparency": "true"})
        for item in data.get("jobs", []):
            if item.get("id") and item.get("title"):
                yield self._job(item)

    def fetch_job(self, external_id: str):
        from ..http import SourceHTTPError

        try:
            item = self.http.get_json(f"{base_url()}/{self.identifier}/jobs/{external_id}", params={"pay_transparency": "true"})
        except SourceHTTPError as exc:
            if exc.status_code == 404:
                return None
            raise
        return self._job(item)

    def fetch_application_questions(self, external_id: str) -> list[ApplicationQuestion]:
        item = self.http.get_json(f"{base_url()}/{self.identifier}/jobs/{external_id}", params={"questions": "true"})
        questions = []
        groups = [("questions", False), ("location_questions", False), ("compliance", True), ("demographic_questions", True)]
        for key, always_sensitive in groups:
            block = item.get(key) or []
            if isinstance(block, dict):
                block = block.get("questions") or []
            flat = []
            for entry in block:  # compliance groups nest their questions one level deeper
                flat.extend(entry["questions"] if isinstance(entry.get("questions"), list) else [entry])
            for q in flat:
                for f in q.get("fields") or []:
                    name = f.get("name")
                    if not name:
                        continue
                    ftype = {"input_text": "text", "textarea": "textarea", "input_file": "file",
                             "multi_value_single_select": "select", "multi_value_multi_select": "multi_select"}.get(f.get("type"), "text")
                    label = q.get("label") or name
                    questions.append(ApplicationQuestion(
                        key=name, label=label, field_type=ftype, required=bool(q.get("required")),
                        options=[{"label": v.get("label"), "value": v.get("value")} for v in f.get("values") or []],
                        sensitive=always_sensitive or bool(SENSITIVE_RE.search(label)),
                        description=html_to_text(q.get("description") or "")[:500],
                    ))
        return questions

    def submit_application(self, external_id: str, payload: dict) -> SubmissionResult:
        if not self.api_key:
            raise SubmissionNotSupported("No Greenhouse Job Board API key configured for this board")
        body = {
            "first_name": payload["first_name"], "last_name": payload["last_name"], "email": payload["email"],
            "phone": payload.get("phone", ""),
            "resume_content": base64.b64encode(payload["resume_bytes"]).decode(),
            "resume_content_filename": payload["resume_filename"],
        }
        if payload.get("cover_letter_text"):
            body["cover_letter_text"] = payload["cover_letter_text"]
        for key, value in (payload.get("answers") or {}).items():
            body[key] = value
        auth = base64.b64encode(f"{self.api_key}:".encode()).decode()
        response = self.http.request("POST", f"{base_url()}/{self.identifier}/jobs/{external_id}", json=body,
                                     headers={"Authorization": f"Basic {auth}"})
        try:
            data = response.json()
        except ValueError:
            data = {}
        confirmed = response.status_code == 200 and bool(data.get("success"))
        return SubmissionResult(
            confirmed=confirmed, reference=str(data.get("application_id") or data.get("id") or ""),
            status_code=response.status_code, message=str(data.get("success") or data.get("error") or "")[:300],
            response_excerpt={k: data[k] for k in list(data)[:5] if isinstance(data.get(k), (str, int, bool))},
        )
