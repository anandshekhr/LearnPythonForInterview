"""Workday career sites (`https://{tenant}.wd{N}.myworkdayjobs.com/{site}`).

Workday publishes no public jobs API. Every Workday career site loads its listings from a
JSON endpoint under `/wday/cxs/{tenant}/{site}/` that the employer's own public page calls
anonymously. This adapter reads that endpoint in the same way, read-only, with conservative
throttling. It obeys robots.txt and stops on HTTP 403: some employers refuse automated
access, and JobPilot never works around that. The endpoint is undocumented and may change, so
parsing is defensive and failures surface as source errors instead of bad data.
"""
from __future__ import annotations

import re
from datetime import date, datetime, timedelta, timezone
from urllib.parse import urlsplit

from django.utils import timezone as dj_timezone

from apps.core.text import html_to_text

from ..crawl import RobotsPolicy, url_id
from ..http import SourceHTTPError
from ..normalization import NormalizedJob, detect_work_model, normalize_employment_type
from .base import AdapterConfigError, JobSourceAdapter, SourceCapabilities

HOST_RE = re.compile(r"^(?P<tenant>[a-z0-9][a-z0-9\-]*)\.(?P<pod>wd\d{1,3})\.myworkdayjobs\.com$", re.I)
LOCALE_RE = re.compile(r"^[a-z]{2}-[A-Z]{2}$")
PAGE = 20  # Workday career sites page 20 jobs at a time
POSTED_RE = re.compile(r"(\d+)\+?\s+days?\s+ago", re.I)


class WorkdayAdapter(JobSourceAdapter):
    source_type = "workday"
    allowed_config = {"search_text": str, "max_jobs": int, "detail_refresh_days": int}
    public_only = True
    min_request_interval = 1.5
    capabilities = SourceCapabilities(
        salary_data=False,
        application_questions=False,
        api_submission=False,
        browser_assist=False,
        documentation_url="",
        limitations=(
            "Workday has no documented public jobs API; this reads the JSON endpoint behind the employer's public career site.",
            "Some employers block automated access (HTTP 403); JobPilot then stops instead of working around it.",
            "Applying on Workday needs a candidate account on the employer's site, so applications are always made there.",
            "Large employers list thousands of jobs: use search keywords and a job cap to keep syncs small.",
        ),
    )

    @classmethod
    def validate_identifier(cls, identifier: str) -> str:
        value = (identifier or "").strip()
        if "://" not in value:
            value = "https://" + value
        parts = urlsplit(value)
        host = (parts.hostname or "").lower()
        if not HOST_RE.match(host):
            raise AdapterConfigError("Use the career site URL, e.g. https://acme.wd5.myworkdayjobs.com/AcmeCareers")
        segments = [s for s in parts.path.split("/") if s]
        if segments and LOCALE_RE.match(segments[0]):
            segments = segments[1:]
        if not segments or not re.fullmatch(r"[A-Za-z0-9_\-]{1,100}", segments[0]):
            raise AdapterConfigError("The URL must include the career site name, e.g. …myworkdayjobs.com/AcmeCareers")
        return f"{host}/{segments[0]}"

    @classmethod
    def validate_config(cls, config: dict) -> dict:
        config = super().validate_config(config)
        if "max_jobs" in config and not 1 <= config["max_jobs"] <= 2000:
            raise AdapterConfigError("max_jobs must be between 1 and 2000.")
        if len(config.get("search_text") or "") > 100:
            raise AdapterConfigError("search_text is limited to 100 characters.")
        return config

    @property
    def host(self) -> str:
        return self.identifier.split("/", 1)[0]

    @property
    def site(self) -> str:
        return self.identifier.split("/", 1)[1]

    @property
    def tenant(self) -> str:
        return HOST_RE.match(self.host).group("tenant")

    @property
    def api(self) -> str:
        return f"https://{self.host}/wday/cxs/{self.tenant}/{self.site}"

    def public_url(self, external_path: str) -> str:
        return f"https://{self.host}/{self.site}{external_path}"

    def _guard(self, url):
        if not getattr(self, "_robots", None):
            self._robots = RobotsPolicy(self.http)
        if not self._robots.allowed(url):
            raise SourceHTTPError(f"robots.txt of {self.host} does not allow automated access to {urlsplit(url).path}")

    def _call(self, method, url, **kwargs):
        try:
            return self.http.request(method, url, **kwargs)
        except SourceHTTPError as exc:
            if exc.status_code == 403:
                raise SourceHTTPError("The employer's Workday site refused automated access (HTTP 403). "
                                      "JobPilot does not work around this; disable the source or apply on the site directly.",
                                      status_code=403) from exc
            raise

    def _list(self):
        max_jobs = int(self.config.get("max_jobs") or 300)
        offset, total, postings = 0, None, []
        self._guard(f"{self.api}/jobs")
        while True:
            body = {"appliedFacets": {}, "limit": PAGE, "offset": offset, "searchText": self.config.get("search_text") or ""}
            data = self._call("POST", f"{self.api}/jobs", json=body).json()
            page = data.get("jobPostings") or []
            if total is None:
                total = int(data.get("total") or 0)
            postings.extend(p for p in page if p.get("externalPath") and p.get("title"))
            offset += PAGE
            if not page or offset >= total or len(postings) >= max_jobs:
                break
        postings = postings[:max_jobs]
        if total and len(postings) < total:
            self.listing_complete = False
            self.warnings.append(f"Workday lists {total} jobs; fetched the first {len(postings)} (raise max_jobs or narrow search_text).")
        return postings

    @staticmethod
    def external_id(posting: dict) -> str:
        path = posting.get("externalPath") or ""
        tail = path.rsplit("_", 1)[-1] if "_" in path else ""
        if tail and re.fullmatch(r"[A-Za-z0-9\-]{2,40}", tail):
            return tail
        return url_id(path)

    def fetch_listings(self):
        refresh = timedelta(days=int(self.config.get("detail_refresh_days") or 7))
        now = dj_timezone.now()
        for posting in self._list():
            ext_id = self.external_id(posting)
            url = self.public_url(posting["externalPath"])
            stored = self.known_jobs.get(ext_id)
            if stored and now - stored < refresh:
                yield NormalizedJob(external_id=ext_id, title=posting["title"], canonical_url=url, partial=True)
                continue
            detail = self._call("GET", f"{self.api}{posting['externalPath']}").json()
            job = self._job(detail, posting, ext_id)
            if job:
                yield job

    def fetch_job(self, external_id: str):
        return None  # details are fetched by path during listing

    def confirm_removed(self, job) -> bool:
        path = urlsplit(job.canonical_url).path.split(f"/{self.site}", 1)[-1]
        try:
            self._call("GET", f"{self.api}{path}")
        except SourceHTTPError as exc:
            return exc.status_code in (404, 410)
        return False

    def _job(self, detail: dict, posting: dict, ext_id: str) -> NormalizedJob | None:
        info = detail.get("jobPostingInfo") or {}
        if not info.get("title"):
            return None
        locations = [x for x in [info.get("location"), *(info.get("additionalLocations") or [])] if x]
        if not locations and posting.get("locationsText"):
            locations = [posting["locationsText"]]
        remote_hint = str(info.get("remoteType") or "").lower().replace("-", "")
        work_model = {"remote": "remote", "hybrid": "hybrid", "onsite": "onsite", "on site": "onsite"}.get(remote_hint, "unknown")
        description = html_to_text(info.get("jobDescription") or "")
        url = info.get("externalUrl") or self.public_url(posting["externalPath"])
        return NormalizedJob(
            external_id=ext_id,
            title=info["title"],
            canonical_url=url,
            apply_url=url,
            description_text=description,
            location_raw=" / ".join(locations),
            locations=locations,
            work_model=work_model if work_model != "unknown" else detect_work_model(" ".join(locations), description),
            employment_type=normalize_employment_type(str(info.get("timeType") or ""), info["title"]),
            published_at=_posted(info.get("startDate"), info.get("postedOn") or posting.get("postedOn")),
            raw_metadata={"job_req_id": info.get("jobReqId"), "hiring_organization": (detail.get("hiringOrganization") or {}).get("name")},
        ).finalize()


def _posted(start_date, posted_on):
    if start_date:
        try:
            d = date.fromisoformat(str(start_date)[:10])
            return datetime(d.year, d.month, d.day, tzinfo=timezone.utc)
        except ValueError:
            pass
    text = str(posted_on or "").lower()
    today = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    if "today" in text:
        return today
    if "yesterday" in text:
        return today - timedelta(days=1)
    m = POSTED_RE.search(text)
    return today - timedelta(days=int(m.group(1))) if m else None
