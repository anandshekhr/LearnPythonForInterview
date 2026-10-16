"""Pluggable job-source adapter interface.

To add a source: subclass JobSourceAdapter, implement the abstract methods, set
`source_type`, and register it in `registry.py`. Sync, dedup, expiry, matching
and the UI then work without further changes.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Iterator

from ..http import SourceHttpClient
from ..normalization import NormalizedJob


@dataclass(frozen=True)
class SourceCapabilities:
    list_jobs: bool = True
    job_details: bool = True
    salary_data: bool = False
    application_questions: bool = False
    # Requires credentials issued by the employer; never available by default.
    api_submission: bool = False
    api_submission_requires: str = ""
    browser_assist: bool = False
    documentation_url: str = ""
    limitations: tuple[str, ...] = ()

    def as_dict(self):
        d = asdict(self)
        d["limitations"] = list(self.limitations)
        return d


@dataclass
class ApplicationQuestion:
    key: str
    label: str
    field_type: str  # text, textarea, file, select, multi_select, boolean
    required: bool = False
    options: list[dict] = field(default_factory=list)
    # Sensitive questions (legal, demographic, disability, sponsorship) are never auto-answered.
    sensitive: bool = False
    description: str = ""


@dataclass
class SubmissionResult:
    confirmed: bool
    reference: str = ""
    status_code: int | None = None
    message: str = ""
    response_excerpt: dict = field(default_factory=dict)


class AdapterConfigError(ValueError):
    pass


class SubmissionNotSupported(Exception):
    pass


class JobSourceAdapter(ABC):
    source_type: str = ""
    capabilities = SourceCapabilities()

    def __init__(self, identifier: str, config: dict | None = None, http: SourceHttpClient | None = None,
                 api_key: str = ""):
        self.identifier = identifier
        self.config = config or {}
        self.http = http or SourceHttpClient(throttle_key=f"{self.source_type}:{identifier}")
        self.api_key = api_key

    @classmethod
    def validate_identifier(cls, identifier: str) -> str:
        import re

        value = (identifier or "").strip()
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.\-]{0,119}", value):
            raise AdapterConfigError("Identifier may only contain letters, digits, '.', '_' and '-'.")
        return value

    @abstractmethod
    def fetch_listings(self) -> Iterator[NormalizedJob]:
        """Yield every currently published job (the complete set, used to detect removals)."""

    @abstractmethod
    def fetch_job(self, external_id: str) -> NormalizedJob | None:
        """Fetch one job's details, or None if it is no longer published."""

    def application_url(self, job) -> str:
        return job.apply_url or job.canonical_url

    def fetch_application_questions(self, external_id: str) -> list[ApplicationQuestion]:
        return []

    def submit_application(self, external_id: str, payload: dict) -> SubmissionResult:
        raise SubmissionNotSupported(f"{self.source_type} does not support API submission")
