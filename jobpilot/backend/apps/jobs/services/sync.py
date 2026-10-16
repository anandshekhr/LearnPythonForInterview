"""Synchronise a job source: fetch -> normalise -> upsert -> dedupe -> expire.

The network fetch happens outside any database transaction; each job upsert is
its own short transaction so a failure mid-way leaves consistent data and the
next run resumes naturally (upserts are idempotent on (source, external_id)).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from django.db import transaction
from django.utils import timezone

from ..adapters.registry import adapter_for
from ..http import SourceHTTPError
from ..models import JobPosting, JobPostingChange, JobSource
from ..normalization import NormalizedJob, annualize, detect_seniority, job_fingerprint, normalize_title

logger = logging.getLogger(__name__)

TRACKED_FIELDS = ["title", "description_text", "location_raw", "employment_type", "department", "salary_min", "salary_max",
                  "apply_url", "work_model"]
MASS_REMOVAL_GUARD = 0.8  # refuse to expire >80% of a large board in one sync (likely an API glitch)


@dataclass
class SyncResult:
    fetched: int = 0
    created: list[int] = field(default_factory=list)
    updated: list[int] = field(default_factory=list)
    unchanged: int = 0
    expired: list[int] = field(default_factory=list)
    reactivated: list[int] = field(default_factory=list)
    skipped_invalid: int = 0
    warnings: list[str] = field(default_factory=list)

    def as_stats(self):
        return {
            "fetched": self.fetched, "created": len(self.created), "updated": len(self.updated), "unchanged": self.unchanged,
            "expired": len(self.expired), "reactivated": len(self.reactivated), "skipped_invalid": self.skipped_invalid,
            "warnings": self.warnings[:10],
        }


def _values(source: JobSource, nj: NormalizedJob) -> dict:
    salary = nj.salary
    return {
        "company": source.company,
        "canonical_url": nj.canonical_url[:600],
        "apply_url": (nj.apply_url or nj.canonical_url)[:600],
        "title": nj.title[:300],
        "normalized_title": normalize_title(nj.title)[:300],
        "department": nj.department[:200],
        "location_raw": nj.location_raw[:500],
        "locations": [loc[:120] for loc in nj.locations],
        "work_model": nj.work_model,
        "employment_type": nj.employment_type,
        "seniority": detect_seniority(nj.title),
        "description_text": nj.description_text,
        "salary_min": annualize(salary.minimum, salary.period),
        "salary_max": annualize(salary.maximum, salary.period),
        "salary_currency": salary.currency[:3],
        "salary_period": "year" if salary.minimum else "",
        "salary_text": salary.text[:200],
        "published_at": nj.published_at,
        "source_updated_at": nj.source_updated_at,
        "content_hash": nj.content_hash,
        "fingerprint": job_fingerprint(source.company_id, nj.title, nj.locations),
        "raw_metadata": nj.raw_metadata,
    }


def _find_duplicate(job: JobPosting):
    """Same posting published through another source (same canonical URL or fingerprint)."""
    qs = JobPosting.objects.filter(status=JobPosting.Status.ACTIVE, duplicate_of__isnull=True).exclude(pk=job.pk).exclude(source=job.source)
    return (qs.filter(canonical_url=job.canonical_url).first() if job.canonical_url else None) or qs.filter(
        fingerprint=job.fingerprint).order_by("first_seen_at").first()


def upsert_job(source: JobSource, nj: NormalizedJob, now, result: SyncResult):
    values = _values(source, nj)
    with transaction.atomic():
        job = JobPosting.objects.select_for_update().filter(source=source, external_id=nj.external_id).first()
        if job is None:
            job = JobPosting.objects.create(source=source, external_id=nj.external_id, last_seen_at=now, **values)
            dup = _find_duplicate(job)
            if dup:
                job.duplicate_of = dup
                job.save(update_fields=["duplicate_of"])
            result.created.append(job.pk)
            return
        changed = [f for f in TRACKED_FIELDS if getattr(job, f) != values[f]]
        was_expired = job.status == JobPosting.Status.EXPIRED
        previous_hash = job.content_hash
        for key, value in values.items():
            setattr(job, key, value)
        job.last_seen_at = now
        if was_expired:
            job.status, job.expired_at = JobPosting.Status.ACTIVE, None
            JobPostingChange.objects.create(job=job, change_type="reactivated")
            result.reactivated.append(job.pk)
        if previous_hash != values["content_hash"]:
            job.description_version += 1
            JobPostingChange.objects.create(job=job, change_type="updated", fields_changed=changed or ["content"],
                                            previous_content_hash=previous_hash)
            result.updated.append(job.pk)
        elif not was_expired:
            result.unchanged += 1
        job.save()


def expire_job(job: JobPosting, now):
    job.status, job.expired_at = JobPosting.Status.EXPIRED, now
    job.save(update_fields=["status", "expired_at", "updated_at"])
    JobPostingChange.objects.create(job=job, change_type="expired")


def sync_source(source: JobSource, http=None) -> SyncResult:
    result = SyncResult()
    now = timezone.now()
    adapter = adapter_for(source, http=http)
    try:
        listings = list(adapter.fetch_listings())  # network: no DB transaction held
    except SourceHTTPError as exc:
        record_failure(source, str(exc))
        raise
    finally:
        if http is None:
            adapter.http.close()

    seen_ids = set()
    for nj in listings:
        if not nj.external_id or not nj.title or not nj.canonical_url:
            result.skipped_invalid += 1
            continue
        if nj.external_id in seen_ids:
            continue
        seen_ids.add(nj.external_id)
        result.fetched += 1
        upsert_job(source, nj, now, result)

    active = JobPosting.objects.filter(source=source, status=JobPosting.Status.ACTIVE).exclude(external_id__in=seen_ids)
    missing = list(active)
    previously_active = JobPosting.objects.filter(source=source, status=JobPosting.Status.ACTIVE).count()
    if previously_active >= 5 and len(missing) / previously_active > MASS_REMOVAL_GUARD:
        result.warnings.append(f"Refused to expire {len(missing)} of {previously_active} jobs in one sync; will re-check next run.")
    else:
        for job in missing:
            expire_job(job, now)
            result.expired.append(job.pk)

    JobSource.objects.filter(pk=source.pk).update(
        status=JobSource.Status.ACTIVE, last_synced_at=now, last_success_at=now, last_error="", consecutive_failures=0,
        active_job_count=JobPosting.objects.filter(source=source, status=JobPosting.Status.ACTIVE).count(),
    )
    logger.info("source_synced", extra={"source_id": source.pk, "sync_stats": result.as_stats()})
    return result


def record_failure(source: JobSource, message: str):
    failures = source.consecutive_failures + 1
    JobSource.objects.filter(pk=source.pk).update(
        last_synced_at=timezone.now(), last_error=message[:1000], last_error_at=timezone.now(), consecutive_failures=failures,
        status=JobSource.Status.ERROR if failures >= 3 else JobSource.Status.DEGRADED,
    )
