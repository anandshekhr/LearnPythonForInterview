import hashlib

from django.core.files.base import ContentFile
from django.db import transaction
from django.utils import timezone

from apps.core import audit
from apps.profiles.models import ResumeVersion

from ..models import GeneratedDocument
from . import render
from .cover_letter import draft_cover_letter
from .tailoring import create_tailored_version


def _store(doc: GeneratedDocument, data: bytes, filename: str):
    doc.file.save(filename, ContentFile(data), save=False)
    doc.size_bytes = len(data)
    doc.sha256 = hashlib.sha256(data).hexdigest()


def render_resume_documents(version: ResumeVersion, template="classic", formats=("pdf", "docx")):
    docs = []
    for fmt in formats:
        data = render.resume_pdf(version.content, template) if fmt == "pdf" else render.resume_docx(version.content, template)
        doc = GeneratedDocument(user=version.user, job=version.job, resume_version=version, kind=GeneratedDocument.Kind.RESUME,
                                format=fmt, template=template, status=GeneratedDocument.Status.AWAITING_APPROVAL,
                                title=f"Resume – {version.label}"[:200], generator=version.generator,
                                warnings=version.tailoring_notes.get("warnings", []))
        _store(doc, data, f"resume.{fmt}")
        doc.save()
        docs.append(doc)
    return docs


def render_cover_letter(version: ResumeVersion, job, text: str, template="classic", warnings=None, generator="template",
                        doc: GeneratedDocument | None = None):
    data = render.letter_pdf(text, template)
    if doc is None:
        doc = GeneratedDocument(user=version.user, job=job, resume_version=version, kind=GeneratedDocument.Kind.COVER_LETTER,
                                format=GeneratedDocument.Format.PDF, template=template, title=f"Cover letter – {version.label}"[:200],
                                generator=generator, warnings=warnings or [])
    else:
        doc.file.delete(save=False)
    doc.text_content = text
    doc.status = GeneratedDocument.Status.AWAITING_APPROVAL
    _store(doc, data, "cover_letter.pdf")
    doc.save()
    return doc


def generate_for_job(user, job, include_cover_letter=True, template="classic", formats=("pdf", "docx")):
    version = create_tailored_version(user, job)
    docs = render_resume_documents(version, template, formats)
    if include_cover_letter:
        text, warnings, generator = draft_cover_letter(version, job)
        docs.append(render_cover_letter(version, job, text, template, warnings, generator))
    audit.record("documents.generated", user=user, obj=version, job_id=job.pk, documents=len(docs))
    from apps.applications.services.lifecycle import on_documents_generated

    on_documents_generated(user, job, version)
    return version, docs


@transaction.atomic
def approve_version(version: ResumeVersion, user):
    if version.status not in (ResumeVersion.Status.AWAITING_APPROVAL, ResumeVersion.Status.DRAFT):
        from apps.core.exceptions import DomainError

        raise DomainError(f"Version is {version.status}; only drafts awaiting approval can be approved.", code="invalid_state",
                          status_code=409)
    now = timezone.now()
    version.status, version.approved_at = ResumeVersion.Status.APPROVED, now
    version.save(update_fields=["status", "approved_at", "updated_at"])
    version.documents.filter(kind=GeneratedDocument.Kind.RESUME).update(status=GeneratedDocument.Status.APPROVED, approved_at=now)
    audit.record("resume_version.approved", user=user, obj=version)
    from apps.applications.services.lifecycle import on_version_approved

    on_version_approved(user, version)
    return version


@transaction.atomic
def reject_version(version: ResumeVersion, user):
    if version.status == ResumeVersion.Status.APPROVED and version.applications.exclude(submitted_at=None).exists():
        from apps.core.exceptions import DomainError

        raise DomainError("This version was used in a submitted application and cannot be rejected.", code="in_use", status_code=409)
    version.status = ResumeVersion.Status.REJECTED
    version.save(update_fields=["status", "updated_at"])
    version.documents.update(status=GeneratedDocument.Status.REJECTED)
    audit.record("resume_version.rejected", user=user, obj=version)
    return version
