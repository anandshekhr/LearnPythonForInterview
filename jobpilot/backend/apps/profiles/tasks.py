import logging

from celery import shared_task

from apps.automation.runs import track_run

from .models import Resume
from .services import parser
from .services.validation import ALLOWED

logger = logging.getLogger(__name__)


@shared_task(bind=True, autoretry_for=(OSError,), retry_backoff=True, retry_kwargs={"max_retries": 3}, acks_late=True)
def parse_resume(self, resume_id: int):
    resume = Resume.objects.filter(pk=resume_id).first()
    if resume is None:
        return {"status": "missing"}
    with track_run("resume_parse", user=resume.user, task_id=self.request.id, target=resume) as run:
        Resume.objects.filter(pk=resume.pk).update(parse_status=Resume.ParseStatus.PARSING)
        try:
            kind = next(k for k, ct in ALLOWED.items() if ct == resume.content_type)
            with resume.file.open("rb") as fh:
                data = fh.read()
            text = parser.extract_text(kind, data)
            if len(text) < 100:
                raise ValueError("Very little text could be extracted (is the file a scanned image?)")
            parsed = parser.parse_resume_text(text)
        except Exception as exc:
            logger.warning("resume_parse_failed", extra={"resume_id": resume.pk, "error": type(exc).__name__})
            Resume.objects.filter(pk=resume.pk).update(
                parse_status=Resume.ParseStatus.FAILED, parse_error=str(exc)[:500] if isinstance(exc, ValueError) else "Parsing failed."
            )
            run.fail(type(exc).__name__)
            return {"status": "failed"}
        Resume.objects.filter(pk=resume.pk).update(
            raw_text=text, parsed_data=parsed, parser=parsed["parser"], parse_status=Resume.ParseStatus.PARSED, parse_error=""
        )
        run.stats = {"employment": len(parsed["employment"]), "skills": len(parsed["skills"])}
        return {"status": "parsed"}
