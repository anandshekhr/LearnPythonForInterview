from celery import shared_task
from django.contrib.auth import get_user_model

from apps.automation.runs import track_run
from apps.jobs.models import JobPosting

from .services.generation import generate_for_job
from .services.tailoring import TailoringError


@shared_task(bind=True, acks_late=True)
def generate_documents(self, user_id: int, job_id: int, include_cover_letter=True, template="classic"):
    user = get_user_model().objects.filter(pk=user_id).first()
    job = JobPosting.objects.select_related("company").filter(pk=job_id).first()
    if user is None or job is None:
        return {"status": "missing"}
    with track_run("document_generation", user=user, task_id=self.request.id, target=job) as run:
        try:
            version, docs = generate_for_job(user, job, include_cover_letter, template)
        except TailoringError as exc:
            run.fail(str(exc))
            return {"status": "failed", "error": str(exc)}
        run.stats = {"resume_version_id": version.pk, "documents": [d.pk for d in docs]}
        return {"status": "ok", "resume_version_id": version.pk}
