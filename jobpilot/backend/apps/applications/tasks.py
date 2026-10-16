from celery import shared_task

from apps.automation.runs import track_run

from .models import Application
from .services.submission import perform_submission


@shared_task(bind=True, acks_late=True)
def process_submission(self, application_id: int):
    """Network call for an automated submission. Not auto-retried: an ambiguous failure must be
    checked by the user before another attempt (see submission.py)."""
    app = Application.objects.select_related("user").filter(pk=application_id).first()
    if app is None:
        return None
    with track_run("application_submission", user=app.user, task_id=self.request.id, target=app) as run:
        app = perform_submission(app.pk)
        run.stats = {"status": app.status, "reference": app.submission_reference}
        if app.status != "submitted":
            run.fail(app.last_error)
        return app.status


@shared_task(bind=True, acks_late=True)
def prepare_answers_task(self, application_id: int):
    from .services.answers import prepare_answers

    app = Application.objects.select_related("job", "job__source", "user").filter(pk=application_id).first()
    if app is None:
        return 0
    with track_run("prepare_answers", user=app.user, task_id=self.request.id, target=app) as run:
        answers = prepare_answers(app)
        run.stats = {"answers": len(answers)}
        return len(answers)


@shared_task(bind=True, acks_late=True)
def run_browser_assist(self, application_id: int):
    """Placeholder routed to the optional 'browser' queue; see apps/applications/browser/."""
    from .browser.runner import run_assist_for_application

    return run_assist_for_application(application_id)
