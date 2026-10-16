"""Run on YOUR machine:  python manage.py assist_apply <application_id>

Opens the official application form in a visible browser, pre-fills the fields
JobPilot can identify, uploads your approved resume and then hands control to you.
It never presses submit. Complete any CAPTCHA, review every field, submit yourself,
then close the browser window and confirm here.
"""
from django.core.management.base import BaseCommand, CommandError

from apps.applications.browser.filler import AssistNotPermitted
from apps.applications.browser.runner import run_assist_for_application
from apps.applications.models import Application, SubmissionChannel
from apps.applications.services.submission import confirm_manual_submission


class Command(BaseCommand):
    help = "Pre-fill an official application form in a visible browser (never submits)."

    def add_arguments(self, parser):
        parser.add_argument("application_id", type=int)

    def handle(self, application_id, **options):
        from django.conf import settings

        settings.BROWSER_AUTOMATION_ENABLED = True

        def wait_for_user(page, report):
            self.stdout.write(self.style.SUCCESS(f"Filled: {', '.join(report.filled) or 'nothing'}"))
            if report.not_found:
                self.stdout.write(self.style.WARNING(f"Not found on page (fill manually): {', '.join(report.not_found)}"))
            for item in report.skipped:
                self.stdout.write(self.style.WARNING(f"Skipped: {item}"))
            if report.blocked_reason:
                self.stdout.write(self.style.WARNING(report.blocked_reason))
            self.stdout.write("Review the form, submit it yourself, then close the browser window.")
            page.wait_for_event("close", timeout=0)

        try:
            run_assist_for_application(application_id, headless=False, keep_open=True, wait_for_user=wait_for_user)
        except AssistNotPermitted as exc:
            raise CommandError(str(exc))
        answer = input("Did you submit the application? Enter the confirmation reference, 'y', or leave empty for no: ").strip()
        if answer:
            app = Application.objects.get(pk=application_id)
            confirm_manual_submission(app, app.user, SubmissionChannel.BROWSER_ASSISTED, "" if answer.lower() == "y" else answer)
            self.stdout.write(self.style.SUCCESS("Recorded as submitted."))
        else:
            self.stdout.write("Not marked as submitted.")
