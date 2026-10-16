from django.db.models import Prefetch
from django.utils import timezone
from drf_spectacular.utils import OpenApiParameter, extend_schema
from rest_framework import mixins, status, viewsets
from rest_framework.decorators import action
from rest_framework.response import Response

from apps.automation.services import get_settings
from apps.core import audit
from apps.core.exceptions import DomainError
from apps.documents.models import GeneratedDocument
from apps.jobs.models import JobPosting
from apps.profiles.models import ResumeVersion

from . import tasks
from .filters import ApplicationFilter, InterviewFilter
from .models import Application, ApplicationAnswer, ApplicationStatus, FollowUp, Interview
from .serializers import (
    AnswerUpdateSerializer,
    ApplicationAnswerSerializer,
    ApplicationCreateSerializer,
    ApplicationDetailSerializer,
    ApplicationListSerializer,
    AttachResumeSerializer,
    ConfirmSubmittedSerializer,
    FollowUpSerializer,
    InterviewSerializer,
    StatusChangeSerializer,
    SubmitSerializer,
)
from .services import lifecycle, submission
from .services.answers import readiness_problems

S = ApplicationStatus


class ApplicationViewSet(mixins.ListModelMixin, mixins.RetrieveModelMixin, mixins.UpdateModelMixin, mixins.DestroyModelMixin,
                         viewsets.GenericViewSet):
    filterset_class = ApplicationFilter
    search_fields = ["company_name", "job_title", "notes"]
    ordering_fields = ["updated_at", "submitted_at", "match_score", "company_name", "created_at"]
    http_method_names = ["get", "post", "patch", "delete", "head", "options"]

    def get_queryset(self):
        if getattr(self, "swagger_fake_view", False):  # schema generation
            return Application.objects.none()
        qs = Application.objects.filter(user=self.request.user).select_related("resume_version")
        upcoming = Interview.objects.filter(scheduled_at__gte=timezone.now(), outcome=Interview.Outcome.PENDING)
        qs = qs.prefetch_related(Prefetch("interviews", queryset=upcoming, to_attr="upcoming_interviews"))
        if self.action == "retrieve":
            qs = qs.prefetch_related("history", "answers", "interviews", "follow_ups")
        return qs

    def get_serializer_class(self):
        if self.action == "list":
            return ApplicationListSerializer
        if self.action == "create":
            return ApplicationCreateSerializer
        return ApplicationDetailSerializer

    def get_throttles(self):
        self.throttle_scope = "submit" if self.action == "submit" else None
        return super().get_throttles()

    def _detail(self, app):
        return Response(ApplicationDetailSerializer(self.get_queryset().get(pk=app.pk), context={"request": self.request}).data)

    @extend_schema(request=ApplicationCreateSerializer, responses={201: ApplicationDetailSerializer})
    def create(self, request):
        """Track a job (shortlist it). Returns 409 with the existing application id if one already exists."""
        serializer = ApplicationCreateSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        job = JobPosting.objects.select_related("company", "source").filter(pk=serializer.validated_data["job"]).first()
        if job is None:
            raise DomainError("Job not found.", code="not_found", status_code=404)
        app = lifecycle.create_application(request.user, job, serializer.validated_data["status"])
        response = self._detail(app)
        response.status_code = status.HTTP_201_CREATED
        return response

    def perform_update(self, serializer):
        serializer.save()
        audit.record("application.updated", obj=serializer.instance, request=self.request, fields=sorted(serializer.validated_data))

    def perform_destroy(self, instance):
        if instance.is_submitted or instance.status == S.SUBMITTING:
            raise DomainError("Submitted applications cannot be deleted; withdraw or close them instead.", code="submitted",
                              status_code=409)
        audit.record("application.deleted", obj=instance, request=self.request)
        instance.delete()

    @extend_schema(request=StatusChangeSerializer, responses={200: ApplicationDetailSerializer})
    @action(detail=True, methods=["post"], url_path="status")
    def change_status(self, request, pk=None):
        """Manually update the status (e.g. interview scheduled, rejected, offer received)."""
        app = self.get_object()
        serializer = StatusChangeSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        to = serializer.validated_data["status"]
        if to == S.SUBMITTED:
            raise DomainError("Use confirm-submitted to record a submission.", code="use_confirm_submitted")
        app = lifecycle.transition(app, to, user=request.user, note=serializer.validated_data.get("note", ""))
        return self._detail(app)

    @extend_schema(request=AttachResumeSerializer, responses={200: ApplicationDetailSerializer})
    @action(detail=True, methods=["post"], url_path="attach-resume")
    def attach_resume(self, request, pk=None):
        """Use an approved resume version (and optionally an approved cover letter). Moves the application to ready_to_apply."""
        app = self.get_object()
        serializer = AttachResumeSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        version = ResumeVersion.objects.filter(pk=serializer.validated_data["resume_version"], user=request.user).first()
        if version is None:
            raise DomainError("Resume version not found.", code="not_found", status_code=404)
        cover = None
        if serializer.validated_data.get("cover_letter"):
            cover = GeneratedDocument.objects.filter(pk=serializer.validated_data["cover_letter"], user=request.user,
                                                     kind=GeneratedDocument.Kind.COVER_LETTER).first()
            if cover is None:
                raise DomainError("Cover letter not found.", code="not_found", status_code=404)
        if version.kind == ResumeVersion.Kind.MASTER and not version.documents.exists():
            from apps.documents.services.generation import render_resume_documents

            for doc in render_resume_documents(version):
                doc.status, doc.approved_at = GeneratedDocument.Status.APPROVED, timezone.now()
                doc.save(update_fields=["status", "approved_at"])
        app = lifecycle.attach_resume(app, version, request.user, cover)
        return self._detail(app)

    @extend_schema(request=None, responses={202: dict})
    @action(detail=True, methods=["post"], url_path="prepare-answers")
    def prepare_answers(self, request, pk=None):
        """Fetch the application questions (where the source exposes them) and pre-fill profile answers."""
        app = self.get_object()
        tasks.prepare_answers_task.delay(app.pk)
        return Response({"queued": True}, status=status.HTTP_202_ACCEPTED)

    @extend_schema(request=AnswerUpdateSerializer(many=True), responses={200: ApplicationAnswerSerializer(many=True)})
    @action(detail=True, methods=["patch"])
    def answers(self, request, pk=None):
        """Update answers. Sensitive answers count as reviewed only when `reviewed` is true."""
        app = self.get_object()
        if app.is_submitted:
            raise DomainError("Answers cannot be changed after submission.", code="already_submitted", status_code=409)
        serializer = AnswerUpdateSerializer(data=request.data, many=True)
        serializer.is_valid(raise_exception=True)
        for item in serializer.validated_data:
            answer = ApplicationAnswer.objects.filter(application=app, pk=item["id"]).first()
            if answer is None:
                raise DomainError(f"Answer {item['id']} not found.", code="not_found", status_code=404)
            answer.answer = item["answer"]
            answer.origin = ApplicationAnswer.Origin.USER
            answer.reviewed_at = timezone.now() if (item["reviewed"] or not answer.sensitive) else None
            answer.save()
        audit.record("application.answers_updated", obj=app, request=request, count=len(serializer.validated_data))
        return Response(ApplicationAnswerSerializer(app.answers.all(), many=True).data)

    @extend_schema(responses={200: dict})
    @action(detail=True, methods=["get"])
    def kit(self, request, pk=None):
        """Everything needed to apply: official URL, approved documents, answers, readiness and available modes."""
        app = self.get_object()
        settings_ = get_settings(request.user)
        api_ok, api_reason = submission.api_submission_available(app) if app.job else (False, "Job no longer exists")
        return Response({
            "application_id": app.pk, "status": app.status, "official_url": app.job_url,
            "job_active": bool(app.job and app.job.status == JobPosting.Status.ACTIVE),
            "resume_document": app.resume_document_id, "cover_letter": app.cover_letter_id,
            "answers": ApplicationAnswerSerializer(app.answers.all(), many=True).data,
            "problems": readiness_problems(app) + ([] if app.resume_version_id else ["No approved resume attached yet."]),
            "mode": settings_.application_mode,
            "automated_submission": {"available": api_ok and settings_.automated_submission_allowed,
                                     "source_supports_api": api_ok, "reason": api_reason,
                                     "consent_given": bool(settings_.automated_submission_consent_at)},
            "browser_assist_enabled": bool(app.job and app.job.source.source_type in ("greenhouse", "lever")),
        })

    @extend_schema(request=SubmitSerializer, responses={202: ApplicationDetailSerializer},
                   parameters=[OpenApiParameter("Idempotency-Key", str, OpenApiParameter.HEADER, required=True)])
    @action(detail=True, methods=["post"])
    def submit(self, request, pk=None):
        """Mode 3: submit through the source's authorized API. Requires an Idempotency-Key header;
        repeating the same key never submits twice."""
        app = self.get_object()
        serializer = SubmitSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        app, started = submission.start_automated_submission(
            app, request.user, request.headers.get("Idempotency-Key", ""), serializer.validated_data["confirm_retry"])
        if started:
            tasks.process_submission.delay(app.pk)
        response = self._detail(app)
        response.status_code = status.HTTP_202_ACCEPTED if started else status.HTTP_200_OK
        return response

    @extend_schema(request=ConfirmSubmittedSerializer, responses={200: ApplicationDetailSerializer})
    @action(detail=True, methods=["post"], url_path="confirm-submitted")
    def confirm_submitted(self, request, pk=None):
        """Mode 2: you submitted on the official page – record it (with the confirmation reference if you have one)."""
        app = self.get_object()
        serializer = ConfirmSubmittedSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        app = submission.confirm_manual_submission(app, request.user, **serializer.validated_data)
        audit.record("application.submitted", obj=app, request=request, channel=app.submission_channel)
        return self._detail(app)


class _OwnedViewSet(viewsets.ModelViewSet):
    def get_queryset(self):
        if getattr(self, "swagger_fake_view", False):
            return self.queryset.none()
        return self.queryset.filter(application__user=self.request.user).select_related("application")


class InterviewViewSet(_OwnedViewSet):
    queryset = Interview.objects.all()
    serializer_class = InterviewSerializer
    filterset_class = InterviewFilter
    ordering_fields = ["scheduled_at"]
    ordering = ["scheduled_at"]

    def perform_create(self, serializer):
        interview = serializer.save()
        app = interview.application
        if app.status in (S.SUBMITTED, S.ASSESSMENT, S.INTERVIEW_COMPLETED):
            lifecycle.transition(app, S.INTERVIEW_SCHEDULED, user=self.request.user,
                                 note=f"{interview.round_name} on {timezone.localtime(interview.scheduled_at):%d %b %Y %H:%M}")


class FollowUpViewSet(_OwnedViewSet):
    queryset = FollowUp.objects.all()
    serializer_class = FollowUpSerializer
    filterset_fields = ["application"]
    ordering_fields = ["due_date"]
    ordering = ["due_date"]
