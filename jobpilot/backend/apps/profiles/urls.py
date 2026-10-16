from django.urls import path
from rest_framework.routers import DefaultRouter

from . import views

router = DefaultRouter()
router.register("profile/skills", views.CandidateSkillViewSet, basename="candidate-skill")
router.register("profile/employment", views.EmploymentViewSet, basename="employment")
router.register("profile/education", views.EducationViewSet, basename="education")
router.register("profile/projects", views.ProjectViewSet, basename="project")
router.register("profile/certifications", views.CertificationViewSet, basename="certification")
router.register("resumes", views.ResumeViewSet, basename="resume")
router.register("resume-versions", views.ResumeVersionViewSet, basename="resume-version")

urlpatterns = [
    path("profile/", views.ProfileView.as_view(), name="profile"),
    path("profile/preferences/", views.PreferenceView.as_view(), name="preferences"),
] + router.urls
