from rest_framework.routers import DefaultRouter

from . import views

router = DefaultRouter()
router.register("applications", views.ApplicationViewSet, basename="application")
router.register("interviews", views.InterviewViewSet, basename="interview")
router.register("follow-ups", views.FollowUpViewSet, basename="follow-up")
urlpatterns = router.urls
