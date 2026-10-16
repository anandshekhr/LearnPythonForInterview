from django.urls import path
from rest_framework.routers import DefaultRouter

from . import views

router = DefaultRouter()
router.register("matches", views.JobMatchViewSet, basename="match")
urlpatterns = [path("scoring-config/", views.ScoringConfigView.as_view(), name="scoring-config")] + router.urls
