from django.urls import path
from rest_framework.routers import DefaultRouter

from . import views

router = DefaultRouter()
router.register("automation/runs", views.AutomationRunViewSet, basename="automation-run")
router.register("scheduled-searches", views.ScheduledSearchViewSet, basename="scheduled-search")
urlpatterns = [path("automation/settings/", views.AutomationSettingsView.as_view(), name="automation-settings")] + router.urls
