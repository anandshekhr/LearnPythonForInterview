from rest_framework.routers import DefaultRouter

from . import views

router = DefaultRouter()
router.register("companies", views.CompanyViewSet, basename="company")
router.register("sources", views.JobSourceViewSet, basename="source")
router.register("jobs", views.JobPostingViewSet, basename="job")
urlpatterns = router.urls
