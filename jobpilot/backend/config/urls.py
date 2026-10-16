from django.contrib import admin
from django.urls import include, path
from drf_spectacular.views import SpectacularAPIView, SpectacularSwaggerView

from apps.core.views import HealthView

api_v1 = [
    path("auth/", include("apps.accounts.urls")),
    path("", include("apps.profiles.urls")),
    path("", include("apps.jobs.urls")),
    path("", include("apps.matching.urls")),
    path("", include("apps.documents.urls")),
    path("", include("apps.applications.urls")),
    path("", include("apps.automation.urls")),
    path("", include("apps.notifications.urls")),
    path("", include("apps.analytics.urls")),
]

urlpatterns = [
    path("admin/", admin.site.urls),
    path("health/", HealthView.as_view(), name="health"),
    path("api/v1/", include(api_v1)),
    path("api/schema/", SpectacularAPIView.as_view(), name="schema"),
    path("api/docs/", SpectacularSwaggerView.as_view(url_name="schema"), name="swagger-ui"),
]
