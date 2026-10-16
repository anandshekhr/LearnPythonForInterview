from django.urls import path
from rest_framework.routers import DefaultRouter

from . import views

router = DefaultRouter()
router.register("documents", views.GeneratedDocumentViewSet, basename="document")
urlpatterns = [path("documents/generate/", views.GenerateDocumentsView.as_view(), name="documents-generate")] + router.urls
