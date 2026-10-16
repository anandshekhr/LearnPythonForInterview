from django.urls import path

from . import views

urlpatterns = [
    path("analytics/overview/", views.OverviewView.as_view(), name="analytics-overview"),
    path("analytics/timeseries/", views.TimeseriesView.as_view(), name="analytics-timeseries"),
    path("analytics/breakdowns/", views.BreakdownView.as_view(), name="analytics-breakdowns"),
    path("analytics/applications.csv", views.ApplicationReportView.as_view(), name="analytics-report"),
]
