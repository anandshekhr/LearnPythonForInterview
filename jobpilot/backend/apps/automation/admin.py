from django.contrib import admin

from .models import AutomationRun, AutomationSettings, ScheduledSearch


@admin.register(AutomationRun)
class AutomationRunAdmin(admin.ModelAdmin):
    list_display = ("started_at", "kind", "status", "user", "target_type", "target_id", "error")
    list_filter = ("kind", "status")
    readonly_fields = [f.name for f in AutomationRun._meta.fields]


@admin.register(AutomationSettings)
class AutomationSettingsAdmin(admin.ModelAdmin):
    list_display = ("user", "application_mode", "automated_submission_consent_at", "max_submissions_per_day")


@admin.register(ScheduledSearch)
class ScheduledSearchAdmin(admin.ModelAdmin):
    list_display = ("name", "user", "enabled", "interval_hours", "next_run_at")
