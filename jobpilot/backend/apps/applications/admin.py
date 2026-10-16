from django.contrib import admin

from .models import Application, ApplicationStatusHistory, FollowUp, Interview


class HistoryInline(admin.TabularInline):
    model = ApplicationStatusHistory
    extra = 0
    readonly_fields = ("from_status", "to_status", "actor", "changed_by", "note", "created_at")


@admin.register(Application)
class ApplicationAdmin(admin.ModelAdmin):
    list_display = ("company_name", "job_title", "user", "status", "submission_channel", "submitted_at")
    list_filter = ("status", "submission_channel", "source_type")
    search_fields = ("company_name", "job_title", "user__email")
    inlines = [HistoryInline]
    readonly_fields = ("submission_response", "idempotency_key")


admin.site.register(Interview)
admin.site.register(FollowUp)
