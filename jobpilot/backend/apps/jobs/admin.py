from django.contrib import admin

from .models import Company, JobPosting, JobSource


@admin.register(Company)
class CompanyAdmin(admin.ModelAdmin):
    list_display = ("name", "company_type", "industry", "created_at")
    search_fields = ("name",)


@admin.register(JobSource)
class JobSourceAdmin(admin.ModelAdmin):
    list_display = ("company", "source_type", "identifier", "enabled", "status", "last_success_at", "active_job_count")
    list_filter = ("source_type", "status", "enabled")
    exclude = ("submission_api_key_encrypted",)


@admin.register(JobPosting)
class JobPostingAdmin(admin.ModelAdmin):
    list_display = ("title", "company", "status", "work_model", "published_at", "last_seen_at")
    list_filter = ("status", "work_model", "source__source_type")
    search_fields = ("title", "company__name")
    exclude = ("embedding",)
