from django.contrib import admin

from .models import CandidateProfile, Resume, ResumeVersion, Skill


@admin.register(CandidateProfile)
class CandidateProfileAdmin(admin.ModelAdmin):
    list_display = ("user", "full_name", "total_experience_years", "verified_at")
    exclude = ("embedding",)


@admin.register(Resume)
class ResumeAdmin(admin.ModelAdmin):
    list_display = ("user", "original_filename", "parse_status", "is_master", "created_at")
    exclude = ("raw_text", "parsed_data")  # personal data: not shown in admin


@admin.register(ResumeVersion)
class ResumeVersionAdmin(admin.ModelAdmin):
    list_display = ("user", "version_number", "kind", "status", "job", "created_at")
    list_filter = ("kind", "status")
    exclude = ("content",)


@admin.register(Skill)
class SkillAdmin(admin.ModelAdmin):
    list_display = ("name", "category")
    search_fields = ("name",)
