from django.contrib import admin

from .models import JobMatch, ScoringConfig


@admin.register(JobMatch)
class JobMatchAdmin(admin.ModelAdmin):
    list_display = ("user", "job", "score", "category", "user_state")
    list_filter = ("category", "user_state")


admin.site.register(ScoringConfig)
