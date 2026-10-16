from django.contrib import admin

from .models import GeneratedDocument


@admin.register(GeneratedDocument)
class GeneratedDocumentAdmin(admin.ModelAdmin):
    list_display = ("user", "kind", "format", "status", "job", "created_at")
    list_filter = ("kind", "status", "format")
    exclude = ("text_content",)
