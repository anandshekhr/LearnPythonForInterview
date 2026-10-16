from django.contrib import admin
from django.contrib.auth.admin import UserAdmin as BaseUserAdmin

from .models import User


@admin.register(User)
class UserAdmin(BaseUserAdmin):
    list_display = ("email", "full_name", "is_staff", "date_joined")
    ordering = ("email",)
    fieldsets = BaseUserAdmin.fieldsets + (("Profile", {"fields": ("full_name",)}),)
