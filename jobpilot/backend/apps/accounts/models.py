from django.contrib.auth.models import AbstractUser
from django.db import models


class User(AbstractUser):
    """Login is by email. `username` is kept for admin compatibility and mirrors email."""

    email = models.EmailField(unique=True)
    full_name = models.CharField(max_length=200, blank=True)

    USERNAME_FIELD = "email"
    REQUIRED_FIELDS = ["username"]

    def save(self, *args, **kwargs):
        self.email = (self.email or "").strip().lower()
        if not self.username:
            self.username = self.email
        super().save(*args, **kwargs)
