from rest_framework import serializers

from .models import Notification


class NotificationSerializer(serializers.ModelSerializer):
    class Meta:
        model = Notification
        fields = ["id", "kind", "title", "body", "link", "object_type", "object_id", "read_at", "created_at"]
        read_only_fields = fields
