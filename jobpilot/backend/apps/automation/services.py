from .models import AutomationSettings


def get_settings(user) -> AutomationSettings:
    obj, _ = AutomationSettings.objects.get_or_create(user=user)
    return obj
