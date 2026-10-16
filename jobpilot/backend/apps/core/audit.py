import logging

from .logging import get_request_id
from .models import AuditLog

logger = logging.getLogger(__name__)

SENSITIVE_KEYS = {"password", "token", "access", "refresh", "api_key", "secret", "authorization", "resume_text", "raw_text"}


def _scrub(value):
    if isinstance(value, dict):
        return {k: ("[redacted]" if k.lower() in SENSITIVE_KEYS else _scrub(v)) for k, v in value.items()}
    if isinstance(value, list):
        return [_scrub(v) for v in value]
    return value


def record(action, user=None, obj=None, request=None, **metadata):
    """Write an audit entry. Failures are logged, never raised to the caller."""
    try:
        ip = None
        if request is not None:
            ip = request.META.get("REMOTE_ADDR")
            if user is None and getattr(request, "user", None) is not None and request.user.is_authenticated:
                user = request.user
        return AuditLog.objects.create(
            user=user if (user is not None and getattr(user, "pk", None)) else None,
            action=action,
            object_type=obj.__class__.__name__ if obj is not None else "",
            object_id=str(obj.pk) if obj is not None and obj.pk is not None else "",
            metadata=_scrub(metadata),
            ip_address=ip,
            request_id=get_request_id() or "",
        )
    except Exception:  # pragma: no cover - audit must not break the request
        logger.exception("audit_write_failed", extra={"action": action})
        return None
