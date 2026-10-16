import logging

from django.core.exceptions import ValidationError as DjangoValidationError
from django.db import IntegrityError
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import exception_handler as drf_exception_handler

from .logging import get_request_id

logger = logging.getLogger(__name__)


class DomainError(Exception):
    """A business-rule violation that should surface as a 4xx with a stable code."""

    status_code = status.HTTP_400_BAD_REQUEST

    def __init__(self, message, code="invalid", status_code=None, details=None):
        super().__init__(message)
        self.message = message
        self.code = code
        self.details = details or {}
        if status_code is not None:
            self.status_code = status_code


def exception_handler(exc, context):
    if isinstance(exc, DomainError):
        return Response(
            {"error": {"code": exc.code, "message": exc.message, "details": exc.details}, "request_id": get_request_id()},
            status=exc.status_code,
        )
    if isinstance(exc, DjangoValidationError):
        return Response(
            {"error": {"code": "invalid", "message": "; ".join(exc.messages)}, "request_id": get_request_id()},
            status=status.HTTP_400_BAD_REQUEST,
        )
    if isinstance(exc, IntegrityError):
        logger.warning("integrity_error", extra={"view": context.get("view").__class__.__name__})
        return Response(
            {"error": {"code": "conflict", "message": "The request conflicts with existing data."}, "request_id": get_request_id()},
            status=status.HTTP_409_CONFLICT,
        )
    response = drf_exception_handler(exc, context)
    if response is not None and isinstance(response.data, dict):
        response.data.setdefault("request_id", get_request_id())
    return response
