import re
import uuid

from .logging import reset_request_id, set_request_id

_VALID = re.compile(r"^[A-Za-z0-9\-]{8,64}$")


class RequestIdMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        incoming = request.headers.get("X-Request-ID", "")
        request_id = incoming if _VALID.match(incoming) else uuid.uuid4().hex
        token = set_request_id(request_id)
        try:
            response = self.get_response(request)
        finally:
            reset_request_id(token)
        response["X-Request-ID"] = request_id
        return response
