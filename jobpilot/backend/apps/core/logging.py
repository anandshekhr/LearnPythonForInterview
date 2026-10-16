import contextvars
import logging

_request_id = contextvars.ContextVar("request_id", default="")


def set_request_id(value):
    return _request_id.set(value)


def reset_request_id(token):
    _request_id.reset(token)


def get_request_id():
    return _request_id.get()


class RequestIdFilter(logging.Filter):
    def filter(self, record):
        record.request_id = get_request_id()
        return True
