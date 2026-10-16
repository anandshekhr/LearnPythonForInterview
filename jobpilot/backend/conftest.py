import os

import pytest

os.environ.setdefault("JOBPILOT_TESTING", "1")


@pytest.fixture(autouse=True)
def _immediate_on_commit(monkeypatch):
    """Tests run inside a transaction that never commits; run on_commit hooks immediately
    so eager Celery tasks queued after commit still execute."""
    monkeypatch.setattr("django.db.transaction.on_commit", lambda func, using=None, robust=False: func())


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    """Block real outbound HTTP in tests. respx mocks patch the same layer later and take precedence."""
    import httpcore

    def guard(self, request):
        host = request.url.host.decode() if isinstance(request.url.host, bytes) else request.url.host
        raise RuntimeError(f"Real network access attempted in tests: {host}")

    monkeypatch.setattr(httpcore._sync.connection.HTTPConnection, "handle_request", guard)
