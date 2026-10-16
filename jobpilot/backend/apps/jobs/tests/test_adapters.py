import httpx
import pytest

from apps.jobs.adapters.greenhouse import GreenhouseAdapter
from apps.jobs.adapters.lever import LeverAdapter
from apps.jobs.http import SourceHttpClient, SourceHTTPError

from .helpers import load, mock_greenhouse, mock_lever


def client(**kw):
    return SourceHttpClient("test", min_interval=0, sleep=lambda s: None, **kw)


def test_greenhouse_normalisation(respx_mock):
    mock_greenhouse(respx_mock)
    jobs = list(GreenhouseAdapter("examplesaas", http=client()).fetch_listings())
    assert len(jobs) == 3
    job = jobs[0]
    assert job.external_id == "4012001"
    assert job.canonical_url == "https://boards.greenhouse.io/examplesaas/jobs/4012001?gh_jid=4012001"  # utm dropped
    assert job.work_model == "hybrid"
    assert job.locations == ["Noida"] or "Noida" in job.locations[0]
    assert job.employment_type == "full_time"
    assert (job.salary.minimum, job.salary.maximum, job.salary.currency) == (3_200_000, 4_200_000, "INR")
    assert "<" not in job.description_text and "Django REST Framework" in job.description_text
    assert job.published_at.year == 2026
    assert jobs[1].work_model == "remote"
    assert jobs[2].work_model == "onsite"


def test_greenhouse_questions_flag_sensitive(respx_mock):
    respx_mock.get("https://boards-api.greenhouse.io/v1/boards/examplesaas/jobs/4012001").respond(
        200, json=load("greenhouse_job_questions.json"))
    questions = GreenhouseAdapter("examplesaas", http=client()).fetch_application_questions("4012001")
    by_key = {q.key: q for q in questions}
    assert by_key["first_name"].required and not by_key["first_name"].sensitive
    assert by_key["question_1002"].sensitive  # visa sponsorship
    assert by_key["gender"].sensitive  # EEOC compliance block
    assert by_key["resume"].field_type == "file"


def test_lever_normalisation_and_pagination(respx_mock):
    route = mock_lever(respx_mock)
    jobs = list(LeverAdapter("examplelever", http=client()).fetch_listings())
    assert route.call_count == 1  # fewer than page size -> single request
    assert route.calls[0].request.url.params["mode"] == "json"
    fs, data = jobs
    assert fs.work_model == "hybrid" and fs.locations == ["Gurugram", "Noida"]
    assert fs.employment_type == "full_time"
    assert (fs.salary.minimum, fs.salary.maximum, fs.salary.currency) == (2_800_000, 3_600_000, "INR")  # parsed from text
    assert fs.apply_url.endswith("/apply")
    assert data.employment_type == "contract" and data.salary.minimum == 2_000_000


def test_lever_eu_region(respx_mock):
    route = respx_mock.get("https://api.eu.lever.co/v0/postings/euco").respond(200, json=[])
    assert list(LeverAdapter("euco", {"region": "eu"}, http=client()).fetch_listings()) == []
    assert route.called


def test_retries_with_backoff_then_succeeds(respx_mock):
    sleeps = []
    route = respx_mock.get("https://boards-api.greenhouse.io/v1/boards/x/jobs").mock(side_effect=[
        httpx.Response(503), httpx.Response(429, headers={"Retry-After": "2"}), httpx.Response(200, json={"jobs": []}),
    ])
    http = SourceHttpClient("t", min_interval=0, sleep=sleeps.append, backoff_base=1)
    assert list(GreenhouseAdapter("x", http=http).fetch_listings()) == []
    assert route.call_count == 3
    assert sleeps[1] == 2.0  # honours Retry-After
    assert 1 <= sleeps[0] <= 2


def test_non_retryable_error_raises_immediately(respx_mock):
    route = respx_mock.get("https://boards-api.greenhouse.io/v1/boards/missing/jobs").respond(404)
    with pytest.raises(SourceHTTPError) as exc:
        list(GreenhouseAdapter("missing", http=client()).fetch_listings())
    assert exc.value.status_code == 404 and not exc.value.retryable
    assert route.call_count == 1


def test_gives_up_after_max_retries(respx_mock):
    route = respx_mock.get("https://boards-api.greenhouse.io/v1/boards/down/jobs").respond(502)
    with pytest.raises(SourceHTTPError) as exc:
        list(GreenhouseAdapter("down", http=client(max_retries=2)).fetch_listings())
    assert exc.value.retryable and route.call_count == 3


def test_throttle_waits_between_requests(respx_mock, settings):
    settings.SOURCE_THROTTLE_ENABLED = True
    respx_mock.get("https://boards-api.greenhouse.io/v1/boards/t/jobs").respond(200, json={"jobs": []})
    sleeps = []
    http = SourceHttpClient("throttle-test", min_interval=5, sleep=sleeps.append)
    adapter = GreenhouseAdapter("t", http=http)
    list(adapter.fetch_listings())
    list(adapter.fetch_listings())
    assert sleeps and 4 < sleeps[0] <= 5


def test_identifier_validation():
    from apps.jobs.adapters.base import AdapterConfigError

    assert GreenhouseAdapter.validate_identifier(" acme ") == "acme"
    with pytest.raises(AdapterConfigError):
        GreenhouseAdapter.validate_identifier("../../etc")
