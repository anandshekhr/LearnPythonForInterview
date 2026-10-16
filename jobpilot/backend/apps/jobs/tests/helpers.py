import copy
import json
from pathlib import Path

FIXTURES = Path(__file__).parent / "fixtures"


def load(name):
    return copy.deepcopy(json.loads((FIXTURES / name).read_text()))


def mock_greenhouse(respx_mock, token="examplesaas", payload=None, status=200):
    return respx_mock.get(f"https://boards-api.greenhouse.io/v1/boards/{token}/jobs").respond(
        status, json=payload if payload is not None else load("greenhouse_jobs.json"))


def mock_lever(respx_mock, site="examplelever", payload=None):
    return respx_mock.get(f"https://api.lever.co/v0/postings/{site}").respond(
        200, json=payload if payload is not None else load("lever_postings.json"))
