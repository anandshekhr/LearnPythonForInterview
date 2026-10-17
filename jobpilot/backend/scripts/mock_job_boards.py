"""Development-only server that mimics the documented Greenhouse and Lever listing endpoints
using the test fixtures, so the full pipeline can be exercised without internet access.

    python scripts/mock_job_boards.py --port 8765
    SOURCE_API_BASE_OVERRIDES=greenhouse=http://127.0.0.1:8765/v1/boards,lever=http://127.0.0.1:8765/v0/postings
"""
import argparse
import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

FIXTURES = Path(__file__).resolve().parent.parent / "apps" / "jobs" / "tests" / "fixtures"


CAREER_JOBS = {
    "201-senior-python-developer": ("Senior Python Developer", "Noida", "6+ years of experience with Python, Django, PostgreSQL and Celery"),
    "202-ai-engineer": ("AI Engineer", "Gurugram", "4+ years of experience with Python, LLMs, RAG and AWS"),
}


def _career_job_html(slug):
    title, city, req = CAREER_JOBS[slug]
    posting = {"@context": "https://schema.org", "@type": "JobPosting", "title": title, "datePosted": "2026-10-10",
               "description": f"<h3>Requirements</h3><ul><li>{req}</li></ul>", "employmentType": "FULL_TIME",
               "hiringOrganization": {"@type": "Organization", "name": "Example Careers Co"},
               "jobLocation": {"@type": "Place", "address": {"addressLocality": city, "addressCountry": "IN"}},
               "baseSalary": {"@type": "MonetaryAmount", "currency": "INR",
                              "value": {"@type": "QuantitativeValue", "minValue": 3000000, "maxValue": 4000000, "unitText": "YEAR"}}}
    return f'<html><head><script type="application/ld+json">{json.dumps(posting)}</script></head><body>{title}</body></html>'


class Handler(BaseHTTPRequestHandler):
    def _html(self, status, body):
        data = body.encode()
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _json(self, status, payload):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        path = self.path.split("?")[0]
        if re.fullmatch(r"/v1/boards/[\w.\-]+/jobs", path):
            return self._json(200, json.loads((FIXTURES / "greenhouse_jobs.json").read_text()))
        if re.fullmatch(r"/v1/boards/[\w.\-]+/jobs/\d+", path):
            return self._json(200, json.loads((FIXTURES / "greenhouse_job_questions.json").read_text()))
        if path == "/careers":  # a company-hosted careers page publishing schema.org JobPosting
            links = "".join(f'<a href="/careers/jobs/{slug}">{t}</a>' for slug, (t, _, _) in CAREER_JOBS.items())
            return self._html(200, f"<html><body><h1>Careers</h1>{links}<a href='/about'>About</a></body></html>")
        m = re.fullmatch(r"/careers/jobs/([\w\-]+)", path)
        if m and m.group(1) in CAREER_JOBS:
            return self._html(200, _career_job_html(m.group(1)))
        if re.fullmatch(r"/v0/postings/[\w.\-]+", path):
            return self._json(200, json.loads((FIXTURES / "lever_postings.json").read_text()))
        return self._json(404, {"error": "not found"})

    def do_POST(self):  # noqa: N802
        return self._json(403, {"error": "The mock server does not accept applications"})

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
