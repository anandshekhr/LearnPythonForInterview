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


class Handler(BaseHTTPRequestHandler):
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
