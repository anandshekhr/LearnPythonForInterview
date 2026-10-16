# Testing

## Backend (pytest)

```bash
cd jobpilot/backend
.venv/bin/python -m pytest              # needs PostgreSQL (with pgvector) and Redis-free settings (config.settings_test)
.venv/bin/python -m pytest --create-db  # after changing migrations
.venv/bin/ruff check .
```

`config/settings_test.py` makes Celery tasks run eagerly, uses an in-memory cache, turns off throttling and back-off
sleeps, and forces `LLM_PROVIDER=none`. `conftest.py` blocks **all** real outbound HTTP (tests fail if any code tries
to reach the internet) and runs `on_commit` hooks immediately.

| Area | Where |
|---|---|
| Resume parsing, validation (bad types, oversize, macro DOCX, fake PDFs), LLM grounding | `apps/profiles/tests/test_parser.py` |
| Auth, profile & preferences API, upload→parse→review→apply, encryption at rest, isolation, account deletion | `apps/profiles/tests/test_api.py` |
| Greenhouse/Lever adapters, retries/back-off/Retry-After, throttling, identifiers | `apps/jobs/tests/test_adapters.py` |
| Normalisation (titles, seniority, work model, salary, locations, URLs), requirement extraction | `apps/jobs/tests/test_normalization.py` |
| Sync pipeline, change detection, expiry + mass-removal guard, cross-source dedup, failures | `apps/jobs/tests/test_sync.py` |
| Registry permissions, staff-only encrypted credentials, filters, matches API | `apps/jobs/tests/test_api.py` |
| Scoring signals, unknown-information handling, caps, exclusions, configurable weights | `apps/matching/tests/test_scoring.py` |
| Tailoring factuality, LLM rewrite rejection, approval, assisted flow, state machine, duplicates, interviews/reminders, automated submission (consent, sensitive review, idempotency, ambiguous failures, daily limit, interrupted recovery, Lever multipart) | `tests/test_end_to_end.py` |
| Playwright form pre-fill against local fixture pages (never submits, CAPTCHA/login detection, host allow-list) | `apps/applications/tests/test_browser_assist.py` |
| Scheduling, Celery retries, source-failure notifications, notifications API & preferences, analytics, CSV injection, retention, audit scrubbing, health, OpenAPI | `tests/test_operations.py` |
| Prompt-injection delimiting, LLM request shape, crypto, HTML sanitising, skill extraction, embeddings | `apps/core/tests/test_core.py` |

Browser tests need Chromium. Set `BROWSER_EXECUTABLE_PATH` if Playwright's bundled browser isn't installed
(`playwright install chromium` otherwise).

## Frontend

```bash
cd jobpilot/frontend
CHROME_BIN=/path/to/chrome npx ng test --watch=false   # Karma + headless Chrome (karma.conf.js)
npx ng build                                            # production build
```

## End-to-end UI workflow

`e2e/ui_workflow.py` drives the real UI against a running stack. Steps:

1. Register and upload a resume.
2. Review and apply the parsed details.
3. Add a company and a Greenhouse source, then wait for the Celery worker to sync and score.
4. Open the recommended jobs and tailor a resume.
5. Approve it and record the submission.
6. Check the dashboard and analytics.

It fails on any browser console error.

```bash
# fresh database; mock boards; backend with overrides; worker; UI
.venv/bin/python manage.py flush --noinput
.venv/bin/python scripts/mock_job_boards.py --port 8765 &
export SOURCE_API_BASE_OVERRIDES="greenhouse=http://127.0.0.1:8765/v1/boards,lever=http://127.0.0.1:8765/v0/postings"
.venv/bin/python manage.py runserver & .venv/bin/celery -A config worker &
(cd ../frontend && npx ng serve &)
.venv/bin/python -c "import django,os;os.environ['DJANGO_SETTINGS_MODULE']='config.settings';django.setup();\
from apps.core.testing import make_docx;open('/tmp/test.docx','wb').write(make_docx())"
.venv/bin/python ../e2e/ui_workflow.py --resume /tmp/test.docx --base http://127.0.0.1:4200 [--mobile]
```

The synthetic resume (`apps/core/testing.py`) describes a fictional person; never commit real resume data.
