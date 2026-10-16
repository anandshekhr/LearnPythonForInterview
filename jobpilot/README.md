# JobPilot — AI-assisted job discovery & application management

JobPilot discovers openings on company job boards, scores them against your **verified** career profile
with an explainable model, prepares factual tailored resumes and cover letters, and tracks every
application from discovery to offer. It is built for quality, not volume: nothing is submitted without
your approval, and it never invents qualifications.

| Layer | Technology |
|---|---|
| API | Python 3.13, Django 5.2, Django REST Framework, SimpleJWT, django-filter, drf-spectacular (OpenAPI) |
| Data | PostgreSQL 16 + pgvector (HNSW cosine index), Redis (cache, broker) |
| Async | Celery workers + Celery Beat |
| AI | Pluggable LLM (`none` / Anthropic Claude / OpenAI-compatible) with JSON-schema outputs; local hashing embeddings or OpenAI-compatible embeddings |
| Documents | pdfplumber, pypdf, python-docx (parsing); reportlab + python-docx (rendering) |
| Automation | Playwright (assisted form pre-fill, never submits) |
| UI | Angular 19 (standalone components, signals), Angular Material, Chart.js |
| Deploy | Docker, Docker Compose, Nginx, Gunicorn |

## What works today (verified by tests)

- **Profile & resumes**: register/login (JWT with rotation + blacklist), upload PDF/DOCX (validated, encrypted at rest),
  background parsing into a *draft*, a review screen where you correct everything, then apply → verified profile +
  master resume snapshot. The uploaded master file is never modified.
- **Discovery**: company registry; Greenhouse and Lever job-board adapters (public, documented listing APIs);
  scheduled sync with throttling, retries, back-off; normalisation (title, seniority, work model, location, salary,
  employment type); dedup by source id, canonical URL and cross-source fingerprint; change detection with history;
  expiry (with a mass-removal guard); source health.
- **Matching**: explainable 0–100 score from 8 weighted, configurable signals (required/preferred skills, experience,
  role & seniority, pgvector semantic similarity, location/work model, salary, employment type). Unknown information is
  skipped, never counted as zero. Gaps, relevant experience, category and next action are shown for each job.
- **Tailoring**: per-job resume that *selects, reorders and (optionally, with an LLM) rephrases* verified facts only;
  every AI rewrite is checked for new numbers, unverified skills and unknown names and rejected if it adds facts.
  Side-by-side comparison, requirement→evidence map, PDF + DOCX export, cover letter, mandatory approval.
- **Applications**: full lifecycle (discovered → … → offer/rejected/withdrawn/closed) with validated transitions and
  history; duplicate prevention (DB constraints + cross-source checks); three modes:
  1. *Discovery only* — official link only.
  2. *Assisted* — application kit, prepared answers, official page; optional local Playwright helper
     (`manage.py assist_apply <id>`) that pre-fills supported Greenhouse/Lever forms and stops before submit.
  3. *Authorized automated* — Greenhouse/Lever **apply APIs with an employer-issued key only**, explicit consent,
     daily limit, idempotency keys, sensitive answers only after explicit review, ambiguous outcomes never retried
     silently.
- **Tracking**: interviews (calendar), follow-ups, notes, reminders, notifications, analytics and CSV export.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the design and [docs/SOURCES.md](docs/SOURCES.md) for what each
source can and cannot do.

## Quick start (Docker)

```bash
cd jobpilot
cp .env.example .env
# Fill in DJANGO_SECRET_KEY, POSTGRES_PASSWORD and DATA_ENCRYPTION_KEY:
python3 -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
docker compose up -d --build
open http://localhost:8080          # UI; API docs at http://localhost:8080/api/docs/ (via /api/schema/)
docker compose exec api python manage.py createsuperuser   # optional, for /admin and employer API keys
```

Then: register → **Resume manager** → upload → **Review & verify** → **Companies & sources** → add a company and its
Greenhouse board token or Lever site name → **Recommended**.

## Local development

Requirements: Python 3.13, Node 22, PostgreSQL 16 with pgvector, Redis 7.

```bash
# Backend
cd jobpilot/backend
python3 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
cat > .env <<EOF
DJANGO_DEBUG=1
DJANGO_SECRET_KEY=dev-only-change-me
DATA_ENCRYPTION_KEY=$(.venv/bin/python -c "from cryptography.fernet import Fernet;print(Fernet.generate_key().decode())")
EOF
createuser -s jobpilot; createdb -O jobpilot jobpilot     # or use docker compose up db redis
.venv/bin/python manage.py migrate
.venv/bin/python manage.py runserver                       # http://localhost:8000/api/docs/
.venv/bin/celery -A config worker -l info                  # second terminal
.venv/bin/celery -A config beat -l info                    # third terminal (scheduled discovery)

# Frontend
cd ../frontend && npm ci && npx ng serve                   # http://localhost:4200 (proxies /api → :8000)
```

No internet access to job boards (e.g. in a sandbox)? Run the fixture-backed mock board and point the adapters at it
(development only — refused when `DJANGO_DEBUG=0`):

```bash
.venv/bin/python scripts/mock_job_boards.py --port 8765 &
export SOURCE_API_BASE_OVERRIDES="greenhouse=http://127.0.0.1:8765/v1/boards,lever=http://127.0.0.1:8765/v0/postings"
```

## Tests

```bash
cd jobpilot/backend && .venv/bin/python -m pytest            # 114 tests: unit, API, integration, Playwright fixtures
.venv/bin/ruff check .
cd ../frontend && CHROME_BIN=/path/to/chrome npx ng test --watch=false   # Angular unit tests
npx ng build                                                              # production build
# Full UI workflow against a running stack (fresh database):
cd ../backend && .venv/bin/python ../e2e/ui_workflow.py --resume /path/to/test.docx --base http://localhost:4200
```

Details in [docs/TESTING.md](docs/TESTING.md). Tests never touch the network: outbound HTTP is blocked and all
sources are mocked; no real applications are ever submitted.

## Configuration

All configuration is via environment variables — see [`.env.example`](.env.example). Highlights:

| Variable | Purpose |
|---|---|
| `DATA_ENCRYPTION_KEY` | Fernet key encrypting uploaded/generated files and employer API keys (required in production) |
| `LLM_PROVIDER`, `LLM_MODEL`, `LLM_API_KEY` | `none` (default, deterministic), `anthropic` (default model `claude-opus-5-5`), or `openai`-compatible |
| `EMBEDDING_PROVIDER` | `hashing` (local, default) or `openai`-compatible returning 384-d vectors |
| `BROWSER_AUTOMATION_ENABLED` | Enables the Playwright helper/worker (off by default) |
| `THROTTLE_*` | API rate limits (auth, uploads, submissions, syncs) |

Your initial preferences (target roles, ~7 years, Noida/Delhi NCR/India, hybrid/remote, ₹30–40 LPA) are seeded as
defaults for every new profile and are fully editable under **Career profile → Preferences**.

## Known limitations

- **Employer keys for automated submission.** Greenhouse and Lever only accept API applications with an
  API key that the employer issues. For a typical candidate, Mode 3 is therefore unavailable, and JobPilot
  falls back to Mode 2 (assisted). It never scrapes or bypasses forms to get around this.
- **No other sources yet.** Only Greenhouse and Lever are integrated. Ashby was evaluated, but its official
  documentation couldn't be verified from the build environment, so no adapter ships. Workday,
  SmartRecruiters and others have no adapter.
- **Status tracking is manual.** After submission, application status is updated by you. ATSs don't expose
  candidate-side status.
- **Matching without an embedding model.** Without one configured, semantic similarity uses lexical hashing
  embeddings. This is useful, but weaker than model embeddings, which is why it's only one weighted signal.
- **Rule-based parsing has limits.** Resume parsing is heuristic and works best on conventional layouts. The
  optional LLM pass improves it, and every result goes through your review either way.
- **Email notifications aren't built.** Notifications are in-app only. Email is a configurable placeholder.

## Documentation

- [Architecture & data model](docs/ARCHITECTURE.md) (includes the Phase 1 repository assessment)
- [Job sources, capabilities and compliance](docs/SOURCES.md)
- [REST API overview](docs/API.md) — the live OpenAPI/Swagger UI is at `/api/docs/`
- [Testing](docs/TESTING.md)
- [Operations: deployment, backups, recovery](docs/OPERATIONS.md)
