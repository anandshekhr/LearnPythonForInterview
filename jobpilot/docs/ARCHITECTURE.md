# Architecture

## Phase 1 — repository assessment

The repository `LearnPythonForInterview` contained only learning material: `dqn.py`, `dqn_implementation.py`,
`jssp.py`, `interviewLearn.ipynb` and `haberman.csv`. There was no Django or Angular code, no models, authentication,
deployment configuration or reusable UI components, so JobPilot is a new, self-contained project under `jobpilot/`.
The existing files were left untouched.

Environment findings that shaped the implementation:

- Python 3.13, Node 22, PostgreSQL 16 and Redis 7 were available. pgvector was installed from the distribution package.
- **Angular 19.** Version 19.2 was used as requested.
- **Job-board documentation.** The Greenhouse and Lever docs were checked before building their adapters:
  - Listing endpoints are public.
  - Both apply endpoints require an employer-issued API key.
- **Network access.** Outbound access to `boards-api.greenhouse.io`, `api.lever.co` and `api.ashbyhq.com` was
  blocked in the build sandbox. Adapters are therefore tested against fixtures that follow the documented
  response shapes. A dev-only mock server lets the whole pipeline run end-to-end without internet access.

## System overview

```
                ┌──────────── Angular 19 SPA (Nginx) ─────────────┐
                │ dashboard · profile · resumes · jobs · matches   │
                │ tailoring · applications · calendar · settings   │
                └───────────────┬──────────────────────────────────┘
                                │ /api/v1 (JWT)
┌───────────────────────────────▼──────────────────────────────────────────┐
│ Django + DRF (Gunicorn)                                                   │
│ accounts · profiles · jobs · matching · documents · applications ·        │
│ automation · notifications · analytics · core (audit, crypto, LLM, text)  │
└───────┬───────────────────────────┬───────────────────────┬──────────────┘
        │ ORM                       │ enqueue               │ cache/throttle
┌───────▼────────┐        ┌─────────▼─────────┐     ┌───────▼──────┐
│ PostgreSQL 16  │◄───────│ Celery workers    │────►│ Redis        │
│ + pgvector     │        │ (+ Beat schedule) │     └──────────────┘
└────────────────┘        └─────────┬─────────┘
                                    │ HTTPS (throttled, retried)
                     ┌──────────────▼──────────────┐
                     │ Source adapters             │──► Greenhouse / Lever job-board APIs
                     │ (registry, capabilities)    │
                     └─────────────────────────────┘
          optional: browser worker / local `assist_apply` (Playwright, never submits)
```

## Django apps

| App | Responsibility |
|---|---|
| `core` | `TimeStampedModel`, `AuditLog` + scrubbed `audit.record`, request IDs, JSON logging, error envelope, encrypted private storage, Fernet crypto, untrusted-text helpers, skill taxonomy, LLM and embedding providers, health |
| `accounts` | Email-based `User`, register/login/refresh/logout (blacklist), profile, account deletion (removes files) |
| `profiles` | `CandidateProfile`, `CareerPreference`, `Skill`, `CandidateSkill`, `EmploymentHistory`, `Education`, `Project`, `Certification`, `Resume`, `ResumeVersion`; upload validation, parsing, apply-to-profile, master snapshots |
| `jobs` | `Company`, `JobSource`, `JobPosting`, `JobSkill`, `JobPostingChange`; adapters, HTTP client, normalisation, sync/dedup/expiry, requirement extraction, embeddings |
| `matching` | `ScoringConfig`, `JobMatch`; pure scoring engine + orchestration, pgvector similarity, semantic search |
| `documents` | `GeneratedDocument`; tailoring, fact validation, comparison, DOCX/PDF rendering, cover letters, approval |
| `applications` | `Application`, `ApplicationStatusHistory`, `ApplicationAnswer`, `Interview`, `FollowUp`; lifecycle state machine, answers, submission engine, Playwright helper |
| `automation` | `AutomationSettings` (mode/consent/limits/notification prefs), `AutomationRun`, `ScheduledSearch`; scheduler, recovery and retention tasks |
| `notifications` | `Notification` (deduplicated), preference-aware `notify()`, reminder task |
| `analytics` | Overview metrics, time series, breakdowns, CSV export — computed from real rows only |

## Data model highlights

- **Ownership.** Every user-owned row links back to the user, either directly or via the profile or application.
  Every queryset is filtered by the requesting user, so another user's rows return 404, not 403.
- **Shared registry.** `Company`, `JobSource` and `JobPosting` are shared reference data. Editing a company or
  source is limited to its creator or staff. Employer API keys can be set only by staff and are stored encrypted.
- **Constraints.**

  | Constraint | Purpose |
  |---|---|
  | `uniq_job_per_source(source, external_id)` | One posting per external ID per source |
  | `uniq_source_identifier` | One source per board identifier |
  | `uniq_match_per_user_job` | One match per user and job |
  | `uniq_application_user_job` (partial) | One application per user and job |
  | `uniq_application_idempotency` (partial) | Repeated submission requests can't create duplicates |
  | `one_master_resume_per_user` (partial) | Exactly one master resume |
  | `uniq_resume_file_per_user` | The same file can't be uploaded twice |
  | `uniq_resume_version_number` | Version numbers are unique per user |
  | `uniq_profile_skill`, `uniq_job_skill`, `uniq_answer_per_question`, `uniq_notification_dedupe` | No duplicate rows |
  | Check constraints | Valid salary ranges, valid date ranges, valid application status |
- **Postgres arrays** hold short, user-edited lists (target roles, locations, achievements). Anything that needs
  its own attributes gets a table (skills, employment).
- **JSON fields** are used only for genuinely flexible data:
  - parsed resume drafts;
  - resume content snapshots;
  - match components;
  - adapter config;
  - submission response excerpts;
  - audit metadata.
- **Vectors.** `JobPosting.embedding` and `CandidateProfile.embedding` are 384-dimensional `vector` columns with an
  HNSW cosine index. The extension is created by migration `core.0001_pgvector`.
- **Indexes.** Cover the main access paths: status and date filters, user and category, canonical URL, fingerprint.

## Key flows

### Resume → verified profile
1. `POST /resumes/` validates the extension, size and file signature. It rejects:
   - encrypted or scripted PDFs;
   - macro-enabled or zip-bomb DOCX files;
   - duplicate uploads.

   The file is stored encrypted (Fernet), and parsing is queued.
2. `profiles.tasks.parse_resume` extracts text and produces a rule-based draft:
   - sections;
   - date ranges, with overlapping roles merged for total experience;
   - title and company split;
   - education, projects, certifications and skills.

   An optional LLM pass refines the draft. Its output is **grounded**: any employer, title, institution or skill
   that isn't in the source text is dropped, with a warning.
3. The user edits the draft in the UI and calls `POST /resumes/{id}/apply/`. The profile rows become `verified=True`,
   the resume becomes the master, and an approved master `ResumeVersion` snapshot is created.

### Discovery → matching
1. Celery Beat runs `dispatch_due_searches` (every 15 minutes). This queues `sync_source_task` for each due source,
   with at most one sync per source per 30 minutes even when several users share it.
2. `sync_source` fetches the full listing **outside any transaction**. Each job is upserted in its own short
   transaction, which detects:
   - new jobs;
   - changed jobs (via content hash), recording a change history;
   - reactivated jobs.

   Jobs missing from a complete listing are expired. A guard refuses to expire more than 80% of a large board in
   one sync.
3. `process_job` extracts requirements (rules plus optional grounded LLM), writes `JobSkill` rows, embeds the job,
   then `match_job` scores it for every verified profile. Scores are skipped if the input signature hasn't changed.
4. High scores create one deduplicated notification. Optionally, the job is auto-shortlisted.

### Scoring (apps/matching/scoring.py)
- **Signals.** Eight signals, each scored 0–1, or *unknown*. Weights are per user.
- **Unknown signals** are excluded and the remaining weights renormalised. Information coverage is reported, and
  the explanation says "indicative only" when coverage is low.
- **Missing required skills.** If fewer than half the required skills are verified, the score is capped below
  "good".
- **Exclusions.** Excluded companies or keywords cap the score and say why.
- **Evidence.** Only verified skills count. Unverified skills are listed separately and never credited.

### Tailoring → approval → application
1. `POST /documents/generate/` runs a Celery task that:
   - builds the tailored content from the verified profile snapshot, reordering skills and achievements by
     relevance;
   - optionally has an LLM rephrase it, with each rewrite fact-checked (`check_claims`);
   - validates the result structurally (`validate_tailored_content`);
   - renders PDF and DOCX, and drafts a cover letter.

   Everything is created as `awaiting_approval`, and the application moves to *awaiting approval*.
2. The user reviews the side-by-side comparison and the requirement map, may edit (edits are re-validated), and
   approves. The application becomes *ready to apply*, with the PDF attached.
3. Submission follows the configured mode (see README). Automated submission:
   - takes a row lock by moving the application to `submitting`, only from `ready_to_apply` or
     `submission_failed`;
   - honours `Idempotency-Key` replays;
   - checks consent and the daily limit;
   - makes the network call in a task;
   - records `submitted` only when the source confirms.

   Timeouts and network errors are marked *unconfirmed*, and retrying then requires `confirm_retry=true`.
   Submissions stuck in `submitting` are released by `recover_interrupted_runs`.

## Background tasks & reliability

| Task | Schedule / trigger | Notes |
|---|---|---|
| `dispatch_due_searches` | Beat, every 15 min | Uses `select_for_update(skip_locked=True)` so concurrent beats don't double-dispatch |
| `sync_source_task` | Dispatcher / "Sync now" / new source | Retries transient errors with exponential back-off (60 s × 2ⁿ, max 3); marks the source degraded, then error; notifies the owner |
| `process_job`, `match_job`, `rematch_user` | After sync / profile edits | Idempotent via content hashes and input signatures |
| `parse_resume`, `generate_documents`, `prepare_answers_task` | User actions | Tracked in `AutomationRun` |
| `process_submission` | Submit | Deliberately **not** auto-retried, so an application is never sent twice |
| `check_sources_health` | Beat, every 6 h | — |
| `expire_stale_jobs` | Beat, daily | Safety net for jobs not seen within the grace period |
| `send_due_reminders` | Beat, hourly | Interviews in the next 24 h; follow-ups due |
| `recover_interrupted_runs` | Beat, every 30 min | Runs without a heartbeat become *interrupted*; stuck submissions are released |
| `purge_expired_data` | Beat, daily | Per-user retention and run-history pruning |

Workers run with `acks_late` and `reject_on_worker_lost`, plus time limits and prefetch 1. All task execution is
recorded in `AutomationRun`, which the UI shows under *Automation settings → Execution history*.

## Security & privacy

- **Authentication.**
  - JWTs: 30-minute access tokens; refresh tokens rotate and are blacklisted on logout or rotation.
  - Passwords: Django validators, minimum length 10.
  - Throttling: scoped limits for auth, uploads, submissions and syncs.
- **Authorisation.** All querysets are scoped to the requesting user. Object permissions are checked for the shared
  registry. Employer credentials are staff-only.
- **Files.**
  - Stored outside the web root, with `0600` file permissions.
  - Encrypted with Fernet (`DATA_ENCRYPTION_KEY`).
  - Downloadable only through authenticated, owner-scoped views, and every download is audited.
- **Untrusted content.** Resumes, job descriptions and web pages are treated as data:
  - They are passed to the LLM only inside `<untrusted_*>` tags, under a fixed system policy.
  - Outputs are schema-constrained and fact-checked.
  - The UI renders descriptions as plain text.
  - CSV exports neutralise spreadsheet formulas.
- **Logging.**
  - Structured JSON with request IDs.
  - The audit log scrubs secrets.
  - Resume text, tokens and passwords are never logged.
- **Retention.**
  - Per-user retention window.
  - Deleting your account removes all rows and files.
  - Resume deletion is user-controlled.
- **Web layer.**
  - Nginx sets a strict CSP (no inline scripts), plus `X-Frame-Options`, `nosniff` and a referrer policy.
  - Django sets secure cookies in production, with optional SSL redirect and HSTS.
