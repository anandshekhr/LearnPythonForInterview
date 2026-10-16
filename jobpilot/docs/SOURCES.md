# Job sources, capabilities and compliance

JobPilot only uses **official, publicly documented** job-board interfaces. It does not scrape authenticated or
restricted pages and never bypasses CAPTCHA, MFA, rate limits or other access controls.

## Supported sources

| Source | Listing (public) | Details | Salary | Questions | API submission | Browser assist |
|---|---|---|---|---|---|---|
| Greenhouse Job Board API | `GET boards-api.greenhouse.io/v1/boards/{token}/jobs?content=true` | ✓ | `pay_input_ranges` when published | ✓ (`?questions=true`, incl. compliance/EEOC flagged sensitive) | `POST /v1/boards/{token}/jobs/{id}` **with the employer's Job Board API key** (HTTP Basic) | ✓ hosted form |
| Lever Postings API | `GET api.lever.co/v0/postings/{site}?mode=json` (EU: `api.eu.lever.co`) | ✓ | `salaryRange` when published | ✗ (not exposed by the API) | `POST /v0/postings/{site}/{id}?key=…` **with the employer's key**; Lever limits this to 2 POST/s | ✓ hosted form |

How to find the identifier:
- Greenhouse: the board token is the path segment in `boards.greenhouse.io/<token>` or `job-boards.greenhouse.io/<token>`.
- Lever: the site name is the path segment in `jobs.lever.co/<site>`; set `config: {"region": "eu"}` for `jobs.eu.lever.co`.

Use **Test** on the sources page to confirm a board is reachable before relying on it.

### About automated submission
Both ATSs require an API key created by the **employer** to accept applications through their APIs. A candidate
normally does not have one, so:
- Mode 3 is enabled **per source** only when a staff user stores an employer-issued key *and* confirms that API
  submission is authorised for that board (`POST /sources/{id}/submission-credentials/`).
- Otherwise the application kit (approved resume, prepared answers, official link) and the optional browser helper
  are used, and you submit on the employer's page.
- A submission is recorded as submitted only when the API confirms it (Greenhouse: `200` with `success`; Lever:
  `{"ok": true, "applicationId": …}`), or when you confirm that you submitted it yourself.

### Sources evaluated but not integrated
- **Ashby** (`api.ashbyhq.com/posting-api/job-board/{name}`): appears to offer a public listing endpoint, but its
  official documentation could not be retrieved from the build environment, and the third-party descriptions found
  disagree on the response shape. Per project policy it is not integrated until verified against Ashby's own docs.
- LinkedIn, Naukri, Indeed and similar aggregators: their terms do not permit automated access to listings or
  applications without a partnership, so there are no adapters.
- Workday, SmartRecruiters, iCIMS and others: possible future adapters, each requiring separate verification of the
  current documentation and terms.

## Browser assistance (Playwright)

- Only hosts in `BROWSER_ASSIST_ALLOWED_HOSTS` (default: Greenhouse and Lever hosted forms, HTTPS only) can be opened.
  Redirects to other hosts are rejected.
- It fills fields it can identify with confidence: name, email, phone, links, resume upload, cover letter, and
  reviewed answers.
- **It never clicks submit.** Login walls stop it, and CAPTCHA / human-verification widgets are reported so that you
  complete them yourself.
- Sensitive questions (legal declarations, demographics, disability, sponsorship/work authorisation, consent) are
  filled only with answers you entered and explicitly marked as reviewed.
- Run it locally in a visible browser: `python manage.py assist_apply <application_id>`. When you close the window it
  asks whether you submitted, and if you did, it records the submission (`browser_assisted`).

## Politeness and reliability

- Each source has its own throttle: a minimum interval between requests (default 1 s), coordinated through Redis.
- The sync interval is at least 60 minutes, with at most one sync per source per 30 minutes across all users.
- `429`/`5xx`/timeouts are retried with exponential back-off and jitter, and `Retry-After` is honoured.
  Non-retryable `4xx` responses fail fast.
- The user agent identifies JobPilot (`SOURCE_USER_AGENT`; include a contact address in production).

## Adding a new source

1. Verify the source's **current** official documentation, access requirements and terms of use.
2. Subclass `apps.jobs.adapters.base.JobSourceAdapter`:
   - set `source_type` and `capabilities` (including limitations and the documentation URL);
   - implement `fetch_listings()` (complete set of published jobs) and `fetch_job()`;
   - optionally implement `fetch_application_questions()` and `submit_application()` (only for documented, authorised APIs).
   - Return `NormalizedJob(...).finalize()` so normalisation stays consistent.
3. Register it in `apps/jobs/adapters/registry.py` and add the choice to `JobSource.SourceType` (plus a migration).
4. Add fixture-based tests (see `apps/jobs/tests/test_adapters.py`) — tests must never call the real service.
