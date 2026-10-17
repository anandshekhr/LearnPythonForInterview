# Job sources, capabilities and compliance

JobPilot reads jobs from:
- documented job-board APIs (Greenhouse, Lever);
- the public data behind Workday career sites;
- structured data that companies publish on their own career pages.

It never reads pages that require a login or are otherwise restricted. It obeys robots.txt and never bypasses
CAPTCHA, MFA, rate limits, HTTP 403 blocks or other access controls.

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

### Workday career sites (`workday`)
- **Identifier:** the career site URL, e.g. `https://acme.wd5.myworkdayjobs.com/AcmeCareers`. Open the company's
  Careers page; if it goes to `…myworkdayjobs.com/…`, copy that address (a locale like `/en-US/` is fine).
- **How it reads jobs:** Workday has **no documented public jobs API**. Every Workday career site loads its
  listings anonymously from a JSON endpoint (`/wday/cxs/{tenant}/{site}/jobs`). JobPilot calls that same
  endpoint, read-only:
  - it checks robots.txt first;
  - it waits at least 1.5 s between requests;
  - it downloads a job's details only when the job is new or was last refreshed more than 7 days ago.
- **When access is refused:** some employers block automated access (HTTP 403). JobPilot then marks the source
  as failing and **stops**; it never tries to get around the block.
- **Options:**
  - `search_text`: keywords that narrow large employers, e.g. `python`.
  - `max_jobs`: the most jobs read per sync (default 300, at most 2000).
- **Capped listings:** if `max_jobs` cuts a listing short, unseen jobs are never expired. A job is expired only
  when its page returns 404 or 410.
- **Applying:** Workday applications need a candidate account on the employer's site, so you always apply there.
  JobPilot links to the posting; there is no API submission or browser pre-fill.
- **Stability:** the endpoint is undocumented and may change. Parsing is defensive, and errors appear on the
  source instead of producing bad data.

### Company career pages (`career_page`)
- **Identifier:** the company's careers listing URL, e.g. `https://www.acme.com/careers`.
- **What it reads:** only [schema.org `JobPosting`](https://schema.org/JobPosting) structured data (JSON-LD). This
  is what sites publish so their jobs appear in search engines' job results. Pages without it are skipped and
  counted in a warning.
- **How it finds jobs:**
  1. It reads job postings embedded on the listing page.
  2. It follows same-site links that look like job pages, or links matching your `job_link_pattern` regex.
  3. If neither finds anything, it falls back to the sitemap(s) listed in robots.txt, or `/sitemap.xml`.
- **Options:**
  - `job_link_pattern`: a regex that selects job links.
  - `max_jobs`: the most jobs read per sync (default 100, at most 500).
  - `render_js`: for careers pages built entirely with JavaScript. This runs on the browser worker, and every
    request the page makes goes through the same public-host check.
- **Already an ATS?** If the careers page links to Greenhouse, Lever or Workday, JobPilot shows a warning
  suggesting you add that board instead. If the URL itself is one of those boards, it is rejected with the same
  advice.
- **Safety rules:**
  - robots.txt is always obeyed. If robots.txt can't be fetched because of a server error, nothing is crawled.
  - The minimum interval between requests is 2 s.
  - Pages are capped at 5 MB.
  - Only public hosts on ports 80/443 are allowed. Private, loopback, link-local (cloud metadata) and other
    non-public addresses are refused, including after redirects.
- **Expiry:** a job is expired only when its page returns 404 or 410, redirects elsewhere, or its `validThrough`
  date has passed.
- **Applying:** applications happen on the company's own site.

### Sources evaluated but not integrated
- **Ashby** (`api.ashbyhq.com/posting-api/job-board/{name}`): appears to offer a public listing endpoint, but its
  official documentation could not be retrieved, and third-party descriptions disagree. It is not integrated
  until verified. (Ashby boards that publish JSON-LD can still be added as a `career_page`.)
- **LinkedIn, Naukri, Indeed and similar job portals:** their terms don't allow automated access without a
  partnership, so there are no adapters.
- **SmartRecruiters, iCIMS and others:** possible future adapters, each needing separate verification of its
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
