# REST API (v1)

Base path `/api/v1/`. Interactive OpenAPI docs: `/api/docs/` (Swagger UI), raw schema: `/api/schema/`.

**Conventions**
- Auth: `Authorization: Bearer <access>` (JWT). Obtain via `POST /auth/login/` or `POST /auth/register/`; rotate via `POST /auth/refresh/`.
- Pagination: `?page=` and `?page_size=` (max 100) → `{count, next, previous, results}`.
- Filtering, search and ordering: documented per endpoint (`?search=`, `?ordering=-field`, filter fields).
- Errors: domain errors use `{"error": {"code", "message", "details"}, "request_id"}`; validation errors use DRF's
  field map. Every response carries `X-Request-ID`.
- Objects owned by other users return `404`. Conflicts (duplicates, invalid state) return `409`. Limits return `429`.
- Submission: `POST /applications/{id}/submit/` **requires** an `Idempotency-Key` header. Replaying the same key
  returns the existing result without submitting again.

| Area | Endpoints |
|---|---|
| Auth | `POST auth/register/`, `auth/login/`, `auth/refresh/`, `auth/logout/`; `GET/PATCH/DELETE auth/me/` (delete = erase account and files) |
| Profile | `GET/PATCH profile/`, `GET/PATCH profile/preferences/`; CRUD `profile/skills/`, `profile/employment/`, `profile/education/`, `profile/projects/`, `profile/certifications/` |
| Resumes | `GET/POST resumes/` (multipart upload), `GET/DELETE resumes/{id}/`, `POST resumes/{id}/parse/`, `POST resumes/{id}/apply/`, `GET resumes/{id}/download/` |
| Resume versions | `GET resume-versions/`, `GET/PATCH resume-versions/{id}/` (validated edit), `POST …/approve/`, `POST …/reject/`, `GET …/compare/`, `POST resume-versions/snapshot/` |
| Companies & sources | CRUD `companies/`; CRUD `sources/`, `GET sources/types/`, `POST sources/{id}/sync/`, `POST sources/{id}/test/`, `POST sources/{id}/submission-credentials/` (staff) |
| Jobs | `GET jobs/` (filters: `q`, `status`, `company`, `source`, `work_model`, `employment_type`, `seniority`, `location`, `skill`, `min_score`, `category`, `published_after`, `seen_after`, `include_duplicates`), `GET jobs/{id}/`, `GET jobs/{id}/changes/` |
| Matching | `GET matches/` (filters: `category`, `user_state`, `job`, `min_score`, `job_status`, `work_model`, `company`, `hide_excluded`), `GET/PATCH matches/{id}/` (`user_state`), `POST matches/rematch/`, `GET matches/semantic/`; `GET/PATCH scoring-config/` |
| Documents | `POST documents/generate/`, `GET documents/`, `GET/PATCH/DELETE documents/{id}/` (cover letter text), `GET documents/{id}/download/`, `POST documents/{id}/approve/` |
| Applications | `GET/POST applications/`, `GET/PATCH/DELETE applications/{id}/`, `POST …/status/`, `POST …/attach-resume/`, `POST …/prepare-answers/`, `PATCH …/answers/`, `GET …/kit/`, `POST …/submit/`, `POST …/confirm-submitted/` |
| Interviews & follow-ups | CRUD `interviews/` (filters `start`, `end`, `outcome`), CRUD `follow-ups/` |
| Automation | `GET/PATCH automation/settings/` (automated mode requires `automated_submission_consent: true`), `GET automation/runs/`, CRUD `scheduled-searches/`, `POST scheduled-searches/{id}/run-now/` |
| Notifications | `GET notifications/` (`?unread=1`), `GET notifications/unread-count/`, `POST notifications/{id}/read/`, `POST notifications/read-all/` |
| Analytics | `GET analytics/overview/`, `GET analytics/timeseries/?weeks=`, `GET analytics/breakdowns/`, `GET analytics/applications.csv` |
| Ops | `GET /health/` (database + cache) |

## Example: discovery-to-approval with curl

```bash
API=http://localhost:8080/api/v1
TOKEN=$(curl -s $API/auth/login/ -H 'Content-Type: application/json' -d '{"email":"me@example.com","password":"…"}' | jq -r .access)
H="Authorization: Bearer $TOKEN"
curl -s -H "$H" -F file=@resume.pdf $API/resumes/                                  # upload → parsed in background
curl -s -H "$H" $API/resumes/1/ | jq .parsed_data                                   # review the draft
curl -s -H "$H" -X POST $API/resumes/1/apply/ -H 'Content-Type: application/json' -d '{}'
curl -s -H "$H" -X POST $API/companies/ -H 'Content-Type: application/json' -d '{"name":"Acme"}'
curl -s -H "$H" -X POST $API/sources/ -H 'Content-Type: application/json' -d '{"company":1,"source_type":"greenhouse","identifier":"acme"}'
curl -s -H "$H" "$API/matches/?category=excellent" | jq '.results[] | {score, title: .job.title, explanation}'
curl -s -H "$H" -X POST $API/documents/generate/ -H 'Content-Type: application/json' -d '{"job": 42}'
curl -s -H "$H" -X POST $API/resume-versions/7/approve/
curl -s -H "$H" $API/applications/?job=42 | jq '.results[0] | {id, status, job_url}'
```
