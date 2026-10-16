# Operations

## Deployment (Docker Compose)

```bash
cp .env.example .env           # set DJANGO_SECRET_KEY, DATA_ENCRYPTION_KEY, POSTGRES_PASSWORD, hosts/origins
docker compose up -d --build   # db, redis, migrate (one-shot), api, worker, beat, web
docker compose ps              # all services should be healthy
docker compose --profile browser up -d browser-worker   # optional
```

| Service | Image | Notes |
|---|---|---|
| `db` | `pgvector/pgvector:pg16` | Data in volume `pgdata`; the `migrate` service creates the `vector` extension |
| `redis` | `redis:7-alpine` | AOF persistence, 128 MB cap, `noeviction` (it is the Celery broker) |
| `migrate` | backend | Runs `migrate` once; other backend services start only after it succeeds |
| `api` | backend | Gunicorn (2 workers × 4 threads), non-root, health-checked on `/health/` |
| `worker` | backend | Celery default queue, concurrency 2, `max-tasks-per-child` 200 |
| `beat` | backend | Celery Beat (exactly one instance!) |
| `browser-worker` | `Dockerfile.browser` | Optional `browser` queue; Chromium installed via Playwright |
| `web` | frontend | Nginx serving the Angular build, proxying `/api`, `/admin`, `/static`, `/health`; strict CSP |

Resource footprint at rest is roughly 600–800 MB RAM for the whole stack. Browser automation and AI calls run only in
workers, never in API requests.

### Production checklist
- Put TLS in front of `web` (a load balancer, Caddy or Traefik). Set `DJANGO_SECURE_SSL_REDIRECT=1`,
  `DJANGO_HSTS_SECONDS=31536000`, and `DJANGO_ALLOWED_HOSTS` / `DJANGO_CSRF_TRUSTED_ORIGINS` / `CORS_ALLOWED_ORIGINS`
  for your domain.
- Generate unique `DJANGO_SECRET_KEY` and `DATA_ENCRYPTION_KEY`, and keep them in your secret manager. **Losing
  `DATA_ENCRYPTION_KEY` makes stored files and employer keys unreadable.**
- Set `SOURCE_USER_AGENT` to include a contact address.
- Optional AI: `LLM_PROVIDER=anthropic` with `LLM_API_KEY` (default model `claude-opus-5-5`), or an OpenAI-compatible
  endpoint.
- Create an admin: `docker compose exec api python manage.py createsuperuser`.
- Never set `SOURCE_API_BASE_OVERRIDES` in production (it is refused unless `DJANGO_DEBUG=1`).

## Health, logging and monitoring
- `GET /health/` checks the database and cache (`200` ok / `503` degraded). Nginx has `/healthz`.
- Logs are JSON on stdout (`docker compose logs -f api worker beat`), with `request_id` correlated through the
  `X-Request-ID` header.
- Background work is recorded in `AutomationRun` (UI: *Automation settings → Execution history*; admin: *Automation
  runs*). Sources that fail three times in a row move to `error`, and their owner is notified.

## Backups and recovery

Back up **three things**: the database, the private media volume, and the secrets (`.env`).

```bash
# Database (daily; keep ≥ 7 days)
docker compose exec -T db pg_dump -U "$POSTGRES_USER" -Fc "$POSTGRES_DB" > backups/jobpilot-$(date +%F).dump
# Encrypted files (resumes, generated documents)
docker run --rm -v jobpilot_private_media:/data -v "$PWD/backups:/backup" alpine \
  tar czf /backup/media-$(date +%F).tgz -C /data .
```

Restore:
```bash
docker compose stop api worker beat
docker compose exec -T db pg_restore -U "$POSTGRES_USER" -d "$POSTGRES_DB" --clean --if-exists < backups/jobpilot-YYYY-MM-DD.dump
docker run --rm -v jobpilot_private_media:/data -v "$PWD/backups:/backup" alpine sh -c "rm -rf /data/* && tar xzf /backup/media-YYYY-MM-DD.tgz -C /data"
docker compose up -d
```
Restore with the **same** `DATA_ENCRYPTION_KEY`. Test restores regularly on a staging copy.

Redis holds only the queue and cache. If it's lost, in-flight tasks are retried or recovered:
`recover_interrupted_runs` releases stuck submissions as *unconfirmed* so that you can check them before retrying.

## Upgrades
```bash
git pull && docker compose build && docker compose up -d   # migrate runs automatically before api/worker/beat
```

## Data protection
- Per-user retention: *Career profile → Delete closed data after (days)*. A daily task purges old generated documents
  and terminal applications.
- Account deletion (`DELETE /api/v1/auth/me/` with your password) removes all of the user's rows and files.
- The audit log (`/admin` → Audit logs) records security-relevant actions without secrets or resume contents.
