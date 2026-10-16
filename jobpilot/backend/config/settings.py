"""Django settings for JobPilot.

All secrets and deployment-specific values come from environment variables.
See ../.env.example for the full list.
"""
from datetime import timedelta
from pathlib import Path
import os

from celery.schedules import crontab

BASE_DIR = Path(__file__).resolve().parent.parent


def _load_dotenv(path):
    """Minimal .env loader for local development; real environment variables win."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv(BASE_DIR / ".env")


def env(name, default=None):
    return os.environ.get(name, default)


def env_bool(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def env_list(name, default=""):
    return [item.strip() for item in env(name, default).split(",") if item.strip()]


TESTING = env_bool("JOBPILOT_TESTING")
DEBUG = env_bool("DJANGO_DEBUG", False)
SECRET_KEY = env("DJANGO_SECRET_KEY") or ("insecure-dev-key" if (DEBUG or TESTING) else None)
if not SECRET_KEY:
    raise RuntimeError("DJANGO_SECRET_KEY must be set when DJANGO_DEBUG is off")

ALLOWED_HOSTS = env_list("DJANGO_ALLOWED_HOSTS", "localhost,127.0.0.1")
CSRF_TRUSTED_ORIGINS = env_list("DJANGO_CSRF_TRUSTED_ORIGINS", "")

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "rest_framework",
    "rest_framework_simplejwt",
    "rest_framework_simplejwt.token_blacklist",
    "django_filters",
    "drf_spectacular",
    "corsheaders",
    "apps.core",
    "apps.accounts",
    "apps.profiles",
    "apps.jobs",
    "apps.matching",
    "apps.documents",
    "apps.applications",
    "apps.automation",
    "apps.notifications",
    "apps.analytics",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "whitenoise.middleware.WhiteNoiseMiddleware",  # serves /static/ (admin, Swagger) without a separate web server
    "corsheaders.middleware.CorsMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
    "apps.core.middleware.RequestIdMiddleware",
]

ROOT_URLCONF = "config.urls"
WSGI_APPLICATION = "config.wsgi.application"
AUTH_USER_MODEL = "accounts.User"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ]
        },
    }
]

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.postgresql",
        "NAME": env("POSTGRES_DB", "jobpilot"),
        "USER": env("POSTGRES_USER", "jobpilot"),
        "PASSWORD": env("POSTGRES_PASSWORD", "jobpilot"),
        "HOST": env("POSTGRES_HOST", "localhost"),
        "PORT": env("POSTGRES_PORT", "5432"),
        "CONN_MAX_AGE": int(env("POSTGRES_CONN_MAX_AGE", "60")),
    }
}

REDIS_URL = env("REDIS_URL", "redis://localhost:6379/0")
CACHES = {
    "default": (
        {"BACKEND": "django.core.cache.backends.locmem.LocMemCache"}
        if TESTING
        else {"BACKEND": "django.core.cache.backends.redis.RedisCache", "LOCATION": REDIS_URL}
    )
}

AUTH_PASSWORD_VALIDATORS = [
    {"NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator"},
    {"NAME": "django.contrib.auth.password_validation.MinimumLengthValidator", "OPTIONS": {"min_length": 10}},
    {"NAME": "django.contrib.auth.password_validation.CommonPasswordValidator"},
    {"NAME": "django.contrib.auth.password_validation.NumericPasswordValidator"},
]

LANGUAGE_CODE = "en-us"
TIME_ZONE = env("TIME_ZONE", "Asia/Kolkata")
USE_I18N = True
USE_TZ = True

STATIC_URL = "/static/"
STATIC_ROOT = BASE_DIR / "staticfiles"
STORAGES = {
    "default": {"BACKEND": "django.core.files.storage.FileSystemStorage"},
    "staticfiles": {"BACKEND": "whitenoise.storage.CompressedManifestStaticFilesStorage"
                    if not (DEBUG or TESTING) else "django.contrib.staticfiles.storage.StaticFilesStorage"},
}
# Uploaded resumes and generated documents. Never served publicly: downloads go
# through authenticated API views that check ownership.
PRIVATE_MEDIA_ROOT = Path(env("PRIVATE_MEDIA_ROOT", str(BASE_DIR / "private_media")))
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# --- Security -----------------------------------------------------------------
SECURE_CONTENT_TYPE_NOSNIFF = True
X_FRAME_OPTIONS = "DENY"
SESSION_COOKIE_SECURE = not DEBUG and not TESTING
CSRF_COOKIE_SECURE = not DEBUG and not TESTING
SECURE_PROXY_SSL_HEADER = ("HTTP_X_FORWARDED_PROTO", "https")
SECURE_SSL_REDIRECT = env_bool("DJANGO_SECURE_SSL_REDIRECT", False)
SECURE_HSTS_SECONDS = int(env("DJANGO_HSTS_SECONDS", "0"))
CORS_ALLOWED_ORIGINS = env_list("CORS_ALLOWED_ORIGINS", "http://localhost:4200")
DATA_UPLOAD_MAX_MEMORY_SIZE = 12 * 1024 * 1024
FILE_UPLOAD_MAX_MEMORY_SIZE = 5 * 1024 * 1024
RESUME_MAX_UPLOAD_BYTES = int(env("RESUME_MAX_UPLOAD_BYTES", str(5 * 1024 * 1024)))

# --- REST framework -----------------------------------------------------------
REST_FRAMEWORK = {
    "DEFAULT_AUTHENTICATION_CLASSES": ["rest_framework_simplejwt.authentication.JWTAuthentication"],
    "DEFAULT_PERMISSION_CLASSES": ["rest_framework.permissions.IsAuthenticated"],
    "DEFAULT_FILTER_BACKENDS": [
        "django_filters.rest_framework.DjangoFilterBackend",
        "rest_framework.filters.SearchFilter",
        "rest_framework.filters.OrderingFilter",
    ],
    "DEFAULT_PAGINATION_CLASS": "apps.core.pagination.StandardPagination",
    "PAGE_SIZE": 20,
    "DEFAULT_SCHEMA_CLASS": "drf_spectacular.openapi.AutoSchema",
    "DEFAULT_THROTTLE_CLASSES": [
        "rest_framework.throttling.UserRateThrottle",
        "rest_framework.throttling.AnonRateThrottle",
        "rest_framework.throttling.ScopedRateThrottle",
    ],
    "DEFAULT_THROTTLE_RATES": {
        "user": env("THROTTLE_USER", "2000/hour"),
        "anon": env("THROTTLE_ANON", "60/hour"),
        "auth": env("THROTTLE_AUTH", "20/minute"),
        "upload": env("THROTTLE_UPLOAD", "30/hour"),
        "submit": env("THROTTLE_SUBMIT", "30/hour"),
        "sync": env("THROTTLE_SYNC", "30/hour"),
    },
    "EXCEPTION_HANDLER": "apps.core.exceptions.exception_handler",
}

SIMPLE_JWT = {
    "ACCESS_TOKEN_LIFETIME": timedelta(minutes=int(env("JWT_ACCESS_MINUTES", "30"))),
    "REFRESH_TOKEN_LIFETIME": timedelta(days=int(env("JWT_REFRESH_DAYS", "7"))),
    "ROTATE_REFRESH_TOKENS": True,
    "BLACKLIST_AFTER_ROTATION": True,
    "AUTH_HEADER_TYPES": ("Bearer",),
}

SPECTACULAR_SETTINGS = {
    "TITLE": "JobPilot API",
    "DESCRIPTION": "AI-assisted job discovery, matching and application management.",
    "VERSION": "1.0.0",
    "SERVE_INCLUDE_SCHEMA": False,
    "COMPONENT_SPLIT_REQUEST": True,
}

# --- Celery -------------------------------------------------------------------
CELERY_BROKER_URL = env("CELERY_BROKER_URL", REDIS_URL)
CELERY_RESULT_BACKEND = env("CELERY_RESULT_BACKEND", REDIS_URL)
CELERY_TASK_ALWAYS_EAGER = env_bool("CELERY_TASK_ALWAYS_EAGER", TESTING)
CELERY_TASK_EAGER_PROPAGATES = TESTING
CELERY_TASK_ACKS_LATE = True
CELERY_TASK_REJECT_ON_WORKER_LOST = True
CELERY_WORKER_PREFETCH_MULTIPLIER = 1
CELERY_TASK_TIME_LIMIT = 15 * 60
CELERY_TASK_SOFT_TIME_LIMIT = 12 * 60
CELERY_TIMEZONE = TIME_ZONE
CELERY_TASK_ROUTES = {
    "apps.applications.tasks.run_browser_assist": {"queue": "browser"},
}
CELERY_BEAT_SCHEDULE = {
    "dispatch-scheduled-searches": {
        "task": "apps.automation.tasks.dispatch_due_searches",
        "schedule": crontab(minute="*/15"),
    },
    "check-source-health": {
        "task": "apps.jobs.tasks.check_sources_health",
        "schedule": crontab(minute=5, hour="*/6"),
    },
    "expire-stale-jobs": {
        "task": "apps.jobs.tasks.expire_stale_jobs",
        "schedule": crontab(minute=20, hour=3),
    },
    "send-reminders": {
        "task": "apps.notifications.tasks.send_due_reminders",
        "schedule": crontab(minute=0, hour="*"),
    },
    "recover-interrupted-runs": {
        "task": "apps.automation.tasks.recover_interrupted_runs",
        "schedule": crontab(minute="*/30"),
    },
    "purge-expired-data": {
        "task": "apps.automation.tasks.purge_expired_data",
        "schedule": crontab(minute=40, hour=4),
    },
}

# --- AI providers -------------------------------------------------------------
# LLM_PROVIDER: "none" (deterministic, rule-based only), "anthropic", or "openai"
# (any OpenAI-compatible chat completions endpoint).
LLM_PROVIDER = env("LLM_PROVIDER", "none")
LLM_MODEL = env("LLM_MODEL", "")
LLM_API_KEY = env("LLM_API_KEY", "")
LLM_BASE_URL = env("LLM_BASE_URL", "")
LLM_TIMEOUT_SECONDS = float(env("LLM_TIMEOUT_SECONDS", "60"))
# EMBEDDING_PROVIDER: "hashing" (local, no network) or "openai" (OpenAI-compatible).
EMBEDDING_PROVIDER = env("EMBEDDING_PROVIDER", "hashing")
EMBEDDING_MODEL = env("EMBEDDING_MODEL", "")
EMBEDDING_API_KEY = env("EMBEDDING_API_KEY", "")
EMBEDDING_BASE_URL = env("EMBEDDING_BASE_URL", "")
EMBEDDING_DIMENSIONS = 384  # fixed by the database column; providers must project to this size

# --- Job sources / automation -------------------------------------------------
SOURCE_HTTP_TIMEOUT_SECONDS = float(env("SOURCE_HTTP_TIMEOUT_SECONDS", "20"))
SOURCE_HTTP_BACKOFF_SECONDS = float(env("SOURCE_HTTP_BACKOFF_SECONDS", "1.0"))
SOURCE_THROTTLE_ENABLED = env_bool("SOURCE_THROTTLE_ENABLED", True)
SOURCE_USER_AGENT = env("SOURCE_USER_AGENT", "JobPilot/1.0 (personal job search assistant)")
JOB_EXPIRY_GRACE_DAYS = int(env("JOB_EXPIRY_GRACE_DAYS", "3"))
# Development/testing only: point adapters at a local server that mimics the documented APIs,
# e.g. SOURCE_API_BASE_OVERRIDES=greenhouse=http://127.0.0.1:8765/v1/boards,lever=http://127.0.0.1:8765/v0/postings
SOURCE_API_BASE_OVERRIDES = dict(
    item.split("=", 1) for item in env_list("SOURCE_API_BASE_OVERRIDES", "") if "=" in item
)
if SOURCE_API_BASE_OVERRIDES and not (DEBUG or TESTING):
    raise RuntimeError("SOURCE_API_BASE_OVERRIDES is only allowed with DJANGO_DEBUG=1")
BROWSER_AUTOMATION_ENABLED = env_bool("BROWSER_AUTOMATION_ENABLED", False)
BROWSER_HEADLESS = env_bool("BROWSER_HEADLESS", True)
BROWSER_EXECUTABLE_PATH = env("BROWSER_EXECUTABLE_PATH", "")
# Hosted application forms that the assisted-fill helper may open. Nothing else is automated.
BROWSER_ASSIST_ALLOWED_HOSTS = env_list(
    "BROWSER_ASSIST_ALLOWED_HOSTS", "boards.greenhouse.io,job-boards.greenhouse.io,jobs.lever.co,jobs.eu.lever.co"
)
# Fernet key (urlsafe base64, 32 bytes) used to encrypt stored files and integration credentials.
DATA_ENCRYPTION_KEY = env("DATA_ENCRYPTION_KEY", "")
ENCRYPT_FILES_AT_REST = env_bool("ENCRYPT_FILES_AT_REST", True)
FRONTEND_URL = env("FRONTEND_URL", "http://localhost:4200")

# --- Logging ------------------------------------------------------------------
LOG_LEVEL = env("LOG_LEVEL", "INFO")
LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "filters": {"request_id": {"()": "apps.core.logging.RequestIdFilter"}},
    "formatters": {
        "json": {
            "()": "pythonjsonlogger.json.JsonFormatter",
            "fmt": "%(asctime)s %(levelname)s %(name)s %(message)s %(request_id)s",
        }
    },
    "handlers": {
        "console": {"class": "logging.StreamHandler", "formatter": "json", "filters": ["request_id"]}
    },
    "root": {"handlers": ["console"], "level": LOG_LEVEL},
    "loggers": {
        "django.db.backends": {"level": "WARNING"},
        "httpx": {"level": "WARNING"},
    },
}
