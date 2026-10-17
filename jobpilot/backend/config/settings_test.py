import os

os.environ["JOBPILOT_TESTING"] = "1"
os.environ.setdefault("CELERY_TASK_ALWAYS_EAGER", "1")
os.environ.setdefault("LLM_PROVIDER", "none")
os.environ.setdefault("EMBEDDING_PROVIDER", "hashing")

from .settings import *  # noqa: E402,F401,F403

PASSWORD_HASHERS = ["django.contrib.auth.hashers.MD5PasswordHasher"]
REST_FRAMEWORK = {**REST_FRAMEWORK, "DEFAULT_THROTTLE_CLASSES": []}  # noqa: F405
LLM_PROVIDER = "none"
EMBEDDING_PROVIDER = "hashing"
SOURCE_HTTP_BACKOFF_SECONDS = 0
SOURCE_THROTTLE_ENABLED = False
SOURCE_ALLOW_PRIVATE_HOSTS = True  # tests use fake hosts; SSRF tests switch this off
