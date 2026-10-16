FROM python:3.13-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
RUN useradd --create-home --uid 10001 app && mkdir -p /app/private_media /app/staticfiles && chown -R app /app/private_media /app/staticfiles
USER app
RUN DJANGO_SECRET_KEY=build-only DJANGO_DEBUG=0 python manage.py collectstatic --noinput >/dev/null
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health/', timeout=4)"
CMD ["gunicorn", "config.wsgi:application", "--bind", "0.0.0.0:8000", "--workers", "2", "--threads", "4", "--timeout", "60", "--access-logfile", "-"]
