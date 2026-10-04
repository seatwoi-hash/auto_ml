FROM python:3.11-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1

WORKDIR /app
RUN pip install poetry==2.5.1
COPY pyproject.toml poetry.lock ./
RUN poetry install --only main --no-root


FROM python:3.11-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:${PATH}"

RUN groupadd --system app \
    && useradd --system --gid app --create-home app

WORKDIR /app
COPY --from=builder /app/.venv /app/.venv

COPY --chown=app:app app.py nextcloud_service.py schemas.py service_ml.py settings.py index.html ./
RUN mkdir -p /app/uploads && chown app:app /app/uploads

USER app
EXPOSE 8877

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8877", "--no-access-log"]
