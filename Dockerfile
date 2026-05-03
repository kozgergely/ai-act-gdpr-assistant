# syntax=docker/dockerfile:1.7

# ---------- builder ----------
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1

# uv for fast, reproducible installs.
COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /usr/local/bin/uv

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install deps. We copy the package source up front because hatchling reads
# README.md and the src/ tree during the install metadata phase, even when
# only the dependencies have changed.
COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN uv venv /opt/venv && \
    uv pip install --python /opt/venv/bin/python .

# Then copy the rest of the source — kept in a separate layer so iterating
# on scripts/eval/load_test does not re-trigger the dependency install.
COPY scripts/ ./scripts/
COPY eval/ ./eval/
COPY load_test/ ./load_test/
COPY data/raw/fixture.jsonl ./data/raw/fixture.jsonl

# Pre-build indices from the shipped fixture so the container boots usable
# out of the box even without EUR-Lex PDFs. If the user mounts their own
# data/ volume they can run `python scripts/build_indices.py` inside.
RUN /opt/venv/bin/python scripts/build_indices.py

# ---------- runtime ----------
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:${PATH}" \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_FILE_WATCHER_TYPE=none \
    STREAMLIT_LOGGER_LEVEL=warning

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 app

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder --chown=app:app /app/src /app/src
COPY --from=builder --chown=app:app /app/scripts /app/scripts
COPY --from=builder --chown=app:app /app/eval /app/eval
COPY --from=builder --chown=app:app /app/load_test /app/load_test
COPY --from=builder --chown=app:app /app/data /app/data

# The agentic_rag package is already installed (non-editable) into /opt/venv
# during the builder stage, so the imports in src/agentic_rag/ui/app.py
# resolve through the venv's site-packages — no runtime re-install needed.

USER app

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "src/agentic_rag/ui/app.py"]
