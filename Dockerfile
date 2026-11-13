
# ---------- Stage 1: Build dependencies ----------
FROM python:3.11-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /build

# Build tools are required only when dependencies need compilation.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy dependencies first to maximize Docker layer caching.
COPY requirements.txt .

# Build wheels for offline installation in the runtime image.
RUN pip wheel --wheel-dir=/wheels -r requirements.txt


# ---------- Stage 2: Runtime ----------
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONHASHSEED=random \
    PORT=8000 \
    MODEL_ARTIFACTS_DIR=/app/artifacts \
    HF_HOME=/tmp/huggingface \
    TORCH_HOME=/tmp/torch

WORKDIR /app

# Runtime libraries only; compiler tools stay in the builder stage.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies from locally built wheels.
COPY --from=builder /wheels /wheels
COPY requirements.txt .

RUN pip install --no-index --find-links=/wheels -r requirements.txt \
    && rm -rf /wheels

# Create an unprivileged application user.
RUN groupadd --gid 10001 appuser \
    && useradd --uid 10001 --gid 10001 --create-home --shell /usr/sbin/nologin appuser \
    && mkdir -p /app/artifacts /app/data \
    && chown -R appuser:appuser /app

# Copy application source after dependencies for better caching.
COPY --chown=appuser:appuser . .

USER 10001:10001

EXPOSE 8000

# Compose overrides this for each service.
CMD ["python", "main.py"]
