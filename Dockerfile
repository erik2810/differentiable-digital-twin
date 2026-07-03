# syntax=docker/dockerfile:1.7
#
# Multi-stage build for the Differentiable Digital Twin training framework.
#   * builder  -- resolves the locked environment (CPU-only PyTorch) with uv
#   * runtime  -- a slim, non-root image carrying only the venv + source
#
# Build:  docker build -t ddtwin .
# Run:    docker run --rm -v "$PWD/out:/app/web/data" ddtwin optimise --float32
#         (the optimised trajectory lands in ./out for the web player)

# --------------------------------------------------------------------- builder
FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# 1. Install dependencies from the lockfile first -- this layer is cached until
#    pyproject.toml / uv.lock change, so source edits don't re-download PyTorch.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

# 2. Install the project itself as a non-editable wheel (so runtime needs no src).
COPY README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable

# --------------------------------------------------------------------- runtime
FROM python:3.11-slim-bookworm AS runtime

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    OMP_NUM_THREADS=4

WORKDIR /app

# Run as an unprivileged user.
RUN groupadd --system app && useradd --system --gid app --home /app app

# The resolved virtual environment (CPU PyTorch) and the web player assets.
COPY --from=builder --chown=app:app /app/.venv /app/.venv
COPY --chown=app:app web ./web

USER app

# `ddtwin` is the console entry point; default to a full optimise + export run.
ENTRYPOINT ["ddtwin"]
CMD ["optimise", "--float32"]
