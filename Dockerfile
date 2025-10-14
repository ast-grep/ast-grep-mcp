# syntax=docker/dockerfile:1

FROM ghcr.io/astral-sh/uv:python3.12-bookworm AS builder
WORKDIR /app

# Install project dependencies using uv (without dev extras)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# Copy application source
COPY . .

# Final runtime image
FROM python:3.12-slim
WORKDIR /app

# Install system dependencies for ast-grep CLI
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl nodejs npm \
    && rm -rf /var/lib/apt/lists/*

# Install ast-grep CLI globally
RUN npm install -g @ast-grep/cli

# Copy virtual environment from builder stage
COPY --from=builder /app/.venv /app/.venv

# Ensure virtual environment binaries are prioritized
ENV PATH="/app/.venv/bin:$PATH"

# Copy application source code
COPY . .

# Default command runs the MCP server entry point
CMD ["ast-grep-server"]
