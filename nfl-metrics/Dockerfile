# syntax=docker/dockerfile:1.7
#
# flaim nfl-metrics: FastMCP server (nflverse metrics + GM tools) over SSE.
# Multi-arch: docker buildx build --platform linux/arm64,linux/amd64 -t flaim-nfl-metrics nfl-metrics
# polars/scipy/numpy/pyarrow all ship aarch64 manylinux wheels, so no compiler is needed.

ARG PYTHON_VERSION=3.12

FROM python:${PYTHON_VERSION}-slim-bookworm AS build
WORKDIR /src
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1
COPY pyproject.toml README.md ./
COPY nfl_metrics ./nfl_metrics
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m venv /opt/venv && /opt/venv/bin/pip install --only-binary=:all: .

FROM python:${PYTHON_VERSION}-slim-bookworm AS runtime
LABEL org.opencontainers.image.title="flaim-nfl-metrics" \
      org.opencontainers.image.description="nflverse efficiency/usage metrics + autonomous GM tools as an MCP server (SSE / Streamable HTTP)" \
      org.opencontainers.image.source="https://github.com/acyounk28/flaim"

ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PORT=8800 \
    HOST=0.0.0.0 \
    NFL_MCP_TRANSPORT=sse \
    NFL_DATA_DIR=/data/nfl \
    NFLREADPY_CACHE_DIR=/data/nfl/nflreadpy \
    POLARS_MAX_THREADS=3

RUN apt-get update \
 && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/* \
 && useradd --system --uid 10001 --create-home nfl \
 && mkdir -p /data/nfl && chown -R nfl:nfl /data

COPY --from=build /opt/venv /opt/venv
USER nfl
WORKDIR /home/nfl
VOLUME ["/data"]
EXPOSE 8800

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD curl -fsS "http://127.0.0.1:${PORT}/health" || exit 1

CMD ["nfl-metrics-mcp"]
