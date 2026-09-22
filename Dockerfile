# syntax=docker/dockerfile:1.7
#
# Flaim self-hosted MCP gateway (fantasy-mcp + espn/sleeper/yahoo clients in one
# Node process). Multi-arch: build with
#   docker buildx build --platform linux/arm64,linux/amd64 -t flaim-mcp .
# See docs/SELF-HOSTING.md.

ARG NODE_VERSION=24

# ---------- build stage ----------
FROM --platform=$BUILDPLATFORM node:${NODE_VERSION}-bookworm-slim AS build
WORKDIR /app
ENV COREPACK_ENABLE_DOWNLOAD_PROMPT=0 \
    PNPM_HOME=/root/.local/share/pnpm

RUN corepack enable

# Workspace manifests first so dependency install is cached.
COPY package.json pnpm-lock.yaml pnpm-workspace.yaml ./
COPY workers/shared/package.json workers/shared/
COPY workers/fantasy-mcp/package.json workers/fantasy-mcp/
COPY workers/espn-client/package.json workers/espn-client/
COPY workers/sleeper-client/package.json workers/sleeper-client/
COPY workers/yahoo-client/package.json workers/yahoo-client/
COPY workers/auth-worker/package.json workers/auth-worker/
COPY workers/selfhost/package.json workers/selfhost/
COPY web/package.json web/

RUN --mount=type=cache,target=/root/.local/share/pnpm/store \
    corepack pnpm install --frozen-lockfile --ignore-scripts \
      --filter flaim-selfhost... --filter fantasy-mcp... --filter espn-client... \
      --filter sleeper-client... --filter yahoo-client...

COPY workers/shared workers/shared
COPY workers/fantasy-mcp workers/fantasy-mcp
COPY workers/espn-client workers/espn-client
COPY workers/sleeper-client workers/sleeper-client
COPY workers/yahoo-client workers/yahoo-client
COPY workers/selfhost workers/selfhost

RUN corepack pnpm --dir workers/selfhost build

# ---------- runtime stage ----------
FROM node:${NODE_VERSION}-bookworm-slim AS runtime
LABEL org.opencontainers.image.title="flaim-mcp" \
      org.opencontainers.image.description="Self-hosted Flaim fantasy MCP gateway (Streamable HTTP / SSE)" \
      org.opencontainers.image.source="https://github.com/acyounk28/flaim"

ENV NODE_ENV=production \
    PORT=8790 \
    HOST=0.0.0.0 \
    FLAIM_LEAGUES_FILE=/config/leagues.json \
    FLAIM_CACHE_DIR=/data/cache

RUN apt-get update \
 && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/* \
 && mkdir -p /config /data/cache \
 && chown -R node:node /config /data

WORKDIR /app
COPY --from=build --chown=node:node /app/workers/selfhost/dist/server.mjs ./server.mjs

USER node
EXPOSE 8790
VOLUME ["/data"]

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -fsS http://127.0.0.1:${PORT}/health || exit 1

CMD ["node", "server.mjs"]
