# workers/selfhost

Node 24 entrypoint that runs the existing `fantasy-mcp`, `espn-client`, `sleeper-client`
and `yahoo-client` Hono apps in one process for self-hosted (Docker / Raspberry Pi)
deployments. It is not deployed to Cloudflare.

- Replaces Cloudflare service bindings with in-process fetch dispatch.
- Replaces `auth-worker` with a local stub (`src/local-auth.ts`) that validates a single
  static bearer token (`FLAIM_MCP_TOKEN`) and serves leagues from `config/leagues.json`.
- Replaces KV with a file-backed store (`src/file-kv.ts`) under `FLAIM_CACHE_DIR`.
- Exposes `GET /health`, `GET /healthz` and the MCP Streamable HTTP endpoint at `/mcp` on
  port `8790`.

```bash
corepack pnpm --dir workers/selfhost type-check
corepack pnpm --dir workers/selfhost test
corepack pnpm --dir workers/selfhost build      # dist/server.mjs (esbuild)
FLAIM_MCP_TOKEN=... FLAIM_LEAGUES_FILE=../../config/leagues.json corepack pnpm --dir workers/selfhost dev
```

Configuration, tunnel setup and league examples: `../../docs/SELF-HOSTING.md`.
