// workers/selfhost/src/server.ts
//
// Runs the Flaim MCP gateway as a single Node process. The Cloudflare
// deployment splits fantasy-mcp, espn-client, sleeper-client, yahoo-client and
// auth-worker into separate Workers connected by service bindings; here each
// Hono app is mounted in-process and exposed to the gateway as a Fetcher-shaped
// object, so the worker source is reused unchanged.
import { serve } from '@hono/node-server';
import fantasyMcp from '../../fantasy-mcp/src/index';
import espnClient from '../../espn-client/src/index';
import sleeperClient from '../../sleeper-client/src/index';
import yahooClient from '../../yahoo-client/src/index';
import type { Env as FantasyEnv } from '../../fantasy-mcp/src/types';
import type { Env as EspnEnv } from '../../espn-client/src/types';
import type { Env as SleeperEnv } from '../../sleeper-client/src/types';
import type { Env as YahooEnv } from '../../yahoo-client/src/types';
import { loadConfig, ConfigError, type SelfhostConfig } from './config';
import { createLocalAuthApp } from './local-auth';
import { FileKV, asKVNamespace } from './file-kv';

type HonoLike<E> = { fetch: (request: Request, env?: E, ctx?: ExecutionContext) => Response | Promise<Response> };

const executionCtx: ExecutionContext = {
  waitUntil(promise: Promise<unknown>) {
    promise.catch((error) => console.warn('[selfhost] background task failed:', error));
  },
  passThroughOnException() {},
  props: {},
} as unknown as ExecutionContext;

function inProcessFetcher<E>(app: HonoLike<E>, env: E): Fetcher {
  return {
    fetch: async (input: RequestInfo | URL, init?: RequestInit) => {
      const request = input instanceof Request ? input : new Request(input, init);
      return app.fetch(request, env, executionCtx);
    },
  } as unknown as Fetcher;
}

const noopRateLimiter: RateLimit = {
  limit: async () => ({ success: true }),
};

export function buildGateway(config: SelfhostConfig, options: { cacheDir: string | null }) {
  const baseEnv = {
    NODE_ENV: 'production',
    ENVIRONMENT: 'selfhost',
    AUTH_WORKER_URL: 'https://auth-worker.internal',
    INTERNAL_SERVICE_TOKEN: config.internalServiceToken,
  };

  const authApp = createLocalAuthApp(config);
  const AUTH_WORKER = inProcessFetcher(authApp, { INTERNAL_SERVICE_TOKEN: config.internalServiceToken });

  const espnEnv: EspnEnv = {
    ...baseEnv,
    AUTH_WORKER,
    ESPN_PLAYERS_CACHE: asKVNamespace(new FileKV(options.cacheDir ? `${options.cacheDir}/espn-players` : null)),
  };
  const sleeperEnv: SleeperEnv = {
    ...baseEnv,
    AUTH_WORKER,
    SLEEPER_PLAYERS_CACHE: asKVNamespace(new FileKV(options.cacheDir ? `${options.cacheDir}/sleeper-players` : null)),
  };
  const yahooEnv: YahooEnv = { ...baseEnv, AUTH_WORKER };

  const gatewayEnv: FantasyEnv = {
    ...baseEnv,
    AUTH_WORKER,
    ESPN: inProcessFetcher(espnClient, espnEnv),
    SLEEPER: inProcessFetcher(sleeperClient, sleeperEnv),
    YAHOO: inProcessFetcher(yahooClient, yahooEnv),
    MCP_RATE_LIMITER: noopRateLimiter,
  };

  return {
    fetch: (request: Request) => fantasyMcp.fetch(request, gatewayEnv, executionCtx),
  };
}

export function createRootHandler(config: SelfhostConfig, options: { cacheDir: string | null }) {
  const gateway = buildGateway(config, options);
  return async (request: Request): Promise<Response> => {
    const url = new URL(request.url);
    if (url.pathname === '/health' || url.pathname === '/healthz') {
      return Response.json({
        status: 'healthy',
        service: 'flaim-selfhost',
        mcpEndpoint: '/mcp',
        leagues: {
          espn: config.leagues.espn?.leagues.length ?? 0,
          sleeper: config.leagues.sleeper?.leagues.length ?? 0,
        },
        providers: {
          espn: config.espnCredentials ? 'ready' : 'missing-credentials',
          sleeper: 'ready',
          yahoo: 'unsupported',
        },
        warnings: config.warnings,
      });
    }
    return gateway.fetch(request);
  };
}

function main() {
  let config: SelfhostConfig;
  try {
    config = loadConfig();
  } catch (error) {
    if (error instanceof ConfigError) {
      console.error(`[selfhost] ${error.message}`);
      process.exit(1);
    }
    throw error;
  }

  for (const warning of config.warnings) console.warn(`[selfhost] warning: ${warning}`);

  const cacheDir = process.env.FLAIM_CACHE_DIR?.trim() || '/data/cache';
  const handler = createRootHandler(config, { cacheDir });

  serve({ fetch: handler, port: config.port, hostname: config.host }, (info) => {
    console.log(
      `[selfhost] Flaim MCP gateway listening on http://${info.address}:${info.port}/mcp ` +
        `(espn=${config.leagues.espn?.leagues.length ?? 0}${config.espnCredentials ? '' : ' [no ESPN credentials]'}, ` +
        `sleeper=${config.leagues.sleeper?.leagues.length ?? 0}, cache=${cacheDir})`
    );
  });
}

const isDirectRun = process.argv[1] && /server\.(ts|js|mjs)$/.test(process.argv[1]);
if (isDirectRun) main();
