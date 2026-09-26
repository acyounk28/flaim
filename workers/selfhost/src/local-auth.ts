// workers/selfhost/src/local-auth.ts
//
// In-process stand-in for auth-worker. It implements only the /internal/*
// contract that fantasy-mcp, espn-client, sleeper-client and yahoo-client
// consume, backed by the static leagues configuration instead of Supabase.
// There is no OAuth server: a single bearer token (FLAIM_MCP_TOKEN)
// authenticates the operator, and every league in the config belongs to them.
import { Hono } from 'hono';
import { validateInternalService } from '@flaim/worker-shared';
import type { SelfhostConfig } from './config';

export const SELFHOST_USER_ID = 'selfhost-operator';
export const SELFHOST_SCOPE = 'mcp:read mcp:write';

export interface LocalAuthEnv {
  INTERNAL_SERVICE_TOKEN: string;
}

async function constantTimeEqual(a: string, b: string): Promise<boolean> {
  const encoder = new TextEncoder();
  const [aHash, bHash] = await Promise.all([
    crypto.subtle.digest('SHA-256', encoder.encode(a)),
    crypto.subtle.digest('SHA-256', encoder.encode(b)),
  ]);
  const aArr = new Uint8Array(aHash);
  const bArr = new Uint8Array(bHash);
  let result = 0;
  for (let i = 0; i < aArr.length; i += 1) result |= aArr[i] ^ bArr[i];
  return result === 0;
}

export async function isAuthorizedBearer(authHeader: string | null | undefined, expectedToken: string): Promise<boolean> {
  if (!authHeader) return false;
  const match = authHeader.match(/^Bearer\s+(.+)$/i);
  if (!match) return false;
  return constantTimeEqual(match[1].trim(), expectedToken);
}

export function createLocalAuthApp(config: SelfhostConfig) {
  const app = new Hono<{ Bindings: LocalAuthEnv }>();

  // Every internal route requires the service token (same defense-in-depth the
  // Cloudflare deployment has) and, except for usage events, the operator bearer.
  app.use('/internal/*', async (c, next) => {
    const result = await validateInternalService(c.req.raw, c.env, c.req.path);
    if (!result.authorized) return c.json(result.error, result.status);
    if (c.req.path === '/internal/usage-event') return next();
    if (!(await isAuthorizedBearer(c.req.header('Authorization'), config.mcpToken))) {
      return c.json({ valid: false, error: 'unauthorized', error_description: 'Invalid bearer token' }, 401);
    }
    return next();
  });

  app.get('/health', (c) => c.json({ status: 'healthy', service: 'selfhost-auth' }));

  app.get('/internal/introspect', (c) =>
    c.json({
      valid: true,
      userId: SELFHOST_USER_ID,
      scope: SELFHOST_SCOPE,
      authType: 'oauth',
      client_name: 'selfhost',
    })
  );

  app.get('/internal/leagues', (c) => {
    const leagues = (config.leagues.espn?.leagues ?? []).map((league) => ({
      leagueId: league.leagueId,
      sport: league.sport,
      teamId: league.teamId,
      seasonYear: league.seasonYear,
      leagueName: league.leagueName,
      teamName: league.teamName,
      platform: 'espn' as const,
    }));
    return c.json({ success: true, leagues, totalLeagues: leagues.length });
  });

  app.get('/internal/leagues/yahoo', (c) => c.json({ leagues: [] }));

  app.get('/internal/leagues/sleeper', (c) => {
    const leagues = (config.leagues.sleeper?.leagues ?? []).map((league) => ({
      sport: league.sport,
      leagueId: league.leagueId,
      leagueName: league.leagueName,
      rosterId: league.rosterId,
      seasonYear: league.seasonYear,
      recurringLeagueId: league.recurringLeagueId ?? league.leagueId,
    }));
    return c.json({ leagues });
  });

  app.get('/internal/leagues/sleeper/authorize', (c) => {
    const leagueId = c.req.query('league_id');
    const sport = c.req.query('sport');
    const seasonYear = Number(c.req.query('season_year'));
    const allowed = (config.leagues.sleeper?.leagues ?? []).some(
      (league) => league.leagueId === leagueId && league.sport === sport && league.seasonYear === seasonYear
    );
    return c.json({ allowed });
  });

  app.get('/internal/user/preferences', (c) => {
    const prefs = config.leagues.preferences ?? {};
    return c.json({
      defaultSport: prefs.defaultSport ?? null,
      defaultFootball: prefs.defaultFootball ?? null,
      defaultBaseball: prefs.defaultBaseball ?? null,
      defaultBasketball: prefs.defaultBasketball ?? null,
      defaultHockey: prefs.defaultHockey ?? null,
    });
  });

  app.get('/internal/credentials/espn/raw', (c) => {
    if (!config.espnCredentials) {
      return c.json(
        {
          error: 'No credentials found',
          message: 'No ESPN credentials configured. Set ESPN_SWID and ESPN_S2 for the self-hosted server.',
        },
        404
      );
    }
    return c.json({ success: true, credentials: config.espnCredentials });
  });

  // Leagues are static in self-hosted mode; refresh reports the current set so
  // the refresh_leagues tool stays functional and honest about what it did.
  app.post('/internal/leagues/refresh', (c) => {
    const espn = config.leagues.espn?.leagues.length ?? 0;
    const sleeper = config.leagues.sleeper?.leagues.length ?? 0;
    return c.json({
      success: true,
      mode: 'static',
      message: `Self-hosted Flaim reads leagues from ESPN_LEAGUE_IDS / SLEEPER_LEAGUE_IDS in .env and from ${config.leaguesFile}. Edit those and restart the container to change leagues.`,
      results: [
        { platform: 'espn', success: true, leagueCount: espn },
        { platform: 'sleeper', success: true, leagueCount: sleeper },
        { platform: 'yahoo', success: true, leagueCount: 0, note: 'Yahoo is not supported in self-hosted mode (requires OAuth).' },
      ],
    });
  });

  app.post('/internal/usage-event', (c) => c.body(null, 204));

  app.notFound((c) => c.json({ error: 'Not found', path: c.req.path }, 404));

  return app;
}
