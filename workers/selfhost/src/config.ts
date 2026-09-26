// workers/selfhost/src/config.ts
//
// Self-hosted Flaim replaces the Supabase-backed auth-worker with a static
// configuration: one operator, a fixed set of leagues, and a single bearer
// token. This module loads and validates that configuration from a JSON file
// (FLAIM_LEAGUES_FILE) plus environment-variable overrides for credentials.
import { readFileSync } from 'node:fs';
import { z } from 'zod';

const ESPN_SPORTS = ['football', 'baseball', 'basketball', 'hockey'] as const;
const SLEEPER_SPORTS = ['football', 'basketball'] as const;
const ALL_SPORTS = ['football', 'baseball', 'basketball', 'hockey'] as const;

const espnLeagueSchema = z.object({
  leagueId: z.string().min(1),
  sport: z.enum(ESPN_SPORTS),
  seasonYear: z.number().int().min(1990).max(2100),
  teamId: z.string().min(1).optional(),
  leagueName: z.string().optional(),
  teamName: z.string().optional(),
});

const sleeperLeagueSchema = z.object({
  leagueId: z.string().min(1),
  sport: z.enum(SLEEPER_SPORTS),
  seasonYear: z.number().int().min(1990).max(2100),
  rosterId: z.number().int().positive().optional(),
  leagueName: z.string().optional(),
  /** Stable id shared by every season of a league (Sleeper "previous_league_id" chain). */
  recurringLeagueId: z.string().optional(),
});

const leagueDefaultSchema = z.object({
  platform: z.enum(['espn', 'sleeper']),
  leagueId: z.string().min(1),
  seasonYear: z.number().int(),
});

const configSchema = z.object({
  espn: z
    .object({
      swid: z.string().optional(),
      s2: z.string().optional(),
      leagues: z.array(espnLeagueSchema).default([]),
    })
    .optional(),
  sleeper: z
    .object({
      username: z.string().optional(),
      userId: z.string().optional(),
      leagues: z.array(sleeperLeagueSchema).default([]),
    })
    .optional(),
  preferences: z
    .object({
      defaultSport: z.enum(ALL_SPORTS).nullable().optional(),
      defaultFootball: leagueDefaultSchema.nullable().optional(),
      defaultBaseball: leagueDefaultSchema.nullable().optional(),
      defaultBasketball: leagueDefaultSchema.nullable().optional(),
      defaultHockey: leagueDefaultSchema.nullable().optional(),
    })
    .optional(),
});

export type EspnLeagueEntry = z.infer<typeof espnLeagueSchema>;
export type SleeperLeagueEntry = z.infer<typeof sleeperLeagueSchema>;
export type LeaguesConfig = z.infer<typeof configSchema>;

export interface SelfhostConfig {
  /** Bearer token MCP clients must present. */
  mcpToken: string;
  /** Token used between in-process services (mirrors INTERNAL_SERVICE_TOKEN in Cloudflare). */
  internalServiceToken: string;
  port: number;
  host: string;
  leaguesFile: string;
  leagues: LeaguesConfig;
  espnCredentials: { swid: string; s2: string } | null;
}

export class ConfigError extends Error {}

function readEnv(name: string): string | undefined {
  const value = process.env[name];
  return value && value.trim().length > 0 ? value.trim() : undefined;
}

function readLeagueIds(name: string): string[] {
  return (readEnv(name) ?? '').split(',').map((id) => id.trim()).filter(Boolean);
}

function randomToken(): string {
  const bytes = new Uint8Array(32);
  crypto.getRandomValues(bytes);
  return Array.from(bytes, (b) => b.toString(16).padStart(2, '0')).join('');
}

/**
 * ESPN's SWID cookie is normally wrapped in braces ("{...}"). Accept either
 * form so operators can paste straight from the browser.
 */
export function normalizeSwid(swid: string): string {
  const trimmed = swid.trim();
  if (trimmed.startsWith('{') && trimmed.endsWith('}')) return trimmed;
  return `{${trimmed.replace(/^\{|\}$/g, '')}}`;
}

export function parseLeaguesConfig(raw: unknown): LeaguesConfig {
  const parsed = configSchema.safeParse(raw);
  if (!parsed.success) {
    const issues = parsed.error.issues
      .map((issue) => `${issue.path.join('.') || '(root)'}: ${issue.message}`)
      .join('; ');
    throw new ConfigError(`Invalid leagues configuration: ${issues}`);
  }
  return parsed.data;
}

function loadLeaguesFile(path: string): LeaguesConfig {
  let text: string;
  try {
    text = readFileSync(path, 'utf8');
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === 'ENOENT') {
      console.warn(`[selfhost] Leagues file ${path} not found; using environment configuration and empty league lists.`);
      return parseLeaguesConfig({});
    }
    throw new ConfigError(
      `Cannot read leagues file at ${path} (${error instanceof Error ? error.message : String(error)}). ` +
        'Set FLAIM_LEAGUES_FILE or mount config/leagues.json into the container.'
    );
  }
  let json: unknown;
  try {
    json = JSON.parse(text);
  } catch (error) {
    throw new ConfigError(`Leagues file ${path} is not valid JSON: ${error instanceof Error ? error.message : String(error)}`);
  }
  return parseLeaguesConfig(json);
}

export function loadConfig(): SelfhostConfig {
  const mcpToken = readEnv('FLAIM_MCP_TOKEN');
  if (!mcpToken) {
    throw new ConfigError('FLAIM_MCP_TOKEN is required. Generate one with: openssl rand -hex 32');
  }
  if (mcpToken.length < 24) {
    throw new ConfigError('FLAIM_MCP_TOKEN must be at least 24 characters.');
  }

  const leaguesFile = readEnv('FLAIM_LEAGUES_FILE') ?? '/config/leagues.json';
  const inlineJson = readEnv('FLAIM_LEAGUES_JSON');
  const leagues = inlineJson ? parseLeaguesConfig(JSON.parse(inlineJson)) : loadLeaguesFile(leaguesFile);
  const seasonYear = new Date().getFullYear();
  const espnLeagueIds = readLeagueIds('ESPN_LEAGUE_IDS');
  const sleeperLeagueIds = readLeagueIds('SLEEPER_LEAGUE_IDS');
  const espn = leagues.espn ?? { leagues: [] };
  const sleeper = leagues.sleeper ?? { leagues: [] };
  const mergedLeagues = parseLeaguesConfig({
    ...leagues,
    espn: { ...espn, leagues: [...espn.leagues, ...espnLeagueIds.filter((id) => !espn.leagues.some((league) => league.leagueId === id)).map((leagueId) => ({ leagueId, sport: 'football', seasonYear }))] },
    sleeper: { ...sleeper, leagues: [...sleeper.leagues, ...sleeperLeagueIds.filter((id) => !sleeper.leagues.some((league) => league.leagueId === id)).map((leagueId) => ({ leagueId, sport: 'football', seasonYear }))] },
  });

  const swid = readEnv('SWID') ?? readEnv('ESPN_SWID') ?? readEnv('espn_swid') ?? readEnv('espn_s2') ?? mergedLeagues.espn?.swid;
  const s2 = readEnv('ESPN_S2') ?? readEnv('espn_s2') ?? mergedLeagues.espn?.s2;
  let espnCredentials: SelfhostConfig['espnCredentials'] = null;
  if (swid && s2) {
    espnCredentials = { swid: normalizeSwid(swid), s2 };
  } else if ((mergedLeagues.espn?.leagues.length ?? 0) > 0) {
    console.warn('[selfhost] ESPN leagues are configured but ESPN credentials are missing or incomplete; ESPN tools will be unavailable. Set SWID and ESPN_S2 (or ESPN_SWID and ESPN_S2).');
  }

  const portRaw = readEnv('PORT') ?? '8790';
  const port = Number(portRaw);
  if (!Number.isInteger(port) || port <= 0 || port > 65535) {
    throw new ConfigError(`PORT must be a valid TCP port, got "${portRaw}"`);
  }

  return {
    mcpToken,
    internalServiceToken: readEnv('INTERNAL_SERVICE_TOKEN') ?? randomToken(),
    port,
    host: readEnv('HOST') ?? '0.0.0.0',
    leaguesFile,
    leagues: mergedLeagues,
    espnCredentials,
  };
}
