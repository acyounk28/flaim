// workers/selfhost/src/config.ts
//
// Self-hosted Flaim replaces the Supabase-backed auth-worker with a static
// configuration: one operator, a fixed set of leagues, and a single bearer
// token. Configuration comes from environment variables first (the .env used
// by docker compose) and optionally from a JSON leagues file
// (FLAIM_LEAGUES_FILE). The file is never required: a missing, invalid, or
// placeholder file only produces warnings, and the server still starts with
// whatever the environment provides.
import { readFileSync } from 'node:fs';
import { z } from 'zod';
import { getDefaultSeasonYear } from '@flaim/worker-shared';

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
export type EspnSport = (typeof ESPN_SPORTS)[number];
export type SleeperSport = (typeof SLEEPER_SPORTS)[number];

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
  /** Non-fatal configuration problems, surfaced in logs and /health. */
  warnings: string[];
}

export class ConfigError extends Error {}

/**
 * Environment variable aliases. The first name that is set wins. These cover
 * both the names documented in docs/SELF-HOSTING.md and the ones used by the
 * pi-homelab-setup compose stack (SWID / espn_s2 / FLAIM_MCP_AUTH_TOKEN / ...).
 */
export const ENV_ALIASES = {
  mcpToken: ['FLAIM_MCP_TOKEN', 'FLAIM_MCP_AUTH_TOKEN'],
  espnSwid: ['ESPN_SWID', 'SWID', 'swid', 'espn_swid'],
  espnS2: ['ESPN_S2', 'espn_s2', 'ESPN_ESPN_S2'],
  espnLeagueIds: ['ESPN_LEAGUE_IDS', 'ESPN_LEAGUE_ID'],
  sleeperLeagueIds: ['SLEEPER_LEAGUE_IDS', 'SLEEPER_LEAGUE_ID'],
  sleeperUsername: ['SLEEPER_USERNAME', 'SLEEPER_USER'],
  sleeperUserId: ['SLEEPER_USER_ID'],
  espnSport: ['ESPN_SPORT', 'FLAIM_DEFAULT_SPORT'],
  sleeperSport: ['SLEEPER_SPORT', 'FLAIM_DEFAULT_SPORT'],
  espnSeasonYear: ['ESPN_SEASON_YEAR', 'FLAIM_SEASON_YEAR'],
  sleeperSeasonYear: ['SLEEPER_SEASON_YEAR', 'FLAIM_SEASON_YEAR'],
  port: ['FLAIM_MCP_PORT', 'PORT'],
  host: ['FLAIM_MCP_HOST', 'HOST'],
} as const;

/** Values copied verbatim from .env.example / leagues.example.json are treated as unset. */
const PLACEHOLDER_PATTERNS = [/^replace-with/i, /^your[-_ ]/i, /^changeme$/i, /^<.*>$/, /^\{?X{8}-X{4}-X{4}-X{4}-X{12}\}?$/i, /^xxx+$/i];

const EXAMPLE_ESPN_LEAGUE_IDS = new Set(['123456', '987654321']);
const EXAMPLE_SLEEPER_LEAGUE_IDS = new Set(['1124838275649073152', '1124910048231305216', '1125000934567890123']);

export function isPlaceholder(value: string | undefined): boolean {
  if (!value) return true;
  const trimmed = value.trim();
  if (!trimmed) return true;
  return PLACEHOLDER_PATTERNS.some((pattern) => pattern.test(trimmed));
}

type EnvSource = Record<string, string | undefined>;

function readEnv(env: EnvSource, names: readonly string[]): string | undefined {
  for (const name of names) {
    const value = env[name];
    if (value && value.trim().length > 0) return value.trim();
  }
  return undefined;
}

/** Like readEnv, but ignores example placeholders such as "replace-with-swid-cookie". */
function readSecretEnv(env: EnvSource, names: readonly string[], warnings: string[]): string | undefined {
  for (const name of names) {
    const value = env[name];
    if (!value || value.trim().length === 0) continue;
    if (isPlaceholder(value)) {
      warnings.push(`${name} still contains the example placeholder value; treating it as unset.`);
      continue;
    }
    return value.trim();
  }
  return undefined;
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

const EMPTY_LEAGUES: LeaguesConfig = { espn: { leagues: [] }, sleeper: { leagues: [] } };

/**
 * Load the optional leagues file. Any problem (missing, unreadable, invalid
 * JSON, schema mismatch) is reported as a warning and yields an empty config
 * so startup can continue on environment variables alone.
 */
function loadLeaguesFile(path: string, warnings: string[]): LeaguesConfig {
  let text: string;
  try {
    text = readFileSync(path, 'utf8');
  } catch (error) {
    const code = (error as NodeJS.ErrnoException).code;
    if (code === 'ENOENT' || code === 'ENOTDIR') {
      warnings.push(`Leagues file ${path} not found; using environment variables only.`);
    } else {
      warnings.push(`Cannot read leagues file ${path} (${error instanceof Error ? error.message : String(error)}); using environment variables only.`);
    }
    return EMPTY_LEAGUES;
  }
  if (text.trim().length === 0) {
    warnings.push(`Leagues file ${path} is empty; using environment variables only.`);
    return EMPTY_LEAGUES;
  }
  let json: unknown;
  try {
    json = JSON.parse(text);
  } catch (error) {
    warnings.push(`Leagues file ${path} is not valid JSON (${error instanceof Error ? error.message : String(error)}); ignoring it.`);
    return EMPTY_LEAGUES;
  }
  try {
    return parseLeaguesConfig(json);
  } catch (error) {
    warnings.push(`${error instanceof Error ? error.message : String(error)} (in ${path}); ignoring the file.`);
    return EMPTY_LEAGUES;
  }
}

/** Drop leagues copied from config/leagues.example.json. */
export function stripExampleLeagues(config: LeaguesConfig, warnings: string[]): LeaguesConfig {
  const espn = config.espn?.leagues ?? [];
  const sleeper = config.sleeper?.leagues ?? [];
  const keptEspn = espn.filter((l) => !EXAMPLE_ESPN_LEAGUE_IDS.has(l.leagueId));
  const keptSleeper = sleeper.filter((l) => !EXAMPLE_SLEEPER_LEAGUE_IDS.has(l.leagueId));
  const dropped = espn.length - keptEspn.length + (sleeper.length - keptSleeper.length);
  if (dropped > 0) {
    warnings.push(`Ignored ${dropped} league(s) copied from config/leagues.example.json (example IDs).`);
  }
  const username = config.sleeper?.username;
  const result: LeaguesConfig = {
    ...config,
    espn: { ...config.espn, leagues: keptEspn },
    sleeper: {
      ...config.sleeper,
      username: isPlaceholder(username) ? undefined : username,
      leagues: keptSleeper,
    },
  };
  const prefs = result.preferences;
  if (prefs) {
    const isKnown = (d: z.infer<typeof leagueDefaultSchema> | null | undefined) =>
      !d ||
      (d.platform === 'espn' ? keptEspn : keptSleeper).some((l) => l.leagueId === d.leagueId && l.seasonYear === d.seasonYear);
    result.preferences = {
      ...prefs,
      defaultFootball: isKnown(prefs.defaultFootball) ? prefs.defaultFootball : undefined,
      defaultBaseball: isKnown(prefs.defaultBaseball) ? prefs.defaultBaseball : undefined,
      defaultBasketball: isKnown(prefs.defaultBasketball) ? prefs.defaultBasketball : undefined,
      defaultHockey: isKnown(prefs.defaultHockey) ? prefs.defaultHockey : undefined,
    };
  }
  return result;
}

/**
 * Parse "id, id:teamId id" style lists. Separators: comma, semicolon, whitespace.
 * An optional ":suffix" carries the operator's team id (ESPN) or roster id (Sleeper).
 */
export function parseLeagueIdList(raw: string | undefined): Array<{ leagueId: string; suffix?: string }> {
  if (!raw) return [];
  const out: Array<{ leagueId: string; suffix?: string }> = [];
  for (const token of raw.split(/[\s,;]+/)) {
    const item = token.trim().replace(/^["']|["']$/g, '');
    if (!item || isPlaceholder(item)) continue;
    const [leagueId, suffix] = item.split(':', 2);
    if (!/^[A-Za-z0-9_-]+$/.test(leagueId)) continue;
    out.push(suffix ? { leagueId, suffix } : { leagueId });
  }
  return out;
}

function parseSport<T extends readonly string[]>(raw: string | undefined, allowed: T, fallback: T[number], label: string, warnings: string[]): T[number] {
  if (!raw) return fallback;
  const lower = raw.toLowerCase();
  if ((allowed as readonly string[]).includes(lower)) return lower as T[number];
  warnings.push(`${label}="${raw}" is not one of ${allowed.join('/')}; using ${fallback}.`);
  return fallback;
}

function parseSeasonYear(raw: string | undefined, sport: EspnSport, label: string, warnings: string[], now: Date): number {
  if (!raw) return getDefaultSeasonYear(sport, now);
  const year = Number(raw);
  if (Number.isInteger(year) && year >= 1990 && year <= 2100) return year;
  warnings.push(`${label}="${raw}" is not a valid season year; using the current ${sport} season.`);
  return getDefaultSeasonYear(sport, now);
}

/**
 * Build league entries from ESPN_LEAGUE_IDS / SLEEPER_LEAGUE_IDS and merge them
 * with anything loaded from the leagues file (file entries win on conflicts so
 * hand-written names/team ids are preserved).
 */
export function leaguesFromEnv(env: EnvSource, warnings: string[], now = new Date()): LeaguesConfig {
  const espnSport = parseSport(readEnv(env, ENV_ALIASES.espnSport), ESPN_SPORTS, 'football', 'ESPN_SPORT', warnings);
  const sleeperSport = parseSport(readEnv(env, ENV_ALIASES.sleeperSport), SLEEPER_SPORTS, 'football', 'SLEEPER_SPORT', warnings);
  const espnSeason = parseSeasonYear(readEnv(env, ENV_ALIASES.espnSeasonYear), espnSport, 'ESPN_SEASON_YEAR', warnings, now);
  const sleeperSeason = parseSeasonYear(readEnv(env, ENV_ALIASES.sleeperSeasonYear), sleeperSport, 'SLEEPER_SEASON_YEAR', warnings, now);

  const espnLeagues: EspnLeagueEntry[] = parseLeagueIdList(readEnv(env, ENV_ALIASES.espnLeagueIds)).map(({ leagueId, suffix }) => ({
    leagueId,
    sport: espnSport,
    seasonYear: espnSeason,
    ...(suffix && /^\d+$/.test(suffix) ? { teamId: suffix } : {}),
  }));

  const sleeperLeagues: SleeperLeagueEntry[] = parseLeagueIdList(readEnv(env, ENV_ALIASES.sleeperLeagueIds)).map(({ leagueId, suffix }) => {
    const rosterId = suffix && /^\d+$/.test(suffix) ? Number(suffix) : undefined;
    return {
      leagueId,
      sport: sleeperSport,
      seasonYear: sleeperSeason,
      ...(rosterId && rosterId > 0 ? { rosterId } : {}),
    };
  });

  const username = readEnv(env, ENV_ALIASES.sleeperUsername);
  const userId = readEnv(env, ENV_ALIASES.sleeperUserId);
  return {
    espn: { leagues: espnLeagues },
    sleeper: {
      ...(username && !isPlaceholder(username) ? { username } : {}),
      ...(userId && !isPlaceholder(userId) ? { userId } : {}),
      leagues: sleeperLeagues,
    },
  };
}

export function mergeLeagues(fromFile: LeaguesConfig, fromEnv: LeaguesConfig): LeaguesConfig {
  const key = (l: { leagueId: string; sport: string; seasonYear: number }) => `${l.leagueId}:${l.sport}:${l.seasonYear}`;
  const espnSeen = new Set((fromFile.espn?.leagues ?? []).map(key));
  const sleeperSeen = new Set((fromFile.sleeper?.leagues ?? []).map(key));
  return {
    espn: {
      ...fromFile.espn,
      leagues: [...(fromFile.espn?.leagues ?? []), ...(fromEnv.espn?.leagues ?? []).filter((l) => !espnSeen.has(key(l)))],
    },
    sleeper: {
      ...fromFile.sleeper,
      username: fromFile.sleeper?.username ?? fromEnv.sleeper?.username,
      userId: fromFile.sleeper?.userId ?? fromEnv.sleeper?.userId,
      leagues: [...(fromFile.sleeper?.leagues ?? []), ...(fromEnv.sleeper?.leagues ?? []).filter((l) => !sleeperSeen.has(key(l)))],
    },
    preferences: fromFile.preferences,
  };
}

/**
 * When the operator set no explicit default and has leagues for exactly one
 * platform in a sport, use the first of them so clients need not pass a
 * league id on every call.
 */
function fillDefaultPreferences(leagues: LeaguesConfig): LeaguesConfig {
  const prefs = { ...(leagues.preferences ?? {}) };
  const pick = (sport: EspnSport) => {
    const espn = (leagues.espn?.leagues ?? []).filter((l) => l.sport === sport);
    const sleeper = (leagues.sleeper?.leagues ?? []).filter((l) => l.sport === sport);
    const all = [
      ...espn.map((l) => ({ platform: 'espn' as const, leagueId: l.leagueId, seasonYear: l.seasonYear })),
      ...sleeper.map((l) => ({ platform: 'sleeper' as const, leagueId: l.leagueId, seasonYear: l.seasonYear })),
    ];
    return all.length === 1 ? all[0] : null;
  };
  if (prefs.defaultFootball === undefined) prefs.defaultFootball = pick('football');
  if (prefs.defaultBaseball === undefined) prefs.defaultBaseball = pick('baseball');
  if (prefs.defaultBasketball === undefined) prefs.defaultBasketball = pick('basketball');
  if (prefs.defaultHockey === undefined) prefs.defaultHockey = pick('hockey');
  if (prefs.defaultSport === undefined) {
    const sports = (['football', 'baseball', 'basketball', 'hockey'] as const).filter(
      (s) =>
        (leagues.espn?.leagues ?? []).some((l) => l.sport === s) || (leagues.sleeper?.leagues ?? []).some((l) => l.sport === s)
    );
    prefs.defaultSport = sports.length === 1 ? sports[0] : null;
  }
  return { ...leagues, preferences: prefs };
}

export interface LoadConfigOptions {
  env?: EnvSource;
  now?: Date;
}

export function loadConfig(options: LoadConfigOptions = {}): SelfhostConfig {
  const env = options.env ?? process.env;
  const now = options.now ?? new Date();
  const warnings: string[] = [];

  const mcpToken = readSecretEnv(env, ENV_ALIASES.mcpToken, warnings);
  if (!mcpToken) {
    throw new ConfigError(
      `${ENV_ALIASES.mcpToken.join(' or ')} is required (a real value, not the example placeholder). Generate one with: openssl rand -hex 32`
    );
  }
  if (mcpToken.length < 24) {
    throw new ConfigError('FLAIM_MCP_TOKEN must be at least 24 characters.');
  }

  const leaguesFile = readEnv(env, ['FLAIM_LEAGUES_FILE']) ?? '/config/leagues.json';
  const inlineJson = readEnv(env, ['FLAIM_LEAGUES_JSON']);
  let fromFile: LeaguesConfig;
  if (inlineJson) {
    try {
      fromFile = parseLeaguesConfig(JSON.parse(inlineJson));
    } catch (error) {
      warnings.push(`FLAIM_LEAGUES_JSON is invalid (${error instanceof Error ? error.message : String(error)}); ignoring it.`);
      fromFile = EMPTY_LEAGUES;
    }
  } else {
    fromFile = loadLeaguesFile(leaguesFile, warnings);
  }
  fromFile = stripExampleLeagues(fromFile, warnings);

  const leagues = fillDefaultPreferences(mergeLeagues(fromFile, leaguesFromEnv(env, warnings, now)));

  const swid = readSecretEnv(env, ENV_ALIASES.espnSwid, warnings) ?? (isPlaceholder(fromFile.espn?.swid) ? undefined : fromFile.espn?.swid);
  const s2 = readSecretEnv(env, ENV_ALIASES.espnS2, warnings) ?? (isPlaceholder(fromFile.espn?.s2) ? undefined : fromFile.espn?.s2);
  const espnLeagueCount = leagues.espn?.leagues.length ?? 0;
  const sleeperLeagueCount = leagues.sleeper?.leagues.length ?? 0;
  let espnCredentials: SelfhostConfig['espnCredentials'] = null;
  if (swid && s2) {
    espnCredentials = { swid: normalizeSwid(swid), s2 };
  } else if (espnLeagueCount > 0) {
    warnings.push(
      `${espnLeagueCount} ESPN league(s) configured but ESPN credentials are incomplete ` +
        `(need both ${ENV_ALIASES.espnSwid[0]}/${ENV_ALIASES.espnSwid[1]} and ${ENV_ALIASES.espnS2[0]}). ` +
        'ESPN tools will return a credentials error until both are set; Sleeper tools are unaffected.'
    );
  } else if (swid || s2) {
    warnings.push('Only one of SWID / ESPN_S2 is set; ESPN access is disabled until both are present.');
  }
  if (espnLeagueCount === 0 && sleeperLeagueCount === 0) {
    warnings.push(
      'No leagues configured. Set ESPN_LEAGUE_IDS and/or SLEEPER_LEAGUE_IDS (comma-separated) in .env, ' +
        `or provide a leagues file at ${leaguesFile}. The server will start but league tools will report zero leagues.`
    );
  }

  const portRaw = readEnv(env, ENV_ALIASES.port) ?? '8790';
  const port = Number(portRaw);
  if (!Number.isInteger(port) || port <= 0 || port > 65535) {
    throw new ConfigError(`PORT must be a valid TCP port, got "${portRaw}"`);
  }

  return {
    mcpToken,
    internalServiceToken: readEnv(env, ['INTERNAL_SERVICE_TOKEN']) ?? randomToken(),
    port,
    host: readEnv(env, ENV_ALIASES.host) ?? '0.0.0.0',
    leaguesFile,
    leagues,
    espnCredentials,
    warnings,
  };
}
