import { describe, it, expect } from 'vitest';
import { mkdtempSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { parseLeaguesConfig, normalizeSwid, loadConfig, parseLeagueIdList, ConfigError, type SelfhostConfig } from '../config';
import { createRootHandler } from '../server';

const TOKEN = 'unit-test-token-0123456789abcdef';

function makeConfig(): SelfhostConfig {
  return {
    mcpToken: TOKEN,
    internalServiceToken: 'internal-token',
    port: 8790,
    host: '127.0.0.1',
    leaguesFile: '/config/leagues.json',
    espnCredentials: { swid: '{SWID}', s2: 's2' },
    warnings: [],
    leagues: parseLeaguesConfig({
      espn: { leagues: [{ leagueId: '1', sport: 'football', seasonYear: 2026, teamId: '2' }] },
      sleeper: {
        leagues: [
          { leagueId: '100', sport: 'football', seasonYear: 2026, rosterId: 1 },
          { leagueId: '200', sport: 'football', seasonYear: 2026, rosterId: 2 },
        ],
      },
    }),
  };
}

function mcpRequest(body: unknown, auth?: string): Request {
  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
    Accept: 'application/json, text/event-stream',
  };
  if (auth) headers.Authorization = auth;
  return new Request('http://localhost/mcp', { method: 'POST', headers, body: JSON.stringify(body) });
}

async function readSseJson(response: Response): Promise<{ result?: Record<string, unknown>; error?: unknown }> {
  const text = await response.text();
  const dataLine = text.split('\n').find((line) => line.startsWith('data: '));
  expect(dataLine, `expected SSE data line in: ${text}`).toBeDefined();
  return JSON.parse(dataLine!.slice('data: '.length));
}

describe('config', () => {
  it('normalizes SWID braces', () => {
    expect(normalizeSwid('abc')).toBe('{abc}');
    expect(normalizeSwid('{abc}')).toBe('{abc}');
  });

  it('rejects unknown sports', () => {
    expect(() => parseLeaguesConfig({ sleeper: { leagues: [{ leagueId: '1', sport: 'hockey', seasonYear: 2026 }] } })).toThrow(
      ConfigError
    );
  });

  it('parses league id lists with optional team/roster suffixes', () => {
    expect(parseLeagueIdList('123, 456:7 "789";replace-with-comma-separated-league-ids')).toEqual([
      { leagueId: '123' },
      { leagueId: '456', suffix: '7' },
      { leagueId: '789' },
    ]);
    expect(parseLeagueIdList(undefined)).toEqual([]);
  });
});

describe('loadConfig (env-only, pi-homelab style)', () => {
  const NOW = new Date('2026-09-26T12:00:00Z');
  const missingFile = join(mkdtempSync(join(tmpdir(), 'flaim-selfhost-')), 'does-not-exist.json');

  it('starts from environment variables when the leagues file is absent', () => {
    const config = loadConfig({
      now: NOW,
      env: {
        FLAIM_MCP_TOKEN: TOKEN,
        FLAIM_LEAGUES_FILE: missingFile,
        SWID: 'ABCDEF12-1234-1234-1234-123456789ABC',
        espn_s2: 'AEBs2cookie',
        ESPN_LEAGUE_IDS: '111111,222222:5',
        SLEEPER_LEAGUE_IDS: '1200000000000000000',
        FLAIM_MCP_PORT: '8001',
        PORT: '8790',
      },
    });
    expect(config.port).toBe(8001);
    expect(config.espnCredentials).toEqual({ swid: '{ABCDEF12-1234-1234-1234-123456789ABC}', s2: 'AEBs2cookie' });
    expect(config.leagues.espn?.leagues).toEqual([
      { leagueId: '111111', sport: 'football', seasonYear: 2026 },
      { leagueId: '222222', sport: 'football', seasonYear: 2026, teamId: '5' },
    ]);
    expect(config.leagues.sleeper?.leagues).toEqual([{ leagueId: '1200000000000000000', sport: 'football', seasonYear: 2026 }]);
    expect(config.leagues.preferences?.defaultSport).toBe('football');
    expect(config.leagues.preferences?.defaultFootball).toBeNull();
    expect(config.warnings.some((w) => w.includes('not found'))).toBe(true);
  });

  it('accepts FLAIM_MCP_AUTH_TOKEN and falls back to PORT', () => {
    const config = loadConfig({ now: NOW, env: { FLAIM_MCP_AUTH_TOKEN: TOKEN, FLAIM_LEAGUES_FILE: missingFile, PORT: '9000' } });
    expect(config.mcpToken).toBe(TOKEN);
    expect(config.port).toBe(9000);
  });

  it('does not crash on missing or placeholder ESPN credentials', () => {
    const config = loadConfig({
      now: NOW,
      env: {
        FLAIM_MCP_TOKEN: TOKEN,
        FLAIM_LEAGUES_FILE: missingFile,
        ESPN_SWID: '{XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX}',
        ESPN_S2: 'replace-with-espn-s2-cookie',
        ESPN_LEAGUE_IDS: '333333',
        SLEEPER_LEAGUE_IDS: '1200000000000000001',
      },
    });
    expect(config.espnCredentials).toBeNull();
    expect(config.leagues.espn?.leagues).toHaveLength(1);
    expect(config.leagues.sleeper?.leagues).toHaveLength(1);
    expect(config.warnings.join('\n')).toMatch(/ESPN credentials are incomplete/);
  });

  it('ignores the example leagues file and merges env leagues into a real one', () => {
    const dir = mkdtempSync(join(tmpdir(), 'flaim-selfhost-'));
    const example = join(dir, 'example.json');
    writeFileSync(
      example,
      JSON.stringify({
        espn: { leagues: [{ leagueId: '123456', sport: 'football', seasonYear: 2026, teamId: '4' }] },
        sleeper: { username: 'your_sleeper_username', leagues: [{ leagueId: '1124838275649073152', sport: 'football', seasonYear: 2026 }] },
        preferences: { defaultFootball: { platform: 'sleeper', leagueId: '1124838275649073152', seasonYear: 2026 } },
      })
    );
    const fromExample = loadConfig({
      now: NOW,
      env: { FLAIM_MCP_TOKEN: TOKEN, FLAIM_LEAGUES_FILE: example, SLEEPER_LEAGUE_IDS: '1200000000000000002' },
    });
    expect(fromExample.leagues.espn?.leagues).toEqual([]);
    expect(fromExample.leagues.sleeper?.username).toBeUndefined();
    expect(fromExample.leagues.sleeper?.leagues.map((l) => l.leagueId)).toEqual(['1200000000000000002']);
    expect(fromExample.leagues.preferences?.defaultFootball).toEqual({
      platform: 'sleeper',
      leagueId: '1200000000000000002',
      seasonYear: 2026,
    });

    const real = join(dir, 'real.json');
    writeFileSync(real, JSON.stringify({ espn: { leagues: [{ leagueId: '555', sport: 'football', seasonYear: 2025, leagueName: 'Named' }] } }));
    const merged = loadConfig({
      now: NOW,
      env: { FLAIM_MCP_TOKEN: TOKEN, FLAIM_LEAGUES_FILE: real, ESPN_LEAGUE_IDS: '555,666', ESPN_SEASON_YEAR: '2025' },
    });
    expect(merged.leagues.espn?.leagues.map((l) => [l.leagueId, l.leagueName])).toEqual([
      ['555', 'Named'],
      ['666', undefined],
    ]);

    const broken = join(dir, 'broken.json');
    writeFileSync(broken, '{ not json');
    const withBroken = loadConfig({ now: NOW, env: { FLAIM_MCP_TOKEN: TOKEN, FLAIM_LEAGUES_FILE: broken } });
    expect(withBroken.warnings.join('\n')).toMatch(/not valid JSON/);
  });

  it('still requires a real MCP token', () => {
    expect(() => loadConfig({ env: { FLAIM_LEAGUES_FILE: missingFile } })).toThrow(ConfigError);
    expect(() => loadConfig({ env: { FLAIM_MCP_TOKEN: 'replace-with-a-separate-strong-token', FLAIM_LEAGUES_FILE: missingFile } })).toThrow(
      ConfigError
    );
  });
});

describe('self-hosted MCP gateway', () => {
  const handler = createRootHandler(makeConfig(), { cacheDir: null });

  it('serves health', async () => {
    const res = await handler(new Request('http://localhost/health'));
    expect(res.status).toBe(200);
    expect(await res.json()).toMatchObject({ status: 'healthy', leagues: { espn: 1, sleeper: 2 }, providers: { espn: 'ready' } });
  });

  it('serves Sleeper leagues when ESPN credentials are missing', async () => {
    const degraded = createRootHandler({ ...makeConfig(), espnCredentials: null }, { cacheDir: null });
    const health = await (await degraded(new Request('http://localhost/health'))).json();
    expect(health).toMatchObject({ providers: { espn: 'missing-credentials', sleeper: 'ready' } });
    const res = await degraded(
      mcpRequest(
        { jsonrpc: '2.0', id: 9, method: 'tools/call', params: { name: 'get_user_session', arguments: {} } },
        `Bearer ${TOKEN}`
      )
    );
    expect(res.status).toBe(200);
    const payload = await readSseJson(res);
    const structured = payload.result?.structuredContent as { totalLeaguesFound: number };
    expect(structured.totalLeaguesFound).toBe(3);
  });

  it('answers initialize over SSE without auth', async () => {
    const res = await handler(
      mcpRequest({
        jsonrpc: '2.0',
        id: 1,
        method: 'initialize',
        params: { protocolVersion: '2025-03-26', capabilities: {}, clientInfo: { name: 't', version: '1' } },
      })
    );
    expect(res.status).toBe(200);
    expect(res.headers.get('content-type')).toContain('text/event-stream');
    const payload = await readSseJson(res);
    expect(payload.result?.serverInfo).toMatchObject({ name: 'fantasy-mcp' });
  });

  it('rejects tool calls with a wrong bearer token', async () => {
    const res = await handler(
      mcpRequest({ jsonrpc: '2.0', id: 2, method: 'tools/call', params: { name: 'get_user_session', arguments: {} } }, 'Bearer nope')
    );
    expect(res.status).toBe(401);
  });

  it('returns configured leagues for the operator token', async () => {
    const res = await handler(
      mcpRequest(
        { jsonrpc: '2.0', id: 3, method: 'tools/call', params: { name: 'get_user_session', arguments: {} } },
        `Bearer ${TOKEN}`
      )
    );
    expect(res.status).toBe(200);
    const payload = await readSseJson(res);
    const structured = payload.result?.structuredContent as { totalLeaguesFound: number; allLeagues: Array<{ platform: string }> };
    expect(structured.totalLeaguesFound).toBe(3);
    expect(structured.allLeagues.map((l) => l.platform).sort()).toEqual(['espn', 'sleeper', 'sleeper']);
  });
});
