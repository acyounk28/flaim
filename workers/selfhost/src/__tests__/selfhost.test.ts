import { describe, it, expect } from 'vitest';
import { parseLeaguesConfig, normalizeSwid, ConfigError, type SelfhostConfig } from '../config';
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
});

describe('self-hosted MCP gateway', () => {
  const handler = createRootHandler(makeConfig(), { cacheDir: null });

  it('serves health', async () => {
    const res = await handler(new Request('http://localhost/health'));
    expect(res.status).toBe(200);
    expect(await res.json()).toMatchObject({ status: 'healthy', leagues: { espn: 1, sleeper: 2 } });
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
