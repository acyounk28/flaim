// Integration tests that exercise the self-hosted gateway over real TCP, the
// way an MCP client (Poke, Claude, ChatGPT, mcp-remote) does: env-only config
// with no leagues file anywhere, a real listening socket, raw HTTP requests
// asserting exact status codes / bodies / headers, and the official MCP SDK
// client driving the full Streamable HTTP lifecycle.
import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import { spawn, type ChildProcess } from 'node:child_process';
import { mkdtempSync } from 'node:fs';
import { createServer, type AddressInfo } from 'node:net';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { serve, type ServerType } from '@hono/node-server';
import { Client } from '@modelcontextprotocol/sdk/client/index.js';
import { StreamableHTTPClientTransport } from '@modelcontextprotocol/sdk/client/streamableHttp.js';
import { loadConfig } from '../config';
import { createRootHandler } from '../server';

const TOKEN = 'integration-test-token-0123456789abcdef';
const MISSING_LEAGUES_FILE = join(mkdtempSync(join(tmpdir(), 'flaim-selfhost-it-')), 'leagues.json');

// Exactly what a Raspberry Pi .env provides when ESPN has not been set up yet:
// a token, one Sleeper league, placeholder ESPN cookies, and no leagues.json.
const PI_ENV: Record<string, string> = {
  FLAIM_MCP_TOKEN: TOKEN,
  FLAIM_LEAGUES_FILE: MISSING_LEAGUES_FILE,
  FLAIM_MCP_HOST: '127.0.0.1',
  SLEEPER_LEAGUE_IDS: '1200000000000000001',
  SWID: 'replace-with-your-swid',
  espn_s2: '',
};

const INITIALIZE = {
  jsonrpc: '2.0',
  id: 1,
  method: 'initialize',
  params: { protocolVersion: '2025-03-26', capabilities: {}, clientInfo: { name: 'integration-test', version: '1.0' } },
};

const EXPECTED_TOOLS = [
  'get_ancient_history',
  'get_draft',
  'get_free_agents',
  'get_league_info',
  'get_matchups',
  'get_players',
  'get_roster',
  'get_standings',
  'get_transactions',
  'get_user_session',
  'refresh_leagues',
];

async function freePort(): Promise<number> {
  return new Promise((res, rej) => {
    const s = createServer();
    s.once('error', rej);
    s.listen(0, '127.0.0.1', () => {
      const { port } = s.address() as AddressInfo;
      s.close(() => res(port));
    });
  });
}

// Open a GET SSE stream, read its first frame, then abort (the stream is
// otherwise held open by the server until the client disconnects).
async function openSseStream(url: string, headers: Record<string, string> = {}) {
  const controller = new AbortController();
  const res = await fetch(url, { headers, signal: controller.signal });
  let firstFrame = '';
  if (res.ok && res.body) {
    const reader = res.body.getReader();
    const timeout = setTimeout(() => reader.cancel(), 5_000);
    const { value } = await reader.read();
    clearTimeout(timeout);
    firstFrame = new TextDecoder().decode(value);
  }
  controller.abort();
  return { res, firstFrame };
}

function parseSse(text: string): Array<{ result?: Record<string, unknown>; error?: { code: number; message: string }; id?: unknown }> {
  return text
    .split('\n')
    .filter((line) => line.startsWith('data: '))
    .map((line) => JSON.parse(line.slice('data: '.length)));
}

describe('self-hosted gateway over HTTP (env-only, no leagues file)', () => {
  let server: ServerType;
  let base: string;
  const warnings: string[] = [];

  beforeAll(async () => {
    const port = await freePort();
    const config = loadConfig({ env: { ...PI_ENV, FLAIM_MCP_PORT: String(port) }, now: new Date('2026-09-26T12:00:00Z') });
    warnings.push(...config.warnings);
    expect(config.port).toBe(port);
    expect(config.host).toBe('127.0.0.1');
    server = serve({ fetch: createRootHandler(config, { cacheDir: null }), port, hostname: '127.0.0.1' });
    base = `http://127.0.0.1:${port}`;
  });

  afterAll(async () => {
    await new Promise<void>((res) => server.close(() => res()));
  });

  function post(body: unknown, headers: Record<string, string> = {}) {
    return fetch(`${base}/mcp`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Accept: 'application/json, text/event-stream', ...headers },
      body: JSON.stringify(body),
    });
  }

  it('boots with warnings (not errors) when leagues.json is absent and ESPN is a placeholder', async () => {
    expect(warnings.some((w) => w.includes('not found'))).toBe(true);
    const res = await fetch(`${base}/health`);
    expect(res.status).toBe(200);
    expect(res.headers.get('content-type')).toContain('application/json');
    const health = (await res.json()) as { warnings: unknown };
    expect(health).toMatchObject({
      status: 'healthy',
      mcpEndpoint: '/mcp',
      leagues: { espn: 0, sleeper: 1 },
      providers: { espn: 'missing-credentials', sleeper: 'ready', yahoo: 'unsupported' },
    });
    expect(Array.isArray(health.warnings)).toBe(true);
    // /healthz is the alias used by container healthchecks
    expect((await fetch(`${base}/healthz`)).status).toBe(200);
  });

  describe('GET /mcp (SSE stream, what a Poke-style URL validator probes)', () => {
    it.each(['/mcp', '/mcp/', '/fantasy/mcp', '/fantasy/mcp/'])(
      'GET %s opens a 200 text/event-stream with keep-alive headers and a first SSE frame, without auth',
      async (path) => {
        const { res, firstFrame } = await openSseStream(`${base}${path}`, { Accept: 'text/event-stream' });
        expect(res.status).toBe(200);
        expect(res.headers.get('content-type')).toContain('text/event-stream');
        expect(res.headers.get('cache-control')).toContain('no-cache');
        expect(res.headers.get('cache-control')).toContain('no-transform');
        expect(res.headers.get('connection')).toBe('keep-alive');
        expect(res.headers.get('access-control-allow-origin')).toBe('*');
        // valid SSE: a comment line terminated by a blank line
        expect(firstFrame).toMatch(/^: [^\n]*\n\n/);
      }
    );

    it('streams even when the probe sends no Accept header (curl / naive validators)', async () => {
      const { res, firstFrame } = await openSseStream(`${base}/mcp`);
      expect(res.status).toBe(200);
      expect(res.headers.get('content-type')).toContain('text/event-stream');
      expect(firstFrame).toMatch(/^: /);
    });

    it('reflects the Origin on the stream response and exposes Mcp-Session-Id', async () => {
      const { res } = await openSseStream(`${base}/mcp`, { Accept: 'text/event-stream', Origin: 'https://poke.com' });
      expect(res.status).toBe(200);
      expect(res.headers.get('access-control-allow-origin')).toBe('https://poke.com');
      expect(res.headers.get('vary')).toContain('Origin');
      expect(res.headers.get('access-control-expose-headers')?.toLowerCase()).toContain('mcp-session-id');
    });

    it('still rejects a wrong bearer token on GET with 401 (auth is checked when presented)', async () => {
      const res = await fetch(`${base}/mcp`, { headers: { Accept: 'text/event-stream', Authorization: 'Bearer wrong' } });
      expect(res.status).toBe(401);
      expect(res.headers.get('www-authenticate')).toContain('invalid_token');
      expect(res.headers.get('access-control-allow-origin')).toBe('*');
    });
  });

  it('PUT /mcp is 405 with a JSON-RPC body and Allow listing GET/POST/DELETE', async () => {
    const res = await fetch(`${base}/mcp`, { method: 'PUT' });
    expect(res.status).toBe(405);
    expect(res.headers.get('allow')).toBe('GET, POST, DELETE');
    expect(await res.json()).toEqual({ jsonrpc: '2.0', error: { code: -32000, message: 'Method not allowed.' }, id: null });
  });

  describe('OPTIONS /mcp CORS preflight', () => {
    it.each(['/mcp', '/fantasy/mcp'])('%s reflects the origin and allows GET/POST/OPTIONS + MCP headers', async (path) => {
      const res = await fetch(`${base}${path}`, {
        method: 'OPTIONS',
        headers: { Origin: 'https://poke.com', 'Access-Control-Request-Method': 'GET', 'Access-Control-Request-Headers': 'authorization, mcp-session-id' },
      });
      expect(res.status).toBe(200);
      expect(res.headers.get('access-control-allow-origin')).toBe('https://poke.com');
      const methods = res.headers.get('access-control-allow-methods') ?? '';
      for (const m of ['GET', 'POST', 'OPTIONS']) expect(methods).toContain(m);
      const allowHeaders = (res.headers.get('access-control-allow-headers') ?? '').toLowerCase();
      for (const h of ['content-type', 'authorization', 'mcp-session-id']) expect(allowHeaders).toContain(h);
      expect(res.headers.get('access-control-max-age')).toBe('86400');
    });

    it('answers * when the preflight carries no Origin', async () => {
      const res = await fetch(`${base}/mcp`, { method: 'OPTIONS', headers: { 'Access-Control-Request-Method': 'POST' } });
      expect(res.status).toBe(200);
      expect(res.headers.get('access-control-allow-origin')).toBe('*');
    });
  });

  describe('initialize handshake', () => {
    it.each([
      ['no Authorization header', {}],
      ['valid bearer token', { Authorization: `Bearer ${TOKEN}` }],
    ])('returns 200 + SSE event with server capabilities (%s)', async (_label, headers) => {
      const res = await post(INITIALIZE, headers);
      expect(res.status).toBe(200);
      expect(res.headers.get('content-type')).toContain('text/event-stream');
      const [msg] = parseSse(await res.text());
      expect(msg.id).toBe(1);
      expect(msg.result).toMatchObject({
        protocolVersion: '2025-03-26',
        capabilities: { tools: { listChanged: true }, resources: { listChanged: true } },
        serverInfo: { name: 'fantasy-mcp' },
      });
    });

    it('still streams SSE when the client sends no Accept header', async () => {
      const res = await fetch(`${base}/mcp`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(INITIALIZE),
      });
      expect(res.status).toBe(200);
      expect(res.headers.get('content-type')).toContain('text/event-stream');
      expect(parseSse(await res.text())[0].result?.serverInfo).toMatchObject({ name: 'fantasy-mcp' });
    });

    it('accepts notifications/initialized with 202 and an empty body', async () => {
      const res = await post({ jsonrpc: '2.0', method: 'notifications/initialized' }, { Authorization: `Bearer ${TOKEN}` });
      expect(res.status).toBe(202);
      expect(await res.text()).toBe('');
    });

    it('rejects an invalid bearer token on initialize with 401 + WWW-Authenticate invalid_token', async () => {
      const res = await post(INITIALIZE, { Authorization: 'Bearer definitely-not-the-token' });
      expect(res.status).toBe(401);
      expect(res.headers.get('content-type')).toContain('application/json');
      const challenge = res.headers.get('www-authenticate') ?? '';
      expect(challenge).toMatch(/^Bearer realm="fantasy-mcp", resource="http:\/\/127\.0\.0\.1:\d+\/mcp"/);
      expect(challenge).toContain('error="invalid_token"');
      expect(await res.json()).toEqual({
        jsonrpc: '2.0',
        error: { code: -32001, message: 'Authentication required. Please provide a valid Bearer token.' },
        id: null,
      });
    });

    it('treats a malformed Authorization header (no Bearer scheme) as invalid', async () => {
      const res = await post(INITIALIZE, { Authorization: TOKEN });
      expect(res.status).toBe(401);
    });
  });

  describe('tools/list', () => {
    it('lists the full tool set without auth (public discovery)', async () => {
      const res = await post({ jsonrpc: '2.0', id: 2, method: 'tools/list' });
      expect(res.status).toBe(200);
      const [msg] = parseSse(await res.text());
      const tools = msg.result?.tools as Array<{ name: string; inputSchema: unknown; description: string }>;
      expect(tools.map((t) => t.name).sort()).toEqual(EXPECTED_TOOLS);
      for (const tool of tools) {
        expect(tool.description.length).toBeGreaterThan(0);
        expect(tool.inputSchema).toBeTruthy();
      }
    });

    it('lists the same tools with a valid token', async () => {
      const res = await post({ jsonrpc: '2.0', id: 2, method: 'tools/list' }, { Authorization: `Bearer ${TOKEN}` });
      expect(res.status).toBe(200);
      const [msg] = parseSse(await res.text());
      expect((msg.result?.tools as Array<{ name: string }>).map((t) => t.name).sort()).toEqual(EXPECTED_TOOLS);
    });
  });

  describe('tools/call auth', () => {
    const call = { jsonrpc: '2.0', id: 3, method: 'tools/call', params: { name: 'get_user_session', arguments: {} } };

    it('401 without a token, challenge has no error code (RFC 6750 §3.1)', async () => {
      const res = await post(call);
      expect(res.status).toBe(401);
      const challenge = res.headers.get('www-authenticate') ?? '';
      expect(challenge).toContain('realm="fantasy-mcp"');
      expect(challenge).not.toContain('error=');
      expect(((await res.json()) as { error: { code: number } }).error.code).toBe(-32001);
    });

    it('401 with a wrong token', async () => {
      const res = await post(call, { Authorization: 'Bearer wrong-token' });
      expect(res.status).toBe(401);
      expect(res.headers.get('www-authenticate')).toContain('error="invalid_token"');
    });

    it('200 with the operator token, reporting the env-configured Sleeper league and no ESPN', async () => {
      const res = await post(call, { Authorization: `Bearer ${TOKEN}` });
      expect(res.status).toBe(200);
      const [msg] = parseSse(await res.text());
      const structured = msg.result?.structuredContent as {
        totalLeaguesFound: number;
        allLeagues: Array<{ platform: string; leagueId: string }>;
      };
      expect(structured.totalLeaguesFound).toBe(1);
      expect(structured.allLeagues).toEqual([expect.objectContaining({ platform: 'sleeper', leagueId: '1200000000000000001' })]);
    });
  });

  describe('MCP SDK client (Streamable HTTP transport)', () => {
    it('completes initialize -> initialized -> tools/list -> tools/call with a bearer token', async () => {
      const client = new Client({ name: 'flaim-integration-test', version: '1.0.0' });
      const transport = new StreamableHTTPClientTransport(new URL(`${base}/mcp`), {
        requestInit: { headers: { Authorization: `Bearer ${TOKEN}` } },
      });
      await client.connect(transport);
      expect(client.getServerVersion()).toMatchObject({ name: 'fantasy-mcp' });
      expect(client.getServerCapabilities()?.tools).toBeDefined();

      const { tools } = await client.listTools();
      expect(tools.map((t) => t.name).sort()).toEqual(EXPECTED_TOOLS);

      const result = await client.callTool({ name: 'get_user_session', arguments: {} });
      expect(result.isError).toBeFalsy();
      expect((result.structuredContent as { totalLeaguesFound: number }).totalLeaguesFound).toBe(1);
      await client.close();
    });

    it('connects anonymously (public handshake) but tool calls surface the 401', async () => {
      const client = new Client({ name: 'flaim-anon-test', version: '1.0.0' });
      const transport = new StreamableHTTPClientTransport(new URL(`${base}/mcp`));
      await client.connect(transport);
      expect((await client.listTools()).tools.length).toBe(EXPECTED_TOOLS.length);
      await expect(client.callTool({ name: 'get_user_session', arguments: {} })).rejects.toThrow(/Authentication required/);
      await client.close();
    });
  });
});

describe('server.ts process boot (no leagues.json anywhere)', () => {
  let child: ChildProcess | undefined;
  let stderr = '';
  let stdout = '';

  afterAll(() => {
    child?.kill('SIGTERM');
  });

  it('starts, binds the configured host:port, serves /health and /mcp, and stops on SIGTERM', async () => {
    const port = await freePort();
    const serverEntry = resolve(__dirname, '..', 'server.ts');
    const tsx = resolve(__dirname, '..', '..', 'node_modules', '.bin', 'tsx');
    child = spawn(tsx, [serverEntry], {
      env: { PATH: process.env.PATH ?? '', ...PI_ENV, FLAIM_MCP_PORT: String(port), FLAIM_CACHE_DIR: join(tmpdir(), `flaim-it-cache-${port}`) },
      stdio: ['ignore', 'pipe', 'pipe'],
    });
    child.stdout!.on('data', (d) => (stdout += d.toString()));
    child.stderr!.on('data', (d) => (stderr += d.toString()));

    const exited = new Promise<{ code: number | null; signal: NodeJS.Signals | null }>((res) =>
      child!.once('exit', (code, signal) => res({ code, signal }))
    );
    const base = `http://127.0.0.1:${port}`;
    const deadline = Date.now() + 15_000;
    let health: Response | undefined;
    while (Date.now() < deadline && !health) {
      if (child.exitCode !== null) break;
      try {
        health = await fetch(`${base}/health`);
      } catch {
        await new Promise((r) => setTimeout(r, 200));
      }
    }
    expect(health, `server never came up.\nstdout:\n${stdout}\nstderr:\n${stderr}`).toBeDefined();
    expect(health!.status).toBe(200);
    expect(stdout).toContain(`listening on http://127.0.0.1:${port}/mcp`);
    expect(stdout).toContain('[no ESPN credentials]');
    expect(stderr).toContain('[selfhost] warning:');
    expect(stderr).not.toMatch(/Error:|Unhandled|TypeError|ConfigError/);

    const init = await fetch(`${base}/mcp`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Accept: 'application/json, text/event-stream', Authorization: `Bearer ${TOKEN}` },
      body: JSON.stringify(INITIALIZE),
    });
    expect(init.status).toBe(200);
    expect(parseSse(await init.text())[0].result?.serverInfo).toMatchObject({ name: 'fantasy-mcp' });

    const { res: sse, firstFrame } = await openSseStream(`${base}/mcp`, { Accept: 'text/event-stream' });
    expect(sse.status).toBe(200);
    expect(sse.headers.get('content-type')).toContain('text/event-stream');
    expect(firstFrame).toMatch(/^: /);

    child.kill('SIGTERM');
    const { code, signal } = await exited;
    // tsx re-raises the signal (exit 143) or exits 0; anything else is a crash.
    expect(signal === 'SIGTERM' || code === 143 || code === 0).toBe(true);
  }, 30_000);
});
