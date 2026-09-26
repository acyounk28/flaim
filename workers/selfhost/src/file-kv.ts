// workers/selfhost/src/file-kv.ts
//
// Minimal KVNamespace-compatible store backed by a directory on disk. Only the
// subset used by espn-client / sleeper-client player caches is implemented
// (string get/put with expirationTtl, delete). Values persist across container
// restarts so a Raspberry Pi does not re-download multi-megabyte player maps
// every boot.
import { mkdirSync, readFileSync, writeFileSync, unlinkSync, existsSync } from 'node:fs';
import { join } from 'node:path';
import { createHash } from 'node:crypto';

interface Envelope {
  expiresAt: number | null;
  value: string;
}

export class FileKV {
  private readonly memory = new Map<string, Envelope>();
  private readonly dir: string | null;

  constructor(dir: string | null) {
    this.dir = dir;
    if (dir) {
      try {
        mkdirSync(dir, { recursive: true });
      } catch (error) {
        console.warn(`[selfhost] cache dir ${dir} is not writable (${error instanceof Error ? error.message : String(error)}); using in-memory cache only`);
        this.dir = null;
      }
    }
  }

  private filePath(key: string): string | null {
    if (!this.dir) return null;
    const hash = createHash('sha1').update(key).digest('hex');
    return join(this.dir, `${hash}.json`);
  }

  private load(key: string): Envelope | null {
    const inMemory = this.memory.get(key);
    if (inMemory) return inMemory;
    const path = this.filePath(key);
    if (!path || !existsSync(path)) return null;
    try {
      const envelope = JSON.parse(readFileSync(path, 'utf8')) as Envelope;
      this.memory.set(key, envelope);
      return envelope;
    } catch {
      return null;
    }
  }

  async get(key: string): Promise<string | null> {
    const envelope = this.load(key);
    if (!envelope) return null;
    if (envelope.expiresAt !== null && envelope.expiresAt <= Date.now()) {
      await this.delete(key);
      return null;
    }
    return envelope.value;
  }

  async put(key: string, value: string, options?: { expirationTtl?: number; expiration?: number }): Promise<void> {
    let expiresAt: number | null = null;
    if (options?.expirationTtl) expiresAt = Date.now() + options.expirationTtl * 1000;
    else if (options?.expiration) expiresAt = options.expiration * 1000;
    const envelope: Envelope = { expiresAt, value };
    this.memory.set(key, envelope);
    const path = this.filePath(key);
    if (path) {
      try {
        writeFileSync(path, JSON.stringify(envelope));
      } catch (error) {
        console.warn(`[selfhost] failed to persist cache key ${key}:`, error);
      }
    }
  }

  async delete(key: string): Promise<void> {
    this.memory.delete(key);
    const path = this.filePath(key);
    if (path && existsSync(path)) {
      try {
        unlinkSync(path);
      } catch {
        /* ignore */
      }
    }
  }
}

/** Cast helper: the workers only touch get/put/delete, so FileKV is a safe stand-in. */
export function asKVNamespace(kv: FileKV): KVNamespace {
  return kv as unknown as KVNamespace;
}
