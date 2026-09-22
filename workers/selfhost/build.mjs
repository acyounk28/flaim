// Bundles the self-hosted gateway into a single ESM file so the runtime image
// needs only Node (no pnpm workspace, no TypeScript toolchain).
import { build } from 'esbuild';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const here = path.dirname(fileURLToPath(import.meta.url));

await build({
  entryPoints: [path.join(here, 'src/server.ts')],
  bundle: true,
  platform: 'node',
  target: 'node24',
  format: 'esm',
  outfile: path.join(here, 'dist/server.mjs'),
  sourcemap: true,
  alias: { '@flaim/worker-shared': path.join(here, '../shared/src/index.ts') },
  banner: {
    js: "import { createRequire } from 'node:module'; const require = createRequire(import.meta.url);",
  },
  logLevel: 'info',
});
