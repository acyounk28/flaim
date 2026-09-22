import { defineConfig } from "vitest/config";
import path from "node:path";

export default defineConfig({
  resolve: { alias: { "@flaim/worker-shared": path.resolve(__dirname, "../shared/src") } },
  test: { environment: "node", include: ["src/**/*.test.ts"], testTimeout: 20000 },
});
