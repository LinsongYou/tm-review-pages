import { statSync } from 'node:fs';
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

function normalizeBase(input: string | undefined): string {
  if (!input || input === '/') {
    return '/';
  }

  const trimmed = input.trim();
  const withLeading = trimmed.startsWith('/') ? trimmed : `/${trimmed}`;
  return withLeading.endsWith('/') ? withLeading : `${withLeading}/`;
}

function getAssetVersion(path: string): string {
  try {
    const stats = statSync(path);
    return `${stats.size}-${Math.round(stats.mtimeMs)}`;
  } catch {
    return 'dev';
  }
}

export default defineConfig({
  plugins: [react()],
  base: normalizeBase(process.env.VITE_BASE_PATH),
  define: {
    __TM_DB_VERSION__: JSON.stringify(getAssetVersion('public/data/tm_misha_minilm.db')),
  },
});
