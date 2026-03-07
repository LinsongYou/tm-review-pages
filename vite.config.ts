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

export default defineConfig({
  plugins: [react()],
  base: normalizeBase(process.env.VITE_BASE_PATH),
});
