# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

Translation Memory Atlas — a single-page React app that visualizes English/中文 subtitle pairs from a SQLite database as an interactive 3D semantic landscape. Users can search by meaning (via an embedding model running in-browser), explore clustered "islands" of related lines, and view full video transcripts.

## Commands

```bash
npm run dev          # Start Vite dev server
npm run build        # Type-check (tsc -b) then bundle (vite build)
npm run preview      # Preview production build locally
npm run generate:semantic-landscape   # Regenerate public/data/startup-visualizations.json from the SQLite DB
```

No test suite or linter config is set up in the project. Type-check with `npx tsc --noEmit`.

## Architecture

**Frontend stack:** React 19 + TypeScript + Vite. No router, no state library — the entire app is a single `App` component that delegates rendering to `TmAtlasPanel`.

**Data flow:**
1. On mount, `App` spawns a Web Worker (`search/search.worker.ts`) and sends a `boot` message with the SQLite DB URL.
2. The worker downloads `public/data/tm_misha_minilm.db` (a sql.js/WASM SQLite database), reads all `tm_main` rows and `tm_vectors` embeddings into memory, then signals `boot:ok`.
3. Separately, `App` fetches `public/data/startup-visualizations.json` (pre-computed UMAP projections + cluster labels) for the 3D atlas canvas.
4. Searches go worker-ward: the worker loads `Xenova/all-MiniLM-L6-v2` via `@huggingface/transformers` (ONNX Runtime WASM), embeds the query, and does cosine-similarity ranking against stored vectors.

**Key source files:**
- `src/App.tsx` — top-level state, worker lifecycle, search/context/transcript orchestration
- `src/startup/TmAtlasPanel.tsx` — the entire UI: 3D canvas renderer, sidebar, search results, island browser, transcript panel (~1400 lines, no subcomponents)
- `src/search/protocol.ts` — TypeScript interfaces for all worker request/response messages
- `src/search/search.worker.ts` — Web Worker: SQLite loading, embedding model init, semantic search, context windowing, transcript retrieval
- `src/startup/semantic-landscape.ts` — types for the pre-computed visualization JSON
- `src/startup/colors.ts` — hex/rgba color utilities for the canvas renderer
- `src/classes.ts` — `classNames()` helper
- `src/keyboard.ts` — keyboard event helper for accessibility
- `src/format.ts` — model name display formatting

**Worker protocol:** All worker communication uses a discriminated union on `kind` (see `protocol.ts`). Each request gets a monotonically increasing `requestId`; responses carry the same ID so the main thread can resolve the matching `Promise`.

**Asset versioning:** `vite.config.ts` stamps the DB and startup-visualizations JSON with `size-mtime` query params (`__TM_DB_VERSION__`, `__TM_STARTUP_DATA_VERSION__`) to bust browser caches on deploy.

**Data generation:** `scripts/generate_semantic_landscape.mjs` reads the SQLite DB, runs UMAP (2D + 3D) on the MiniLM vectors, clusters via mutual-KNN islands, assigns colors, and writes `startup-visualizations.json`. Run it after updating the DB.

**Deployment:** GitHub Actions builds with `npm ci && npm run build` and deploys to GitHub Pages. `VITE_BASE_PATH` is set to `/<repo-name>/` for correct asset paths.

## Conventions

- Strict TypeScript: `noUncheckedIndexedAccess`, `noUnusedLocals`, `noUnusedParameters` all enabled in `tsconfig.app.json`.
- No CSS modules or utility framework — all styles live in `src/styles.css` as plain CSS with CSS custom properties for theming (dark/light via `data-theme` attribute on `<html>`).
- The 3D canvas is drawn with `<canvas>` 2D context (not WebGL). Projection math (rotation, perspective, depth sorting) is done manually in `TmAtlasPanel.tsx`.
- The sidebar overlays the canvas absolutely — it is not part of the CSS grid flow on desktop.
