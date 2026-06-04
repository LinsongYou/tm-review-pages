# AGENTS.md

## What This Is

Translation Memory Atlas — a single-page React app that visualizes English/中文 subtitle pairs from a SQLite database as an interactive 3D semantic landscape. Users can search by meaning (via an embedding model running in-browser), explore clustered "islands" of related lines, and view full video transcripts.

## Repo Role

- This repo is the deployed static GitHub Pages app for reviewing translation-memory data in the browser.
- The app is read-only and client-side only.
- TM data is generated upstream in `D:\subtitle-workflow-pipeline` and copied into this repo as build input.

## Commands

```bash
npm run dev                  # Start Vite dev server
npm run build                # Type-check (tsc -b) then bundle (vite build)
npm run preview              # Preview production build locally
npm run generate:tm-atlas    # Regenerate public/data/tm-atlas.json from the SQLite DB
```

No test suite or linter config is set up. Type-check with `npx tsc --noEmit`.

## Stack

Vite 7.3, React 19, TypeScript 5.9, `sql.js`, `@huggingface/transformers`, ONNX Runtime Web.

## Architecture

**No router, no state library** — the entire app is a single `App` component that delegates rendering to `TmAtlasPanel`.

**Data flow:**
1. On mount, `App` spawns a Web Worker (`search/search.worker.ts`) and sends a `boot` message with the SQLite DB URL.
2. The worker downloads `public/data/tm_misha_minilm.db` (a sql.js/WASM SQLite database), reads all `tm_main` rows and `tm_vectors` embeddings into memory, then signals `boot:ok`.
3. Separately, `App` fetches `public/data/tm-atlas.json` (pre-computed UMAP projections + cluster labels) for the 3D atlas canvas.
4. Searches go worker-ward: the worker loads `Xenova/all-MiniLM-L6-v2` via `@huggingface/transformers` (ONNX Runtime WASM), embeds the query, and does cosine-similarity ranking against stored vectors.

**Worker protocol:** All worker communication uses a discriminated union on `kind` (see `protocol.ts`). Each request gets a monotonically increasing `requestId`; responses carry the same ID so the main thread can resolve the matching `Promise`.

**Asset versioning:** `vite.config.ts` stamps the DB and atlas JSON with `size-mtime` query params (`__TM_DB_VERSION__`, `__TM_ATLAS_DATA_VERSION__`) to bust browser caches on deploy.

**Deployment:** GitHub Actions (`.github/workflows/deploy.yml`) builds with `npm ci && npm run build` and deploys to GitHub Pages. `VITE_BASE_PATH` is set to `/<repo-name>/` for correct asset paths.

## Atlas & Island System

- All TM entries are projected into 3D space via UMAP and rendered on an HTML `<canvas>` with 2D context (not WebGL). Projection math (rotation, perspective, depth sorting) is done manually in `TmAtlasPanel.tsx`.
- Entries are grouped into semantic clusters ("islands") via mutual-kNN clustering.
- Each island has a label, description, color, top phrases, theme confidence, and medoid entry.
- **Island Browser** in the sidebar lists all islands sorted by size; selecting one focuses the camera and opens the Island Focus Panel.
- **Island Focus Panel** shows cluster metrics and a scrollable list of all entries in that island.
- **Video Trace** draws a path through all entries of a video on the atlas when a transcript is open.
- Camera transitions (island focus, entry focus) are smoothly animated via `requestAnimationFrame` lerping.

## UI Flow

- **Atlas view (home):** 3D point cloud with island browser in sidebar. HUD shows load status chips (Pairs count, Model name, Video count), Reset button, Dark/Light theme toggle.
- **Search:** semantic search input in sidebar; top 24 results shown with score, video ID, segment index. Search hits are connected by dashed lines on the atlas.
- **Island focus:** select an island to see its details, metrics, and entry list. Camera pans to cluster.
- **Entry detail:** shows cluster info, video ID, EN/ZH text, cluster phrases, and local context (neighboring entries from same video).
- **Transcript:** full video transcript panel with timestamp buttons, EN/ZH text per cue, search-from-line buttons.
- **Tooltip:** follows cursor on hover, showing entry ID and English text.
- **Interaction:** drag-to-rotate, scroll-to-zoom, Ctrl+drag to pan, click to select entries/islands.

## Visual Design

- "Holographic command center" aesthetic with scanline overlay, glassmorphism (`backdrop-filter: blur(20px) saturate(160%)`), green accent glow effects, and custom scrollbars.
- Google Fonts: Rajdhani, Outfit, IBM Plex Mono.
- Hand-written CSS (no Tailwind). All styles in `src/styles.css`.
- Responsive breakpoints at 980px and 640px.
- Dark/light theme toggle persisted to localStorage.

## Data Contract

- Treat TM assets in this repo as generated input, not hand-edited source.
- When refreshing data, update `public/data/tm_misha_minilm.db` from the pipeline repo and regenerate `public/data/tm-atlas.json` via `npm run generate:tm-atlas`.
- If raw SQLite becomes too heavy for Pages, prefer slimmer exported read-only artifacts from the pipeline repo rather than adding server infrastructure.

## Key Files

- `src/App.tsx` — application shell, worker lifecycle, theme, search/selection/transcript state management.
- `src/atlas/TmAtlasPanel.tsx` — the entire UI: 3D canvas renderer, sidebar, search results, island browser, transcript panel. Largest and most complex file; prefer targeted edits over broad refactors.
- `src/search/search.worker.ts` — Web Worker for DB boot, model loading, semantic search, context lookup, transcript lookup.
- `src/search/protocol.ts` — TypeScript interfaces for all worker request/response messages.
- `src/atlas/semantic-landscape.ts` — TypeScript interfaces for the atlas JSON data shape (`SemanticLandscapeData`, `SemanticLandscapeCluster`, `SemanticLandscapePoint`, `InitialView`).
- `src/atlas/colors.ts` — hex/rgba color utilities (`hexToRgba`, `blendHexColors`, `isLightHex`).
- `src/classes.ts` — `classNames()` helper.
- `src/keyboard.ts` — keyboard event helper for accessibility.
- `src/format.ts` — model name display formatting.
- `src/styles.css` — all CSS styles with custom properties for theming.
- `scripts/generate_tm_atlas.mjs` — Node.js script to rebuild `tm-atlas.json` from the DB (runs UMAP 2D+3D, mutual-KNN clustering, color assignment).
- `vite.config.ts` — React plugin, GitHub Pages base path, asset versioning via compile-time constants.
- `.github/workflows/deploy.yml` — GitHub Actions build and deploy workflow for Pages.

## Conventions

- LF line endings exclusively — no CRLF. Enforced by `.gitattributes` (`* text=auto eol=lf`).
- Strict TypeScript: `noUncheckedIndexedAccess`, `noUnusedLocals`, `noUnusedParameters` all enabled in `tsconfig.app.json`.
- No CSS modules or utility framework — all styles in `src/styles.css` as plain CSS with CSS custom properties for theming (dark/light via `data-theme` attribute on `<html>`).
- The sidebar overlays the canvas absolutely — it is not part of the CSS grid flow on desktop.

## Hard Constraints

- No editing UI.
- No writes back to SQLite or any other source artifact.
- No backend, serverless API, auth, or admin workflow.
- Keep the site deployable as plain static files on GitHub Pages.

## Commit Hygiene

Before committing any change, always verify that the build is clean:

```bash
npx tsc --noEmit        # Type-check
npm run build            # Full build (type-check + Vite bundle)
```

Do not commit if either command fails. Fix the errors first.

## Future Change Guidance

- Keep this repo focused on the web app. Do not pull general pipeline logic here unless the UI strictly needs it.
- Favor correctness, inspectability, and simple static architecture over clever infrastructure.
- When changing data shape, update both the worker loader and the landscape-generation script.
- Preserve cache-friendly static asset URLs and versioning behavior for the DB and atlas JSON.
- Canvas rendering and camera animation are tightly coupled — test visual behavior after changing projection, zoom, or focus logic.
