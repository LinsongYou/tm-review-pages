# AGENTS.md

## Repo Role
- This repo is the deployed static GitHub Pages app for reviewing translation-memory data in the browser.
- The app is read-only and client-side only.
- TM data is generated upstream in `D:\subtitle-workflow-pipeline` and copied into this repo as build input.

## Current State
- The app is already scaffolded, built, and deployed through GitHub Pages.
- Stack: Vite 7.3, React 19, TypeScript 5.9, `sql.js`, `@huggingface/transformers`, ONNX Runtime Web.
- The primary interface is a **3D UMAP atlas** (`TmAtlasPanel`) that renders all TM entries as a navigable point cloud.
- `public/data/tm_misha_minilm.db` is shipped directly as a static asset and opened in the browser via sql.js WASM.
- `public/data/startup-visualizations.json` contains precomputed UMAP projections and cluster data for the atlas.
- A Web Worker handles DB loading, query-model loading, semantic search, context lookup, and full-transcript lookup.
- Search is English semantic search only, using `Xenova/all-MiniLM-L6-v2` at query time matched against stored `sentence-transformers/all-MiniLM-L6-v2` vectors from the DB.

## Atlas & Island System
- All TM entries are projected into 3D space via UMAP and rendered on an HTML `<canvas>` with depth-based perspective.
- Entries are grouped into semantic clusters ("islands") via mutual-kNN clustering.
- Each island has a label, description, color, top phrases, theme confidence, and medoid entry.
- **Island Browser** in the sidebar lists all islands sorted by size; selecting one focuses the camera and opens the Island Focus Panel.
- **Island Focus Panel** shows cluster metrics and a scrollable list of all entries in that island.
- **Video Trace** draws a path through all entries of a video on the atlas when a transcript is open.
- Camera transitions (island focus, entry focus) are smoothly animated via `requestAnimationFrame` lerping.

## Implemented Review Flow
- **Atlas view (home):** 3D point cloud with island browser in sidebar. HUD shows load status chips (Pairs count, Model name, Video count), Reset button, Dark/Light theme toggle.
- **Search:** semantic search input in sidebar; top 12 results shown with score, video ID, segment index. Search hits are connected by dashed lines on the atlas.
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

## Hard Constraints
- No editing UI.
- No writes back to SQLite or any other source artifact.
- No ArcTime integration.
- No backend, serverless API, auth, or admin workflow.
- Keep the site deployable as plain static files on GitHub Pages.

## Data Contract
- Treat TM assets in this repo as generated input, not hand-edited source.
- When refreshing data, update `public/data/tm_misha_minilm.db` from the pipeline repo and regenerate `public/data/startup-visualizations.json` via `npm run generate:startup-visualizations`.
- If raw SQLite becomes too heavy for Pages, prefer slimmer exported read-only artifacts from the pipeline repo rather than adding server infrastructure.

## Key Files
- `src/App.tsx`: application shell, worker lifecycle, theme, search/selection/transcript state management.
- `src/startup/TmAtlasPanel.tsx`: the main 3D atlas canvas + sidebar UI (island browser, search results, entry detail, transcript panel). Largest file (~1393 lines).
- `src/search/search.worker.ts`: Web Worker for DB boot, model loading, semantic search, context lookup, transcript lookup.
- `src/search/protocol.ts`: worker message type contracts.
- `src/startup/semantic-landscape.ts`: TypeScript interfaces for the startup visualization data shape.
- `src/startup/colors.ts`: hex-to-rgba utility.
- `src/styles.css`: all CSS styles.
- `scripts/generate_semantic_landscape.mjs`: Node.js script to rebuild startup-visualizations.json from the DB (current).
- `scripts/generate_semantic_landscape.py`: Python version of the same (older, still functional).
- `vite.config.ts`: React plugin, GitHub Pages base path, asset versioning via compile-time constants.
- `.github/workflows/deploy.yml`: GitHub Actions build and deploy workflow for Pages.

## Future Change Guidance
- Keep this repo focused on the web app. Do not pull general pipeline logic here unless the UI strictly needs it.
- Favor correctness, inspectability, and simple static architecture over clever infrastructure.
- When changing data shape, update both the worker loader and the landscape-generation script.
- Preserve cache-friendly static asset URLs and versioning behavior for the DB and startup JSON.
- `TmAtlasPanel.tsx` is the largest and most complex file; prefer targeted edits over broad refactors.
- Canvas rendering and camera animation are tightly coupled -- test visual behavior after changing projection, zoom, or focus logic.
