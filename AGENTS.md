# AGENTS.md

## Repo Role
- This repo is the deployed static GitHub Pages app for reviewing translation-memory data in the browser.
- The app is read-only and client-side only.
- TM data is generated upstream in `D:\subtitle-workflow-pipeline` and copied into this repo as build input.

## Current State
- The app is already scaffolded, built, and deployed through GitHub Pages.
- Stack: Vite, React, TypeScript, `sql.js`, `@huggingface/transformers`.
- `public/data/tm_misha_minilm.db` is shipped directly as a static asset and opened in the browser.
- `public/data/semantic-landscape.json` is a generated startup artifact used for the landing-page visualizations.
- A Web Worker handles DB loading, query-model loading, semantic search, context lookup, and full-transcript lookup.
- Search is English semantic search only.
- The current search flow embeds queries with `Xenova/all-MiniLM-L6-v2` and matches against stored `sentence-transformers/all-MiniLM-L6-v2` vectors from the TM DB.

## Implemented Review Flow
- Search controls: query, `Top K`, `Min Chars`, `Score`, and context radius.
- Result cards show score, `video_id#seg_index` identity, block ID, and EN/ZH text.
- Selecting a result shows local context from the same video.
- Clicking a video ID opens a full-video transcript modal.
- Startup/home panels currently include the semantic landscape, cue-time distribution, and video fingerprint wall when data is available.

## Hard Constraints
- No editing UI.
- No writes back to SQLite or any other source artifact.
- No ArcTime integration.
- No backend, serverless API, auth, or admin workflow.
- Keep the site deployable as plain static files on GitHub Pages.

## Data Contract
- Treat TM assets in this repo as generated input, not hand-edited source.
- When refreshing data, update `public/data/tm_misha_minilm.db` from the pipeline repo and regenerate `public/data/semantic-landscape.json`.
- If raw SQLite becomes too heavy for Pages, prefer slimmer exported read-only artifacts from the pipeline repo rather than adding server infrastructure.

## Key Files
- `src/App.tsx`: main UI, startup panels, search flow, context pane, transcript modal.
- `src/search/search.worker.ts`: browser-side DB boot, model loading, search, context, transcript lookup.
- `src/search/protocol.ts`: worker message contracts.
- `scripts/generate_semantic_landscape.py`: rebuilds the startup visualization data from the DB.
- `vite.config.ts`: GitHub Pages base path and static-asset versioning.
- `.github/workflows/deploy.yml`: build and deploy workflow for GitHub Pages.

## Future Change Guidance
- Keep this repo focused on the web app. Do not pull general pipeline logic here unless the UI strictly needs it.
- Favor correctness, inspectability, and simple static architecture over clever infrastructure.
- When changing data shape, update both the worker loader and the landscape-generation script.
- Preserve cache-friendly static asset URLs and versioning behavior for the DB and startup JSON.
