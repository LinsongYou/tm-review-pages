# Translation Memory Pages

Static GitHub Pages app for reviewing translation-memory data in the browser.

## Stack

- Vite
- React + TypeScript
- `sql.js` for browser-side SQLite
- `@huggingface/transformers` for browser-side MiniLM query embeddings
- Web Worker for SQLite loading, semantic search, and context lookup

## Current Prototype

- Loads `public/data/tm_misha_minilm.db` directly in the browser.
- Loads a precomputed `public/data/startup-visualizations.json` startup artifact for the home-page startup visualizations.
- Uses a single English semantic search flow powered by the shipped `sentence-transformers/all-MiniLM-L6-v2` vectors.
- Keeps the DB and startup JSON on stable URLs with asset-version query strings so repeat visits can reuse browser cache until the source data changes.
- Defers MiniLM model initialization until the first semantic search instead of blocking the initial app boot.
- Includes adjustable `Top K`, `Min Chars`, `Score`, and context-radius controls.
- Shows result cards with YouTube ID, transcript entry number, score, block ID, and emphasized EN/ZH text.
- Opens a full-video transcript modal when you click a YouTube ID, using the same context-style row presentation.
- Runs as a fully static site with no backend.

The current UI intentionally keeps search to a single English semantic query path. Results still display both EN and ZH text for review.

## Startup Panels

The home screen currently renders three startup panels when data is available:

- `SemanticLandscapePanel`
- `CueTimeDistributionPanel`
- `VideoFingerprintWallPanel`

Their precomputed data lives in `public/data/startup-visualizations.json`, while cue-time distribution data comes from the browser-side DB boot path.

## Local Development

```bash
npm install
npm run dev
```

The app expects the prototype database at `public/data/tm_misha_minilm.db`.

## Production Build

```bash
npm run build
npm run preview
```

For GitHub Pages builds, set `VITE_BASE_PATH` to the repo path prefix. The included workflow sets:

```bash
VITE_BASE_PATH=/${REPO_NAME}/
```

## Refreshing Data

Copy the latest prototype DB from the pipeline repo:

```powershell
New-Item -ItemType Directory -Force .\public\data
Copy-Item D:\subtitle-workflow-pipeline\tm_tools\tm_misha_minilm.db .\public\data\tm_misha_minilm.db -Force
```

Then regenerate the startup visualization payload:

```powershell
npm run generate:startup-visualizations
```

If the data changes shape or raw SQLite becomes awkward on Pages, the next step is to export slimmer read-only artifacts from the pipeline repo and keep this app unchanged at the UI layer.

## Deployment

The repo includes a GitHub Actions workflow that builds the static site and deploys `dist/` to GitHub Pages.
