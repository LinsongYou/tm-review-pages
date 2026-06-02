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
- Loads a precomputed `public/data/tm-atlas.json` atlas artifact for the 3D UMAP point cloud and island metadata.
- Uses a single English semantic search flow powered by the shipped `sentence-transformers/all-MiniLM-L6-v2` vectors.
- Keeps the DB and atlas JSON on stable URLs with asset-version query strings so repeat visits can reuse browser cache until the source data changes.
- Defers MiniLM model initialization until the first semantic search instead of blocking the initial app boot.
- Renders all TM entries as an interactive 3D canvas with island focus, semantic search, entry detail, and transcript views.
- Runs as a fully static site with no backend.

The current UI intentionally keeps search to a single English semantic query path. Results still display both EN and ZH text for review.

## Atlas Data

The atlas data in `public/data/tm-atlas.json` contains precomputed 2D/3D UMAP coordinates, mutual-kNN island assignments, island labels, colors, medoids, and cluster metrics. The browser loads it as a static asset and draws the atlas client-side.

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

Then regenerate the atlas payload:

```powershell
npm run generate:tm-atlas
```

If the data changes shape or raw SQLite becomes awkward on Pages, the next step is to export slimmer read-only artifacts from the pipeline repo and keep this app unchanged at the UI layer.

## Deployment

The repo includes a GitHub Actions workflow that builds the static site and deploys `dist/` to GitHub Pages.
