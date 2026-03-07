# TM Review Pages

Static GitHub Pages app for reviewing translation-memory data in the browser.

## Stack

- Vite
- React + TypeScript
- `sql.js` for browser-side SQLite
- `@huggingface/transformers` for browser-side MiniLM query embeddings
- Web Worker for SQLite loading, lexical search, semantic search, and context lookup

## Current Prototype

- Loads `public/data/tm_misha_minilm.db` directly in the browser.
- Supports lexical search for English and Chinese.
- Supports semantic search for English using the shipped `sentence-transformers/all-MiniLM-L6-v2` vectors.
- Shows top-k matches with score, `video_id#seg_index`, EN/ZH text, and a local `±2` context panel.
- Runs as a fully static site with no backend.

Chinese semantic indexing is intentionally not shipped yet. The current database only contains English MiniLM vectors, so semantic mode is English-only in this first build.

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

If the data changes shape or raw SQLite becomes awkward on Pages, the next step is to export slimmer read-only artifacts from the pipeline repo and keep this app unchanged at the UI layer.

## Deployment

The repo includes a GitHub Actions workflow that builds the static site and deploys `dist/` to GitHub Pages.
