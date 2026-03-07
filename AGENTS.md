# AGENTS.md

## Project Purpose
- This repo is a GitHub Pages app for reviewing translation-memory data in the browser.
- The app is read-only. It must support reviewing/searching like `tm_tools/review_memory.py`, but it must not support editing, SQLite writes, ArcTime patching, or any server-side workflow.
- Source TM data is prepared in `D:\subtitle-workflow-pipeline` and copied or exported into this repo.

## Product Scope
- Required:
  - Search TM entries by semantic similarity.
  - Search TM entries by lexical match.
  - Toggle search language between English and Chinese.
  - Show top-k results with score, `video_id#seg_index`, and EN/ZH text.
  - Show local context around a selected result (equivalent to `/ctx`).
  - Support minimum-length filtering.
  - Work as a static GitHub Pages site with no backend.
- Not allowed:
  - No editing UI.
  - No ArcTime integration.
  - No writes back to SQLite.
  - No auth or admin features.
  - No server dependency for search or embedding.

## Current Data Decision
- Current embedding model for the browser target is `sentence-transformers/all-MiniLM-L6-v2`.
- Current TM database expected for the first prototype is `tm_misha_minilm.db` from the pipeline repo.
- First implementation may ship the SQLite DB itself as a static asset because it is small enough for a prototype.
- If performance or bundle shape becomes a problem, the next step is to export slimmer read-only artifacts instead of shipping raw SQLite.

## Recommended Stack
- Build tool: Vite.
- UI: React + TypeScript.
- Styling: plain CSS or lightweight CSS modules. Do not introduce a large UI framework unless there is a clear need.
- Browser DB access: `sql.js` or `wa-sqlite`.
- Browser embeddings: `@huggingface/transformers`.
- Heavy search/model work should run in a Web Worker.
- Keep the app deployable as plain static files for GitHub Pages.

## Architecture Plan
### Phase 1: Repo Scaffold
- Initialize a Vite React TypeScript app.
- Configure base path and static asset handling for GitHub Pages.
- Add a minimal README with local dev and deploy instructions.
- Decide whether the repo will publish from the root or from a build artifact branch/workflow.

### Phase 2: Data Loading
- Place the first review dataset under a public data path, for example `public/data/tm_misha_minilm.db`.
- Implement a browser-side loader that:
  - downloads the DB once,
  - opens it with SQLite WASM,
  - reads `tm_main` and `tm_vectors`,
  - builds in-memory lookup structures for review/search.
- Cache the downloaded asset in the browser when practical.

### Phase 3: Search Engine
- Recreate the useful read-only behavior of `review_memory.py`:
  - semantic search over stored EN vectors,
  - lexical search over active language text,
  - top-k selection,
  - minimum-length filter,
  - EN/ZH language toggle,
  - context window around a chosen hit.
- Do not recreate the REPL literally. Convert commands into explicit UI controls.
- Search behavior should be deterministic and easy to inspect.

### Phase 4: Browser Embedding Query Path
- Load the MiniLM model in-browser with `@huggingface/transformers`.
- Generate query embeddings client-side.
- Keep the first version focused on English semantic search if needed.
- If Chinese semantic search is added, decide whether to:
  - compute ZH query/document embeddings in-browser, or
  - precompute additional ZH vectors offline and ship them.
- Prefer offline precomputation for corpus-scale work over runtime indexing in the browser.

### Phase 5: UI
- Build a single-page review interface with:
  - search box,
  - semantic vs lexical mode toggle,
  - EN/ZH toggle,
  - top-k control,
  - minimum-length control,
  - results table/list,
  - detail/context panel.
- Optimize for scanability and keyboard use.
- Favor clarity over decorative UI.

### Phase 6: Performance Work
- Measure:
  - DB download size,
  - model download size,
  - cold-start time,
  - first-query latency,
  - repeat-query latency.
- If raw SQLite is too slow or awkward, move to exported static files:
  - metadata JSON/NDJSON,
  - vector binary blobs,
  - optional prebuilt lookup maps.
- Keep all optimization choices compatible with GitHub Pages.

### Phase 7: Deployment
- Add a GitHub Pages deployment workflow.
- Document how to refresh app data from the pipeline repo.
- Keep deployment fully static.
- Do not add a backend unless the project direction changes explicitly.

## Implementation Priorities
1. Get a minimal static app running on GitHub Pages.
2. Load the TM data successfully in-browser.
3. Implement lexical review mode first so the page is useful before semantic search is finished.
4. Add semantic search with MiniLM.
5. Add context view and polish.
6. Optimize bundle/data shape only after the end-to-end flow works.

## Data Flow Contract
- Upstream repo: `D:\subtitle-workflow-pipeline`.
- This repo should treat TM data as generated input, not as manually edited source.
- If a future export script is needed, it should live in the pipeline repo unless there is a strong reason to move it here.

## Future Session Guidance
- Keep this repo focused on the web app only.
- Do not pull pipeline scripts into this repo unless strictly needed.
- Avoid overengineering the first version.
- For the first pass, correctness and inspectability matter more than perfect polish.
- If a tradeoff is needed, prefer a simpler static architecture over a clever but fragile one.

## Suggested First Build Order
1. Scaffold Vite + React + TypeScript.
2. Add GitHub Pages-compatible build config.
3. Add a local DB loader proof of concept.
4. Render a simple table of TM rows.
5. Add lexical search and context.
6. Add semantic search via MiniLM.
7. Add caching/performance improvements.
