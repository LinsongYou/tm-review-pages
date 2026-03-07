import initSqlJs from 'sql.js';
import sqlWasmUrl from 'sql.js/dist/sql-wasm.wasm?url';
import { env, pipeline } from '@huggingface/transformers';
import type {
  BootRequest,
  BootStats,
  ContextItem,
  ContextRequest,
  EntrySummary,
  ErrorResponse,
  ModelStatus,
  SearchLanguage,
  SearchRequest,
  SearchResult,
  StatusResponse,
  WorkerRequest,
  WorkerResponse,
} from './protocol';

const workerScope = self as DedicatedWorkerGlobalScope;
const TOKEN_RE = /[\p{Letter}\p{Number}_]+/gu;
const MODEL_LANGUAGE_NOTE =
  'The shipped semantic index is English-only in this prototype. Switch to English or use lexical search for Chinese.';

type SqlBlob = Uint8Array;
type ExtractorOutput = { data: Float32Array | number[] };
type Extractor = (input: string, options?: Record<string, unknown>) => Promise<ExtractorOutput>;

interface Entry extends EntrySummary {
  enNorm: string;
  zhNorm: string;
  enLength: number;
  zhLength: number;
}

interface RankedHit {
  entryIndex: number;
  score: number;
}

interface LoadedState {
  entries: Entry[];
  entryById: Map<string, number>;
  videoGroups: Map<string, number[]>;
  semanticEntryIndexes: Int32Array;
  semanticVectors: Float32Array;
  vectorDim: number;
  vectorModelId: string;
  queryModelId: string;
}

let state: LoadedState | null = null;
let extractor: Extractor | null = null;
let modelStatus: ModelStatus = 'idle';

env.allowLocalModels = false;
env.useBrowserCache = true;

if (env.backends.onnx.wasm) {
  env.backends.onnx.wasm.wasmPaths = undefined;
  env.backends.onnx.wasm.numThreads = 1;
  env.backends.onnx.wasm.proxy = false;
}

function post(message: WorkerResponse): void {
  workerScope.postMessage(message);
}

function postStatus(
  requestId: number,
  scope: StatusResponse['scope'],
  message: string,
  nextModelStatus?: ModelStatus,
): void {
  if (nextModelStatus) {
    modelStatus = nextModelStatus;
  }

  post({
    kind: 'status',
    requestId,
    scope,
    message,
    modelStatus,
  });
}

function postError(requestId: number, error: unknown): void {
  const message = error instanceof Error ? error.message : String(error);
  post({
    kind: 'error',
    requestId,
    message,
  } satisfies ErrorResponse);
}

function normalizeText(text: string): string {
  return text.normalize('NFKC').toLocaleLowerCase().replace(/\s+/g, ' ').trim();
}

function tokenize(text: string): string[] {
  return normalizeText(text).match(TOKEN_RE) ?? [];
}

function toEntryId(videoId: string, segIndex: number): string {
  return `${videoId}#${segIndex}`;
}

function ensureState(): LoadedState {
  if (!state) {
    throw new Error('TM database is not loaded yet.');
  }

  return state;
}

function scoreLexical(
  query: string,
  normalizedQuery: string,
  text: string,
  normalizedText: string,
): number {
  const queryTokens = tokenize(query);
  let tokenScore = 0;

  if (queryTokens.length > 0) {
    let hits = 0;
    for (const token of queryTokens) {
      if (normalizedText.includes(token)) {
        hits += 1;
      }
    }
    tokenScore = hits / queryTokens.length;
  }

  if (!normalizedQuery) {
    return tokenScore;
  }

  if (normalizedText.includes(normalizedQuery)) {
    const spanBoost = normalizedQuery.length / Math.max(text.length, normalizedQuery.length);
    return Math.max(tokenScore, 1 + spanBoost);
  }

  return tokenScore;
}

function rankHits(hits: RankedHit[], topK: number): RankedHit[] {
  hits.sort((left, right) => {
    if (right.score !== left.score) {
      return right.score - left.score;
    }

    return left.entryIndex - right.entryIndex;
  });

  return hits.slice(0, topK);
}

function summarize(entry: Entry): EntrySummary {
  return {
    entryId: entry.entryId,
    videoId: entry.videoId,
    segIndex: entry.segIndex,
    en: entry.en,
    zh: entry.zh,
    blockName: entry.blockName,
    layer: entry.layer,
    updatedAt: entry.updatedAt,
    hasVector: entry.hasVector,
  };
}

function toSearchResult(entry: Entry, score: number, language: SearchLanguage): SearchResult {
  return {
    ...summarize(entry),
    score,
    textLength: language === 'zh' ? entry.zhLength : entry.enLength,
  };
}

function toFloat32(blob: SqlBlob, dim: number): Float32Array {
  return new Float32Array(blob.buffer.slice(blob.byteOffset, blob.byteOffset + dim * 4));
}

async function loadDatabase(request: BootRequest): Promise<BootStats> {
  postStatus(request.requestId, 'boot', 'Downloading and opening the SQLite asset.');

  const start = performance.now();
  const response = await fetch(request.dbUrl, { cache: 'force-cache' });
  if (!response.ok) {
    throw new Error(`Failed to download ${request.dbUrl} (${response.status}).`);
  }

  const buffer = await response.arrayBuffer();
  const SQL = await initSqlJs({
    locateFile: () => sqlWasmUrl,
  });
  const db = new SQL.Database(new Uint8Array(buffer));

  const entries: Entry[] = [];
  const entryById = new Map<string, number>();
  const videoGroups = new Map<string, number[]>();

  const textStatement = db.prepare(`
    SELECT video_id, seg_index, en, zh, block_name, layer, updated_at
    FROM tm_main
    ORDER BY video_id, seg_index
  `);

  while (textStatement.step()) {
    const row = textStatement.getAsObject() as Record<string, string | number | null>;
    const videoId = String(row.video_id ?? '');
    const segIndex = Number(row.seg_index ?? 0);
    const en = String(row.en ?? '');
    const zh = String(row.zh ?? '');
    const blockName = String(row.block_name ?? '');
    const layer = Number(row.layer ?? 0);
    const updatedAt = String(row.updated_at ?? '');
    const id = toEntryId(videoId, segIndex);

    const entry: Entry = {
      entryId: id,
      videoId,
      segIndex,
      en,
      zh,
      blockName,
      layer,
      updatedAt,
      hasVector: false,
      enNorm: normalizeText(en),
      zhNorm: normalizeText(zh),
      enLength: en.length,
      zhLength: zh.length,
    };

    const nextIndex = entries.length;
    entries.push(entry);
    entryById.set(id, nextIndex);

    const group = videoGroups.get(videoId);
    if (group) {
      group.push(nextIndex);
    } else {
      videoGroups.set(videoId, [nextIndex]);
    }
  }

  textStatement.free();
  postStatus(request.requestId, 'boot', `Loaded ${entries.length.toLocaleString()} TM rows. Reading vector table.`);

  const semanticIndexes: number[] = [];
  const vectors: number[] = [];
  let vectorDim = 0;

  const vectorStatement = db.prepare(`
    SELECT m.video_id, m.seg_index, v.vector
    FROM tm_vectors AS v
    JOIN tm_main AS m USING (content_sha)
    WHERE v.model_id = $vectorModelId
    ORDER BY m.video_id, m.seg_index
  `);

  vectorStatement.bind({ $vectorModelId: request.vectorModelId });

  while (vectorStatement.step()) {
    const row = vectorStatement.getAsObject() as Record<string, string | number | SqlBlob | null>;
    const videoId = String(row.video_id ?? '');
    const segIndex = Number(row.seg_index ?? 0);
    const blob = row.vector;

    if (!(blob instanceof Uint8Array)) {
      continue;
    }

    const id = toEntryId(videoId, segIndex);
    const sourceIndex = entryById.get(id);
    if (sourceIndex === undefined) {
      continue;
    }

    const dim = blob.byteLength / 4;
    if (!Number.isInteger(dim) || dim <= 0) {
      continue;
    }

    if (vectorDim === 0) {
      vectorDim = dim;
    } else if (vectorDim !== dim) {
      throw new Error(`Vector dimension mismatch: expected ${vectorDim}, got ${dim}.`);
    }

    const sourceEntry = entries[sourceIndex];
    if (!sourceEntry) {
      continue;
    }

    sourceEntry.hasVector = true;
    semanticIndexes.push(sourceIndex);
    vectors.push(...toFloat32(blob, dim));
  }

  vectorStatement.free();
  db.close();

  state = {
    entries,
    entryById,
    videoGroups,
    semanticEntryIndexes: Int32Array.from(semanticIndexes),
    semanticVectors: Float32Array.from(vectors),
    vectorDim,
    vectorModelId: request.vectorModelId,
    queryModelId: request.queryModelId,
  };

  const loadMs = performance.now() - start;
  postStatus(request.requestId, 'boot', 'SQLite asset is ready.');

  return {
    dbUrl: request.dbUrl,
    dbSizeBytes: buffer.byteLength,
    totalEntries: entries.length,
    vectorEntries: semanticIndexes.length,
    vectorCoverage: entries.length === 0 ? 0 : semanticIndexes.length / entries.length,
    vectorDim,
    vectorModelId: request.vectorModelId,
    queryModelId: request.queryModelId,
    loadMs,
    semanticLanguageSupport: 'en-only',
  };
}

async function ensureExtractor(modelId: string, requestId: number): Promise<Extractor> {
  if (extractor) {
    return extractor;
  }

  postStatus(requestId, 'model', `Loading semantic model ${modelId}.`, 'loading');

  try {
    extractor = (await pipeline('feature-extraction', modelId)) as unknown as Extractor;
    postStatus(requestId, 'model', 'Semantic model is ready.', 'ready');
    return extractor;
  } catch (error) {
    postStatus(requestId, 'model', 'Semantic model failed to load.', 'error');
    throw error;
  }
}

async function embedQuery(modelId: string, requestId: number, query: string): Promise<Float32Array> {
  const featureExtractor = await ensureExtractor(modelId, requestId);
  const output = (await featureExtractor(query, {
    pooling: 'mean',
    normalize: true,
  })) as { data: Float32Array | number[] };
  const data = output.data;
  return data instanceof Float32Array ? data : Float32Array.from(data);
}

function searchLexical(request: SearchRequest, loaded: LoadedState): SearchResult[] {
  const query = request.query.trim();
  const normalizedQuery = normalizeText(query);
  const results: RankedHit[] = [];

  for (let index = 0; index < loaded.entries.length; index += 1) {
    const entry = loaded.entries[index];
    if (!entry) {
      continue;
    }

    const text = request.language === 'zh' ? entry.zh : entry.en;
    const normalizedText = request.language === 'zh' ? entry.zhNorm : entry.enNorm;
    const textLength = request.language === 'zh' ? entry.zhLength : entry.enLength;

    if (request.minLength > 0 && textLength < request.minLength) {
      continue;
    }

    const score = scoreLexical(query, normalizedQuery, text, normalizedText);
    if (score > 0 && score >= request.minScore) {
      results.push({ entryIndex: index, score });
    }
  }

  return rankHits(results, request.topK).map(({ entryIndex, score }) =>
    toSearchResult(loaded.entries[entryIndex]!, score, request.language),
  );
}

function searchSemantic(request: SearchRequest, loaded: LoadedState, queryVector: Float32Array): SearchResult[] {
  const results: RankedHit[] = [];

  for (let row = 0; row < loaded.semanticEntryIndexes.length; row += 1) {
    const entryIndex = loaded.semanticEntryIndexes[row];
    if (entryIndex === undefined) {
      continue;
    }

    const entry = loaded.entries[entryIndex];
    if (!entry) {
      continue;
    }

    if (request.minLength > 0 && entry.enLength < request.minLength) {
      continue;
    }

    const offset = row * loaded.vectorDim;
    let score = 0;

    for (let dim = 0; dim < loaded.vectorDim; dim += 1) {
      score += loaded.semanticVectors[offset + dim]! * queryVector[dim]!;
    }

    if (score >= request.minScore) {
      results.push({ entryIndex, score });
    }
  }

  return rankHits(results, request.topK).map(({ entryIndex, score }) =>
    toSearchResult(loaded.entries[entryIndex]!, score, 'en'),
  );
}

async function handleSearch(request: SearchRequest): Promise<{ results: SearchResult[]; note?: string }> {
  const loaded = ensureState();
  const query = request.query.trim();

  if (!query) {
    return { results: [] };
  }

  if (request.mode === 'lexical') {
    return { results: searchLexical(request, loaded) };
  }

  if (request.language !== 'en') {
    return {
      results: [],
      note: MODEL_LANGUAGE_NOTE,
    };
  }

  if (loaded.vectorDim === 0 || loaded.semanticEntryIndexes.length === 0) {
    return {
      results: [],
      note: 'No semantic vectors were found for the configured model.',
    };
  }

  const queryVector = await embedQuery(loaded.queryModelId, request.requestId, query);
  if (queryVector.length !== loaded.vectorDim) {
    throw new Error(
      `Query vector dimension mismatch: expected ${loaded.vectorDim}, got ${queryVector.length}.`,
    );
  }

  return { results: searchSemantic(request, loaded, queryVector) };
}

function handleContext(request: ContextRequest): ContextItem[] {
  const loaded = ensureState();
  const focusIndex = loaded.entryById.get(request.entryId);
  if (focusIndex === undefined) {
    throw new Error(`No TM entry found for ${request.entryId}.`);
  }

  const focusEntry = loaded.entries[focusIndex];
  if (!focusEntry) {
    throw new Error(`No TM entry found for ${request.entryId}.`);
  }

  const group = loaded.videoGroups.get(focusEntry.videoId) ?? [];
  const position = group.indexOf(focusIndex);
  const radius = request.radius ?? 2;
  const start = Math.max(0, position - radius);
  const end = Math.min(group.length, position + radius + 1);

  const context: ContextItem[] = [];
  for (let index = start; index < end; index += 1) {
    const entryIndex = group[index];
    if (entryIndex === undefined) {
      continue;
    }

    const entry = loaded.entries[entryIndex];
    if (!entry) {
      continue;
    }

    context.push({
      ...summarize(entry),
      isFocus: entry.entryId === request.entryId,
    });
  }

  return context;
}

function handleTranscript(request: { videoId: string; focusEntryId?: string }): ContextItem[] {
  const loaded = ensureState();
  const group = loaded.videoGroups.get(request.videoId);

  if (!group || group.length === 0) {
    throw new Error(`No TM entries found for video ${request.videoId}.`);
  }

  const items: ContextItem[] = [];
  for (const entryIndex of group) {
    const entry = loaded.entries[entryIndex];
    if (!entry) {
      continue;
    }

    items.push({
      ...summarize(entry),
      isFocus: entry.entryId === request.focusEntryId,
    });
  }

  return items;
}

workerScope.addEventListener('message', async (event: MessageEvent<WorkerRequest>) => {
  const request = event.data;

  try {
    switch (request.kind) {
      case 'boot': {
        const stats = await loadDatabase(request);
        post({
          kind: 'boot:ok',
          requestId: request.requestId,
          stats,
        });
        return;
      }

      case 'search': {
        const { results, note } = await handleSearch(request);
        post({
          kind: 'search:ok',
          requestId: request.requestId,
          results,
          note,
        });
        return;
      }

      case 'context': {
        const context = handleContext(request);
        post({
          kind: 'context:ok',
          requestId: request.requestId,
          context,
        });
        return;
      }

      case 'transcript': {
        const items = handleTranscript(request);
        post({
          kind: 'transcript:ok',
          requestId: request.requestId,
          videoId: request.videoId,
          items,
        });
        return;
      }

      default: {
        const exhaustive: never = request;
        throw new Error(`Unsupported worker request: ${String(exhaustive)}`);
      }
    }
  } catch (error) {
    postError(request.requestId, error);
  }
});
