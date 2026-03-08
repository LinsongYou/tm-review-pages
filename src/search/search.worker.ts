import initSqlJs from 'sql.js';
import sqlWasmUrl from 'sql.js/dist/sql-wasm.wasm?url';
import { env, pipeline } from '@huggingface/transformers';
import type {
  BootRequest,
  BootStats,
  CueTimeDistribution,
  ContextItem,
  ContextRequest,
  EntrySummary,
  ErrorResponse,
  SearchRequest,
  SearchResult,
  TranscriptRequest,
  WorkerRequest,
  WorkerResponse,
} from './protocol';

const workerScope = self as DedicatedWorkerGlobalScope;
const DEFAULT_CONTEXT_RADIUS = 3;
const TIME_DISTRIBUTION_BIN_COUNT = 120;
const VECTOR_MODEL_ID = 'sentence-transformers/all-MiniLM-L6-v2';
const QUERY_MODEL_ID = 'Xenova/all-MiniLM-L6-v2';

type SqlBlob = Uint8Array;
type ExtractorOutput = { data: Float32Array | number[] };
type Extractor = (input: string, options?: Record<string, unknown>) => Promise<ExtractorOutput>;

interface Entry extends EntrySummary {
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
}

let state: LoadedState | null = null;
let extractor: Extractor | null = null;

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

function postError(requestId: number, error: unknown): void {
  const message = error instanceof Error ? error.message : String(error);
  post({
    kind: 'error',
    requestId,
    message,
  } satisfies ErrorResponse);
}

function toNullableNumber(value: string | number | null | undefined): number | null {
  if (value === null || value === undefined || value === '') {
    return null;
  }

  const parsed = typeof value === 'number' ? value : Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function toEntryId(videoId: string, segIndex: number): string {
  return `${videoId}#${segIndex}`;
}

function clamp01(value: number): number {
  return Math.min(1, Math.max(0, value));
}

function toStartBinIndex(value: number, binCount: number): number {
  return Math.max(0, Math.min(binCount - 1, Math.floor(value * binCount)));
}

function toEndBinIndex(value: number, binCount: number): number {
  return Math.max(0, Math.min(binCount - 1, Math.ceil(value * binCount) - 1));
}

function ensureState(): LoadedState {
  if (!state) {
    throw new Error('TM database is not loaded yet.');
  }

  return state;
}

function compareRankedHits(left: RankedHit, right: RankedHit): number {
  if (right.score !== left.score) {
    return right.score - left.score;
  }

  return left.entryIndex - right.entryIndex;
}

function pushRankedHit(hits: RankedHit[], candidate: RankedHit, topK: number): void {
  if (topK <= 0) {
    return;
  }

  let insertAt = hits.length;
  while (insertAt > 0 && compareRankedHits(candidate, hits[insertAt - 1]!) < 0) {
    insertAt -= 1;
  }

  if (hits.length === topK && insertAt === hits.length) {
    return;
  }

  hits.splice(insertAt, 0, candidate);
  if (hits.length > topK) {
    hits.pop();
  }
}

function summarize(entry: Entry): EntrySummary {
  return {
    entryId: entry.entryId,
    videoId: entry.videoId,
    segIndex: entry.segIndex,
    en: entry.en,
    zh: entry.zh,
    blockName: entry.blockName,
    startMs: entry.startMs,
    endMs: entry.endMs,
  };
}

function toSearchResult(entry: Entry, score: number): SearchResult {
  return {
    ...summarize(entry),
    score,
    textLength: entry.enLength,
  };
}

function toFloat32(blob: SqlBlob, dim: number): Float32Array {
  return new Float32Array(blob.buffer.slice(blob.byteOffset, blob.byteOffset + dim * 4));
}

function average(values: number[]): number {
  if (values.length === 0) {
    return 0;
  }

  let total = 0;
  for (const value of values) {
    total += value;
  }

  return total / values.length;
}

function median(values: number[]): number {
  if (values.length === 0) {
    return 0;
  }

  const sorted = [...values].sort((left, right) => left - right);
  const middle = Math.floor(sorted.length / 2);

  if (sorted.length % 2 === 0) {
    return (sorted[middle - 1]! + sorted[middle]!) / 2;
  }

  return sorted[middle]!;
}

function buildCueTimeDistribution(
  entries: Entry[],
  videoGroups: Map<string, number[]>,
): CueTimeDistribution | null {
  const binCount = TIME_DISTRIBUTION_BIN_COUNT;
  const coverageTotals = new Float64Array(binCount);
  const cueDurations: number[] = [];
  const videoSpans: number[] = [];
  const binWidth = 1 / binCount;
  let timedEntryCount = 0;
  let timedVideoCount = 0;

  for (const group of videoGroups.values()) {
    let videoStartMs = Number.POSITIVE_INFINITY;
    let videoEndMs = Number.NEGATIVE_INFINITY;

    for (const entryIndex of group) {
      const entry = entries[entryIndex];
      if (!entry) {
        continue;
      }

      if (entry.startMs !== null) {
        videoStartMs = Math.min(videoStartMs, entry.startMs);
      }

      if (entry.endMs !== null) {
        videoEndMs = Math.max(videoEndMs, entry.endMs);
      }
    }

    if (!Number.isFinite(videoStartMs) || !Number.isFinite(videoEndMs) || videoEndMs <= videoStartMs) {
      continue;
    }

    const videoSpanMs = videoEndMs - videoStartMs;
    const videoCoverage = new Float64Array(binCount);
    timedVideoCount += 1;
    videoSpans.push(videoSpanMs);

    for (const entryIndex of group) {
      const entry = entries[entryIndex];
      if (!entry || entry.startMs === null || entry.endMs === null) {
        continue;
      }

      const startMs = Math.max(videoStartMs, entry.startMs);
      const endMs = Math.min(videoEndMs, entry.endMs);
      if (!(endMs > startMs)) {
        continue;
      }

      const startRatio = clamp01((startMs - videoStartMs) / videoSpanMs);
      const endRatio = clamp01((endMs - videoStartMs) / videoSpanMs);
      if (!(endRatio > startRatio)) {
        continue;
      }

      timedEntryCount += 1;
      cueDurations.push(endMs - startMs);

      const startBinIndex = toStartBinIndex(startRatio, binCount);
      const endBinIndex = toEndBinIndex(endRatio, binCount);

      for (let binIndex = startBinIndex; binIndex <= endBinIndex; binIndex += 1) {
        const binStart = binIndex * binWidth;
        const binEnd = binStart + binWidth;
        const overlap = Math.min(endRatio, binEnd) - Math.max(startRatio, binStart);
        if (overlap > 0) {
          videoCoverage[binIndex] = (videoCoverage[binIndex] ?? 0) + overlap / binWidth;
        }
      }
    }

    for (let binIndex = 0; binIndex < binCount; binIndex += 1) {
      coverageTotals[binIndex] = (coverageTotals[binIndex] ?? 0) + Math.min(1, videoCoverage[binIndex] ?? 0);
    }
  }

  if (timedVideoCount === 0 || timedEntryCount === 0) {
    return null;
  }

  const bins = Array.from(coverageTotals, (value) => value / timedVideoCount);
  let peakCoverage = 0;
  let peakBinIndex = 0;
  let coverageTotal = 0;

  for (let index = 0; index < bins.length; index += 1) {
    const value = bins[index] ?? 0;
    coverageTotal += value;
    if (value > peakCoverage) {
      peakCoverage = value;
      peakBinIndex = index;
    }
  }

  return {
    binCount,
    bins,
    timedEntryCount,
    totalEntryCount: entries.length,
    timedVideoCount,
    totalVideoCount: videoGroups.size,
    averageCueDurationMs: average(cueDurations),
    medianVideoSpanMs: median(videoSpans),
    averageCoverage: coverageTotal / binCount,
    peakCoverage,
    peakRangeStart: peakBinIndex / binCount,
    peakRangeEnd: (peakBinIndex + 1) / binCount,
  };
}

async function loadDatabase(request: BootRequest): Promise<BootStats> {
  const response = await fetch(request.dbUrl);
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
  const tmMainColumns = new Set<string>();

  const tableInfoStatement = db.prepare(`PRAGMA table_info(tm_main)`);
  while (tableInfoStatement.step()) {
    const row = tableInfoStatement.getAsObject() as Record<string, string | number | null>;
    tmMainColumns.add(String(row.name ?? ''));
  }
  tableInfoStatement.free();

  const hasCueTiming = tmMainColumns.has('start_ms') && tmMainColumns.has('end_ms');

  const textStatement = db.prepare(`
    SELECT video_id, seg_index, en, zh, block_name${hasCueTiming ? ', start_ms, end_ms' : ''}
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
    const startMs = hasCueTiming ? toNullableNumber(row.start_ms) : null;
    const endMs = hasCueTiming ? toNullableNumber(row.end_ms) : null;
    const id = toEntryId(videoId, segIndex);

    const entry: Entry = {
      entryId: id,
      videoId,
      segIndex,
      en,
      zh,
      blockName,
      startMs,
      endMs,
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

  const semanticIndexes: number[] = [];
  const vectorRows: Float32Array[] = [];
  let vectorDim = 0;

  const vectorStatement = db.prepare(`
    SELECT m.video_id, m.seg_index, v.vector
    FROM tm_vectors AS v
    JOIN tm_main AS m USING (content_sha)
    WHERE v.model_id = $vectorModelId
    ORDER BY m.video_id, m.seg_index
  `);

  vectorStatement.bind({ $vectorModelId: VECTOR_MODEL_ID });

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

    semanticIndexes.push(sourceIndex);
    vectorRows.push(toFloat32(blob, dim));
  }

  vectorStatement.free();
  db.close();

  const semanticVectors = new Float32Array(vectorRows.length * vectorDim);
  for (let rowIndex = 0; rowIndex < vectorRows.length; rowIndex += 1) {
    semanticVectors.set(vectorRows[rowIndex]!, rowIndex * vectorDim);
  }

  state = {
    entries,
    entryById,
    videoGroups,
    semanticEntryIndexes: Int32Array.from(semanticIndexes),
    semanticVectors,
    vectorDim,
  };

  const cueTimeDistribution = buildCueTimeDistribution(entries, videoGroups);

  return {
    totalEntries: entries.length,
    cueTimeDistribution,
  };
}

async function ensureExtractor(): Promise<Extractor> {
  if (extractor) {
    return extractor;
  }

  extractor = (await pipeline('feature-extraction', QUERY_MODEL_ID)) as unknown as Extractor;
  return extractor;
}

async function embedQuery(query: string): Promise<Float32Array> {
  const featureExtractor = await ensureExtractor();
  const output = (await featureExtractor(query, {
    pooling: 'mean',
    normalize: true,
  })) as { data: Float32Array | number[] };
  const data = output.data;
  return data instanceof Float32Array ? data : Float32Array.from(data);
}

function searchSemantic(
  request: SearchRequest,
  loaded: LoadedState,
  queryVector: Float32Array,
): SearchResult[] {
  const topHits: RankedHit[] = [];

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
      pushRankedHit(topHits, { entryIndex, score }, request.topK);
    }
  }

  return topHits.map(({ entryIndex, score }) => toSearchResult(loaded.entries[entryIndex]!, score));
}

async function handleSearch(request: SearchRequest): Promise<{ results: SearchResult[]; note?: string }> {
  const loaded = ensureState();
  const query = request.query.trim();

  if (!query) {
    return { results: [] };
  }

  if (loaded.vectorDim === 0 || loaded.semanticEntryIndexes.length === 0) {
    return {
      results: [],
      note: 'No semantic vectors were found for the configured model.',
    };
  }

  const queryVector = await embedQuery(query);
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
  const radius = request.radius ?? DEFAULT_CONTEXT_RADIUS;
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

function handleTranscript(request: TranscriptRequest): ContextItem[] {
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
