import initSqlJs from 'sql.js';
import sqlWasmUrl from 'sql.js/dist/sql-wasm.wasm?url';
import { env, pipeline } from '@huggingface/transformers';
import type {
  BootProgressResponse,
  BootProgressSnapshot,
  BootRequest,
  BootStats,
  CueTimeDistribution,
  CueTimeDistributionRepresentativeLine,
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
const PAIRS_LABEL = 'English/中文 Pairs';
const ENTRY_PROGRESS_BATCH_SIZE = 2_000;
const VECTOR_PROGRESS_BATCH_SIZE = 2_000;

type SqlBlob = Uint8Array;
type ExtractorOutput = { data: Float32Array | number[] };
type Extractor = (input: string, options?: Record<string, unknown>) => Promise<ExtractorOutput>;
type ProgressReporter = (progress: Omit<BootProgressSnapshot, 'target'>) => void;
type ModelProgressInfo =
  | {
      status: 'initiate' | 'download' | 'done';
      name: string;
      file: string;
    }
  | {
      status: 'progress';
      name: string;
      file: string;
      progress: number;
      loaded: number;
      total: number;
    }
  | {
      status: 'ready';
      task: string;
      model: string;
    };

interface Entry extends EntrySummary {
  enLength: number;
  zhLength: number;
}

interface RankedHit {
  entryIndex: number;
  score: number;
}

interface SemanticBinCandidate {
  entryIndex: number;
  overlapShare: number;
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
let extractorPromise: Promise<Extractor> | null = null;

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

function postBootProgress(requestId: number, progress: BootProgressSnapshot): void {
  post({
    kind: 'boot:progress',
    requestId,
    progress: {
      ...progress,
      progress: clamp01(progress.progress),
    },
  } satisfies BootProgressResponse);
}

function createProgressReporter(
  requestId: number,
  target: BootProgressSnapshot['target'],
  initialName: string,
): ProgressReporter {
  let lastProgress = -1;
  let lastStatusText = '';
  let lastDetail = '';
  let lastName = initialName;

  return (progress) => {
    const nextProgress = clamp01(progress.progress);
    const nextName = progress.name || lastName;
    const nextStatusText = progress.statusText;
    const nextDetail = progress.detail ?? '';
    const shouldSkip =
      nextProgress < 1 &&
      Math.abs(nextProgress - lastProgress) < 0.01 &&
      nextStatusText === lastStatusText &&
      nextDetail === lastDetail &&
      nextName === lastName;

    if (shouldSkip) {
      return;
    }

    lastProgress = nextProgress;
    lastStatusText = nextStatusText;
    lastDetail = nextDetail;
    lastName = nextName;

    postBootProgress(requestId, {
      ...progress,
      target,
      name: nextName,
      progress: nextProgress,
    });
  };
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

function scaleProgress(start: number, end: number, progress: number): number {
  return start + (end - start) * clamp01(progress);
}

function getDisplayModelName(modelId: string): string {
  return modelId.split('/').at(-1) ?? modelId;
}

function formatByteCount(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes <= 0) {
    return '0 B';
  }

  const units = ['B', 'KB', 'MB', 'GB'];
  let value = bytes;
  let unitIndex = 0;

  while (value >= 1024 && unitIndex < units.length - 1) {
    value /= 1024;
    unitIndex += 1;
  }

  const digits = value >= 100 || unitIndex === 0 ? 0 : value >= 10 ? 1 : 2;
  return `${value.toFixed(digits)} ${units[unitIndex]}`;
}

function formatCount(value: number): string {
  return value.toLocaleString('en-US');
}

async function downloadAsset(
  url: string,
  onProgress?: (loaded: number, total: number | null) => void,
): Promise<Uint8Array> {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to download ${url} (${response.status}).`);
  }

  const contentLengthHeader = response.headers.get('Content-Length');
  const total =
    contentLengthHeader !== null && contentLengthHeader !== ''
      ? Number(contentLengthHeader)
      : null;

  if (!response.body) {
    const buffer = new Uint8Array(await response.arrayBuffer());
    onProgress?.(buffer.byteLength, total ?? buffer.byteLength);
    return buffer;
  }

  const reader = response.body.getReader();
  const chunks: Uint8Array[] = [];
  let loaded = 0;

  while (true) {
    const { done, value } = await reader.read();
    if (done) {
      break;
    }

    if (!value) {
      continue;
    }

    chunks.push(value);
    loaded += value.byteLength;
    onProgress?.(loaded, total);
  }

  const buffer = new Uint8Array(loaded);
  let offset = 0;
  for (const chunk of chunks) {
    buffer.set(chunk, offset);
    offset += chunk.byteLength;
  }

  onProgress?.(loaded, total ?? loaded);
  return buffer;
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
  semanticVectors: Float32Array,
  vectorDim: number,
  entryVectorRows: Int32Array,
): CueTimeDistribution | null {
  const binCount = TIME_DISTRIBUTION_BIN_COUNT;
  const coverageTotals = new Float64Array(binCount);
  const cueDurations: number[] = [];
  const videoSpans: number[] = [];
  const binWidth = 1 / binCount;
  const semanticSums = vectorDim > 0 ? new Float64Array(binCount * vectorDim) : new Float64Array(0);
  const semanticCandidates = Array.from({ length: binCount }, () => [] as SemanticBinCandidate[]);
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

      const cueSpan = endRatio - startRatio;
      timedEntryCount += 1;
      cueDurations.push(endMs - startMs);

      const startBinIndex = toStartBinIndex(startRatio, binCount);
      const endBinIndex = toEndBinIndex(endRatio, binCount);
      const vectorRow = entryVectorRows[entryIndex] ?? -1;
      const cueMidpoint = startRatio + cueSpan / 2;
      let dominantBinIndex = -1;
      let dominantOverlap = 0;
      let dominantCenterDistance = Number.POSITIVE_INFINITY;

      for (let binIndex = startBinIndex; binIndex <= endBinIndex; binIndex += 1) {
        const binStart = binIndex * binWidth;
        const binEnd = binStart + binWidth;
        const overlap = Math.min(endRatio, binEnd) - Math.max(startRatio, binStart);
        if (overlap > 0) {
          videoCoverage[binIndex] = (videoCoverage[binIndex] ?? 0) + overlap / binWidth;
          if (vectorDim > 0 && vectorRow >= 0) {
            const binCenter = binStart + binWidth / 2;
            const centerDistance = Math.abs(binCenter - cueMidpoint);
            const overlapDelta = overlap - dominantOverlap;
            const distanceDelta = dominantCenterDistance - centerDistance;

            if (
              dominantBinIndex < 0 ||
              overlapDelta > 1e-9 ||
              (Math.abs(overlapDelta) <= 1e-9 && distanceDelta > 1e-9) ||
              (Math.abs(overlapDelta) <= 1e-9 &&
                Math.abs(distanceDelta) <= 1e-9 &&
                binIndex < dominantBinIndex)
            ) {
              dominantBinIndex = binIndex;
              dominantOverlap = overlap;
              dominantCenterDistance = centerDistance;
            }
          }
        }
      }

      if (vectorDim > 0 && vectorRow >= 0 && dominantBinIndex >= 0) {
        const semanticOffset = dominantBinIndex * vectorDim;
        const vectorOffset = vectorRow * vectorDim;
        semanticCandidates[dominantBinIndex]!.push({
          entryIndex,
          overlapShare: dominantOverlap / cueSpan,
        });

        for (let dim = 0; dim < vectorDim; dim += 1) {
          semanticSums[semanticOffset + dim] =
            (semanticSums[semanticOffset + dim] ?? 0) + semanticVectors[vectorOffset + dim]!;
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
  const binRepresentativeLines = semanticCandidates.map(
    (candidates, binIndex): CueTimeDistributionRepresentativeLine | null => {
      if (vectorDim === 0 || candidates.length === 0) {
        return null;
      }

      let bestCandidate = candidates[0] ?? null;
      let bestScore = Number.NEGATIVE_INFINITY;
      const semanticOffset = binIndex * vectorDim;

      for (const candidate of candidates) {
        const vectorRow = entryVectorRows[candidate.entryIndex] ?? -1;
        if (vectorRow < 0) {
          continue;
        }

        const vectorOffset = vectorRow * vectorDim;
        let score = 0;
        for (let dim = 0; dim < vectorDim; dim += 1) {
          score += semanticVectors[vectorOffset + dim]! * semanticSums[semanticOffset + dim]!;
        }

        if (
          !bestCandidate ||
          score > bestScore ||
          (score === bestScore && candidate.overlapShare > bestCandidate.overlapShare) ||
          (score === bestScore &&
            candidate.overlapShare === bestCandidate.overlapShare &&
            candidate.entryIndex < bestCandidate.entryIndex)
        ) {
          bestCandidate = candidate;
          bestScore = score;
        }
      }

      if (!bestCandidate) {
        return null;
      }

      const sampleEntry = entries[bestCandidate.entryIndex];
      if (!sampleEntry) {
        return null;
      }

      return {
        en: sampleEntry.en,
        zh: sampleEntry.zh,
        candidateCount: candidates.length,
      };
    },
  );
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
    binRepresentativeLines,
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

async function loadDatabase(request: BootRequest, reportProgress: ProgressReporter): Promise<BootStats> {
  reportProgress({
    name: PAIRS_LABEL,
    progress: 0,
    statusText: 'Downloading TM snapshot',
    detail: 'Starting download',
  });

  const buffer = await downloadAsset(request.dbUrl, (loaded, total) => {
    const progress = total && total > 0 ? scaleProgress(0, 0.5, loaded / total) : 0.12;
    const detail =
      total && total > 0
        ? `${formatByteCount(loaded)} / ${formatByteCount(total)}`
        : `${formatByteCount(loaded)} downloaded`;

    reportProgress({
      name: PAIRS_LABEL,
      progress,
      statusText: 'Downloading TM snapshot',
      detail,
    });
  });

  reportProgress({
    name: PAIRS_LABEL,
    progress: 0.54,
    statusText: 'Initializing SQLite runtime',
    detail: 'Opening the browser-side snapshot',
  });

  const SQL = await initSqlJs({
    locateFile: () => sqlWasmUrl,
  });

  reportProgress({
    name: PAIRS_LABEL,
    progress: 0.58,
    statusText: 'Reading English/中文 pairs',
    detail: 'Scanning tm_main',
  });

  const db = new SQL.Database(buffer);

  try {
    const readCount = (statementSql: string, bindValues?: Record<string, string>): number => {
      const statement = db.prepare(statementSql);

      try {
        if (bindValues) {
          statement.bind(bindValues);
        }

        if (!statement.step()) {
          return 0;
        }

        const row = statement.getAsObject() as Record<string, string | number | null>;
        return Number(row.count ?? 0);
      } finally {
        statement.free();
      }
    };

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

    const totalEntryCount = readCount(`SELECT COUNT(*) AS count FROM tm_main`);
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

      if (
        entries.length === 1 ||
        entries.length % ENTRY_PROGRESS_BATCH_SIZE === 0 ||
        entries.length === totalEntryCount
      ) {
        const entryProgress = totalEntryCount > 0 ? entries.length / totalEntryCount : 1;
        reportProgress({
          name: PAIRS_LABEL,
          progress: scaleProgress(0.58, 0.8, entryProgress),
          statusText: 'Reading English/中文 pairs',
          detail: `${formatCount(entries.length)} / ${formatCount(totalEntryCount)}`,
        });
      }
    }

    textStatement.free();

    reportProgress({
      name: PAIRS_LABEL,
      progress: 0.82,
      statusText: 'Loading semantic vectors',
      detail: `Preparing ${getDisplayModelName(VECTOR_MODEL_ID)} vectors`,
    });

    const semanticIndexes: number[] = [];
    const vectorRows: Float32Array[] = [];
    let vectorDim = 0;
    const entryVectorRows = new Int32Array(entries.length);
    entryVectorRows.fill(-1);
    const totalVectorCount = readCount(
      `
        SELECT COUNT(*) AS count
        FROM tm_vectors AS v
        JOIN tm_main AS m USING (content_sha)
        WHERE v.model_id = $vectorModelId
      `,
      { $vectorModelId: VECTOR_MODEL_ID },
    );

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

      entryVectorRows[sourceIndex] = vectorRows.length;
      semanticIndexes.push(sourceIndex);
      vectorRows.push(toFloat32(blob, dim));

      if (
        vectorRows.length === 1 ||
        vectorRows.length % VECTOR_PROGRESS_BATCH_SIZE === 0 ||
        vectorRows.length === totalVectorCount
      ) {
        const vectorProgress = totalVectorCount > 0 ? vectorRows.length / totalVectorCount : 1;
        reportProgress({
          name: PAIRS_LABEL,
          progress: scaleProgress(0.82, 0.96, vectorProgress),
          statusText: 'Loading semantic vectors',
          detail: `${formatCount(vectorRows.length)} / ${formatCount(totalVectorCount)}`,
        });
      }
    }

    vectorStatement.free();

    reportProgress({
      name: PAIRS_LABEL,
      progress: 0.98,
      statusText: 'Finalizing cue indexes',
      detail: `${formatCount(entries.length)} pairs cached in memory`,
    });

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

    const cueTimeDistribution = buildCueTimeDistribution(
      entries,
      videoGroups,
      semanticVectors,
      vectorDim,
      entryVectorRows,
    );

    reportProgress({
      name: PAIRS_LABEL,
      progress: 1,
      statusText: 'Ready',
      detail: `${formatCount(entries.length)} pairs loaded`,
    });

    return {
      totalEntries: entries.length,
      cueTimeDistribution,
      embeddingModelId: QUERY_MODEL_ID,
    };
  } finally {
    db.close();
  }
}

function createModelProgressCallback(reportProgress: ProgressReporter): (info: ModelProgressInfo) => void {
  const files = new Map<string, { loaded: number; total: number | null; done: boolean }>();

  const emit = (statusText: string, detail: string, modelId = QUERY_MODEL_ID): void => {
    let totalBytes = 0;
    let loadedBytes = 0;
    let completedFiles = 0;

    for (const file of files.values()) {
      if (file.total !== null) {
        totalBytes += file.total;
        loadedBytes += Math.min(file.loaded, file.total);
      } else if (file.done) {
        totalBytes += 1;
        loadedBytes += 1;
      }

      if (file.done) {
        completedFiles += 1;
      }
    }

    const progress =
      totalBytes > 0
        ? loadedBytes / totalBytes
        : files.size > 0
          ? completedFiles / files.size
          : 0;

    reportProgress({
      name: getDisplayModelName(modelId),
      progress,
      statusText,
      detail,
    });
  };

  return (info) => {
    if (info.status === 'ready') {
      reportProgress({
        name: getDisplayModelName(info.model),
        progress: 1,
        statusText: 'Ready',
        detail: info.model,
      });
      return;
    }

    const key = `${info.name}:${info.file}`;
    const current = files.get(key) ?? { loaded: 0, total: null, done: false };
    const fileName = info.file.split('/').at(-1) ?? info.file;

    switch (info.status) {
      case 'initiate': {
        files.set(key, current);
        emit('Checking model files', `${files.size} file${files.size === 1 ? '' : 's'} queued`, info.name);
        return;
      }

      case 'download': {
        files.set(key, current);
        emit('Downloading model files', `Fetching ${fileName}`, info.name);
        return;
      }

      case 'progress': {
        current.loaded = info.loaded;
        current.total = info.total;
        current.done = false;
        files.set(key, current);
        emit(
          'Downloading model files',
          `${fileName} - ${formatByteCount(info.loaded)} / ${formatByteCount(info.total)}`,
          info.name,
        );
        return;
      }

      case 'done': {
        current.done = true;
        current.loaded = current.total ?? 1;
        current.total = current.total ?? 1;
        files.set(key, current);
        const completedFiles = Array.from(files.values()).filter((file) => file.done).length;
        emit(
          completedFiles === files.size ? 'Finalizing embedding model' : 'Downloading model files',
          `${completedFiles} / ${files.size} files ready`,
          info.name,
        );
        return;
      }
    }
  };
}

async function ensureExtractor(reportProgress?: ProgressReporter): Promise<Extractor> {
  if (extractor) {
    reportProgress?.({
      name: getDisplayModelName(QUERY_MODEL_ID),
      progress: 1,
      statusText: 'Ready',
      detail: QUERY_MODEL_ID,
    });
    return extractor;
  }

  if (!extractorPromise) {
    reportProgress?.({
      name: getDisplayModelName(QUERY_MODEL_ID),
      progress: 0,
      statusText: 'Preparing embedding model',
      detail: QUERY_MODEL_ID,
    });

    const progressCallback = reportProgress ? createModelProgressCallback(reportProgress) : undefined;

    extractorPromise = pipeline('feature-extraction', QUERY_MODEL_ID, {
      progress_callback: progressCallback,
    })
      .then((loadedExtractor) => {
        extractor = loadedExtractor as unknown as Extractor;
        return extractor;
      })
      .finally(() => {
        extractorPromise = null;
      });
  }

  const loadedExtractor = await extractorPromise;
  reportProgress?.({
    name: getDisplayModelName(QUERY_MODEL_ID),
    progress: 1,
    statusText: 'Ready',
    detail: QUERY_MODEL_ID,
  });
  return loadedExtractor;
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
        const pairsProgress = createProgressReporter(request.requestId, 'pairs', PAIRS_LABEL);
        const modelProgress = createProgressReporter(
          request.requestId,
          'model',
          getDisplayModelName(QUERY_MODEL_ID),
        );

        const [stats] = await Promise.all([
          loadDatabase(request, pairsProgress),
          ensureExtractor(modelProgress),
        ]);
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
