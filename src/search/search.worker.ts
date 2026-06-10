import initSqlJs from 'sql.js';
import sqlWasmUrl from 'sql.js/dist/sql-wasm.wasm?url';
import { env, pipeline } from '@huggingface/transformers';
import { getDisplayModelName } from '../format';
import type {
  BootProgressSnapshot,
  BootRequest,
  BootStats,
  EntrySummary,
  SearchRequest,
  SearchResult,
  TranscriptRequest,
  WorkerRequest,
  WorkerResponse,
} from './protocol';

const workerScope = self as DedicatedWorkerGlobalScope;
const VECTOR_MODEL_ID = 'sentence-transformers/all-MiniLM-L6-v2';
const QUERY_MODEL_ID = 'Xenova/all-MiniLM-L6-v2';
const ORT_WASM_BASE_URL =
  'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0-dev.20250409-89f8206ba4/dist/';
const SEARCH_TOP_K = 24;
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
      loaded: number;
      total: number;
    }
  | {
      status: 'ready';
      model: string;
    };

interface RankedHit {
  entryIndex: number;
  score: number;
}

interface LoadedState {
  entries: EntrySummary[];
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

const wasmBackend = env.backends.onnx.wasm!;
wasmBackend.wasmPaths = ORT_WASM_BASE_URL;
wasmBackend.numThreads = 1;
wasmBackend.proxy = false;

function createProgressReporter(
  requestId: number,
  target: BootProgressSnapshot['target'],
): ProgressReporter {
  let lastProgress = -1;
  let lastStatusText = '';
  let lastDetail = '';

  return (progress) => {
    const nextProgress = clamp01(progress.progress);
    const nextStatusText = progress.statusText;
    const nextDetail = progress.detail ?? '';
    const shouldSkip =
      nextProgress < 1 &&
      Math.abs(nextProgress - lastProgress) < 0.01 &&
      nextStatusText === lastStatusText &&
      nextDetail === lastDetail;

    if (shouldSkip) {
      return;
    }

    lastProgress = nextProgress;
    lastStatusText = nextStatusText;
    lastDetail = nextDetail;

    workerScope.postMessage({
      kind: 'boot:progress',
      requestId,
      progress: { ...progress, target, progress: nextProgress },
    } satisfies WorkerResponse);
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
  onProgress: (loaded: number, total: number | null) => void,
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

  const reader = response.body!.getReader();
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
    onProgress(loaded, total);
  }

  const buffer = new Uint8Array(loaded);
  let offset = 0;
  for (const chunk of chunks) {
    buffer.set(chunk, offset);
    offset += chunk.byteLength;
  }

  onProgress(loaded, total ?? loaded);
  return buffer;
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

function pushRankedHit(hits: RankedHit[], candidate: RankedHit): void {
  let insertAt = hits.length;
  while (insertAt > 0 && compareRankedHits(candidate, hits[insertAt - 1]!) < 0) {
    insertAt -= 1;
  }

  if (hits.length === SEARCH_TOP_K && insertAt === hits.length) {
    return;
  }

  hits.splice(insertAt, 0, candidate);
  if (hits.length > SEARCH_TOP_K) {
    hits.pop();
  }
}

async function loadDatabase(request: BootRequest, reportProgress: ProgressReporter): Promise<BootStats> {
  reportProgress({
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
      progress,
      statusText: 'Downloading TM snapshot',
      detail,
    });
  });

  reportProgress({
    progress: 0.54,
    statusText: 'Initializing SQLite runtime',
    detail: 'Opening the browser-side snapshot',
  });

  const SQL = await initSqlJs({
    locateFile: () => sqlWasmUrl,
  });

  reportProgress({
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

    const entries: EntrySummary[] = [];
    const entryById = new Map<string, number>();
    const videoGroups = new Map<string, number[]>();

    const totalEntryCount = readCount(`SELECT COUNT(*) AS count FROM tm_main`);

    const textStatement = db.prepare(`
      SELECT video_id, seg_index, en, zh, start_ms, end_ms
      FROM tm_main
      ORDER BY video_id, seg_index
    `);

    while (textStatement.step()) {
      const row = textStatement.getAsObject() as Record<string, string | number | null>;
      const videoId = String(row.video_id ?? '');
      const segIndex = Number(row.seg_index ?? 0);
      const en = String(row.en ?? '');
      const zh = String(row.zh ?? '');
      const startMs = toNullableNumber(row.start_ms);
      const endMs = toNullableNumber(row.end_ms);
      const id = toEntryId(videoId, segIndex);

      const entry: EntrySummary = {
        entryId: id,
        videoId,
        segIndex,
        en,
        zh,
        startMs,
        endMs,
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
          progress: scaleProgress(0.58, 0.8, entryProgress),
          statusText: 'Reading English/中文 pairs',
          detail: `${formatCount(entries.length)} / ${formatCount(totalEntryCount)}`,
        });
      }
    }

    textStatement.free();

    reportProgress({
      progress: 0.82,
      statusText: 'Loading semantic vectors',
      detail: `Preparing ${getDisplayModelName(VECTOR_MODEL_ID)} vectors`,
    });

    const semanticIndexes: number[] = [];
    const vectorRows: Float32Array[] = [];
    let vectorDim = 0;
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
      const blob = row.vector as SqlBlob;

      const id = toEntryId(videoId, segIndex);
      const sourceIndex = entryById.get(id)!;

      const dim = blob.byteLength / 4;
      if (vectorDim === 0) {
        vectorDim = dim;
      } else if (vectorDim !== dim) {
        throw new Error(`Vector dimension mismatch: expected ${vectorDim}, got ${dim}.`);
      }

      semanticIndexes.push(sourceIndex);
      vectorRows.push(new Float32Array(blob.buffer.slice(blob.byteOffset, blob.byteOffset + blob.byteLength)));

      if (
        vectorRows.length === 1 ||
        vectorRows.length % VECTOR_PROGRESS_BATCH_SIZE === 0 ||
        vectorRows.length === totalVectorCount
      ) {
        const vectorProgress = totalVectorCount > 0 ? vectorRows.length / totalVectorCount : 1;
        reportProgress({
          progress: scaleProgress(0.82, 0.96, vectorProgress),
          statusText: 'Loading semantic vectors',
          detail: `${formatCount(vectorRows.length)} / ${formatCount(totalVectorCount)}`,
        });
      }
    }

    vectorStatement.free();

    reportProgress({
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
      videoGroups,
      semanticEntryIndexes: Int32Array.from(semanticIndexes),
      semanticVectors,
      vectorDim,
    };

    reportProgress({
      progress: 1,
      statusText: 'Ready',
      detail: `${formatCount(entries.length)} pairs loaded`,
    });

    return {
      totalEntries: entries.length,
      embeddingModelId: QUERY_MODEL_ID,
    };
  } finally {
    db.close();
  }
}

function createModelProgressCallback(reportProgress: ProgressReporter): (info: ModelProgressInfo) => void {
  const files = new Map<string, { loaded: number; total: number | null; done: boolean }>();

  const emit = (statusText: string, detail: string): void => {
    let completedFiles = 0;
    let progressTotal = 0;

    for (const file of files.values()) {
      if (file.done) {
        completedFiles += 1;
        progressTotal += 1;
      } else if (file.total !== null && file.total > 0) {
        progressTotal += Math.min(file.loaded / file.total, 1);
      }
    }

    const progress = files.size > 0 ? progressTotal / files.size : 0;

    reportProgress({
      progress,
      statusText,
      detail,
    });
  };

  return (info) => {
    if (info.status === 'ready') {
      reportProgress({
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
        emit('Checking model files', `${files.size} file${files.size === 1 ? '' : 's'} queued`);
        return;
      }

      case 'download': {
        files.set(key, current);
        emit('Downloading model files', `Fetching ${fileName}`);
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
        );
        return;
      }

      case 'done': {
        current.done = true;
        current.loaded = current.total ?? 1;
        current.total = current.total ?? 1;
        files.set(key, current);
        let completedFiles = 0;
        for (const file of files.values()) {
          if (file.done) completedFiles += 1;
        }
        emit(
          completedFiles === files.size ? 'Finalizing embedding model' : 'Downloading model files',
          `${completedFiles} / ${files.size} files ready`,
        );
        return;
      }
    }
  };
}

async function ensureExtractor(reportProgress?: ProgressReporter): Promise<Extractor> {
  if (extractor) {
    reportProgress?.({
      progress: 1,
      statusText: 'Ready',
      detail: QUERY_MODEL_ID,
    });
    return extractor;
  }

  if (!extractorPromise) {
    reportProgress?.({
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
    progress: 1,
    statusText: 'Ready',
    detail: QUERY_MODEL_ID,
  });
  return loadedExtractor;
}

async function embedQuery(query: string, reportProgress?: ProgressReporter): Promise<Float32Array> {
  const featureExtractor = await ensureExtractor(reportProgress);
  const output = await featureExtractor(query, {
    pooling: 'mean',
    normalize: true,
  });
  const data = output.data;
  return data instanceof Float32Array ? data : Float32Array.from(data);
}

function searchSemantic(loaded: LoadedState, queryVector: Float32Array): SearchResult[] {
  const topHits: RankedHit[] = [];

  for (let row = 0; row < loaded.semanticEntryIndexes.length; row += 1) {
    const entryIndex = loaded.semanticEntryIndexes[row]!;

    const offset = row * loaded.vectorDim;
    let score = 0;

    for (let dim = 0; dim < loaded.vectorDim; dim += 1) {
      score += loaded.semanticVectors[offset + dim]! * queryVector[dim]!;
    }

    if (score >= 0) {
      pushRankedHit(topHits, { entryIndex, score });
    }
  }

  return topHits.map(({ entryIndex, score }) => ({ ...loaded.entries[entryIndex]!, score }));
}

async function handleSearch(
  request: SearchRequest,
  reportProgress?: ProgressReporter,
): Promise<{ results: SearchResult[]; note?: string }> {
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

  const queryVector = await embedQuery(query, reportProgress);
  if (queryVector.length !== loaded.vectorDim) {
    throw new Error(
      `Query vector dimension mismatch: expected ${loaded.vectorDim}, got ${queryVector.length}.`,
    );
  }

  return { results: searchSemantic(loaded, queryVector) };
}

function handleTranscript(request: TranscriptRequest): EntrySummary[] {
  const loaded = ensureState();
  const group = loaded.videoGroups.get(request.videoId);

  if (!group || group.length === 0) {
    throw new Error(`No TM entries found for video ${request.videoId}.`);
  }

  return group.map((entryIndex) => loaded.entries[entryIndex]!);
}

workerScope.addEventListener('message', async (event: MessageEvent<WorkerRequest>) => {
  const request = event.data;

  try {
    switch (request.kind) {
      case 'boot': {
        const pairsProgress = createProgressReporter(request.requestId, 'pairs');

        const stats = await loadDatabase(request, pairsProgress);
        workerScope.postMessage({
          kind: 'boot:ok',
          requestId: request.requestId,
          stats,
        } satisfies WorkerResponse);
        return;
      }

      case 'prepare-model': {
        const modelProgress = createProgressReporter(request.requestId, 'model');
        await ensureExtractor(modelProgress);
        workerScope.postMessage({
          kind: 'prepare-model:ok',
          requestId: request.requestId,
        } satisfies WorkerResponse);
        return;
      }

      case 'search': {
        const modelProgress = createProgressReporter(request.requestId, 'model');
        const { results, note } = await handleSearch(request, modelProgress);
        workerScope.postMessage({
          kind: 'search:ok',
          requestId: request.requestId,
          results,
          note,
        } satisfies WorkerResponse);
        return;
      }

      case 'transcript': {
        const items = handleTranscript(request);
        workerScope.postMessage({
          kind: 'transcript:ok',
          requestId: request.requestId,
          items,
        } satisfies WorkerResponse);
        return;
      }

      default: {
        const exhaustive: never = request;
        throw new Error(`Unsupported worker request: ${String(exhaustive)}`);
      }
    }
  } catch (error) {
    workerScope.postMessage({
      kind: 'error',
      requestId: request.requestId,
      message: error instanceof Error ? error.message : String(error),
    } satisfies WorkerResponse);
  }
});
