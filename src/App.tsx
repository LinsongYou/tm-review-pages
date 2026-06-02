import { startTransition, useEffect, useLayoutEffect, useRef, useState } from 'react';
import { getDisplayModelName } from './format';
import type {
  BootProgressSnapshot,
  BootStats,
  ContextItem,
  SearchResult,
  WorkerPayload,
  WorkerResponse,
} from './search/protocol';
import TmAtlasPanel from './atlas/TmAtlasPanel';
import type { SemanticLandscapeData } from './atlas/semantic-landscape';

type PendingRequest = {
  resolve: (value: WorkerResponse) => void;
  reject: (reason?: unknown) => void;
};

type Theme = 'dark' | 'light';
type HeaderLoadState = Record<BootProgressSnapshot['target'], BootProgressSnapshot>;

const DB_ASSET = 'data/tm_misha_minilm.db';
const ATLAS_DATA_ASSET = 'data/tm-atlas.json';
const THEME_STORAGE_KEY = 'tm-review-theme';
const DEFAULT_SEARCH_TOP_K = 24;
const PAIRS_CHIP_LABEL = 'English/中文 Pairs';
const MODEL_CHIP_LABEL = 'Embedding Model';

function withAssetVersion(path: string, version: string): string {
  return `${path}?v=${encodeURIComponent(version)}`;
}

function getInitialTheme(): Theme {
  if (typeof window === 'undefined') {
    return 'dark';
  }

  const storedTheme = window.localStorage.getItem(THEME_STORAGE_KEY);
  if (storedTheme === 'dark' || storedTheme === 'light') {
    return storedTheme;
  }

  return window.matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark';
}

function createInitialBootProgressState(): HeaderLoadState {
  return {
    pairs: {
      target: 'pairs',
      name: PAIRS_CHIP_LABEL,
      progress: 0,
      statusText: 'Preparing TM snapshot',
    },
    model: {
      target: 'model',
      name: MODEL_CHIP_LABEL,
      progress: 0,
      statusText: 'Queued after pairs',
      detail: 'Waiting for TM pairs',
    },
  };
}

function formatProgressPercent(progress: number): string {
  const bounded = Math.min(1, Math.max(0, progress));
  return `${Math.round(bounded * 100)}%`;
}

function App() {
  const workerRef = useRef<Worker | null>(null);
  const pendingRef = useRef(new Map<number, PendingRequest>());
  const requestIdRef = useRef(1);
  const latestSearchRef = useRef(0);
  const latestTranscriptRef = useRef(0);

  const [bootStats, setBootStats] = useState<BootStats | null>(null);
  const [bootProgress, setBootProgress] = useState<HeaderLoadState>(createInitialBootProgressState);
  const [modelReady, setModelReady] = useState(false);
  const [booting, setBooting] = useState(true);
  const [searching, setSearching] = useState(false);
  const [errorText, setErrorText] = useState<string | null>(null);
  const [searchNote, setSearchNote] = useState<string | null>(null);
  const [query, setQuery] = useState('');
  const [theme, setTheme] = useState<Theme>(getInitialTheme);
  const [results, setResults] = useState<SearchResult[]>([]);
  const [selectedEntryId, setSelectedEntryId] = useState<string | null>(null);
  const [transcriptVideoId, setTranscriptVideoId] = useState<string | null>(null);
  const [transcriptItems, setTranscriptItems] = useState<ContextItem[]>([]);
  const [transcriptFocusEntryId, setTranscriptFocusEntryId] = useState<string | null>(null);
  const [transcriptLoading, setTranscriptLoading] = useState(false);
  const [transcriptErrorText, setTranscriptErrorText] = useState<string | null>(null);
  const [atlasData, setAtlasData] = useState<SemanticLandscapeData | null>(null);
  const [atlasDataLoading, setAtlasDataLoading] = useState(true);
  const [atlasDataErrorText, setAtlasDataErrorText] = useState<string | null>(null);

  const dbUrl = `${import.meta.env.BASE_URL}${withAssetVersion(DB_ASSET, __TM_DB_VERSION__)}`;
  const atlasDataUrl = `${import.meta.env.BASE_URL}${withAssetVersion(
    ATLAS_DATA_ASSET,
    __TM_ATLAS_DATA_VERSION__,
  )}`;

  useLayoutEffect(() => {
    document.documentElement.dataset.theme = theme;
    document.documentElement.style.colorScheme = theme;
    window.localStorage.setItem(THEME_STORAGE_KEY, theme);
  }, [theme]);

  useEffect(() => {
    let disposed = false;
    const worker = new Worker(new URL('./search/search.worker.ts', import.meta.url), {
      type: 'module',
    });
    workerRef.current = worker;

    worker.addEventListener('message', (event: MessageEvent<WorkerResponse>) => {
      const message = event.data;

      if (message.kind === 'boot:progress') {
        startTransition(() => {
          setBootProgress((current) => ({
            ...current,
            [message.progress.target]: message.progress,
          }));
          if (message.progress.target === 'model' && message.progress.statusText === 'Ready') {
            setModelReady(true);
          }
        });
        return;
      }

      const pending = pendingRef.current.get(message.requestId);
      if (!pending) {
        return;
      }

      pendingRef.current.delete(message.requestId);

      if (message.kind === 'error') {
        pending.reject(new Error(message.message));
        return;
      }

      pending.resolve(message);
    });

    worker.addEventListener('error', (event) => {
      if (disposed) {
        return;
      }

      setErrorText(event.message || 'The search worker crashed.');
      setBooting(false);
      setSearching(false);
    });

    void boot(() => disposed);

    return () => {
      disposed = true;
      for (const pending of pendingRef.current.values()) {
        pending.reject(new Error('Worker terminated.'));
      }
      pendingRef.current.clear();
      worker.terminate();
      workerRef.current = null;
    };
  }, [dbUrl]);

  useEffect(() => {
    const controller = new AbortController();
    void loadAtlasData(controller.signal);
    return () => controller.abort();
  }, [atlasDataUrl]);

  async function callWorker(payload: WorkerPayload): Promise<WorkerResponse> {
    const worker = workerRef.current;
    if (!worker) {
      throw new Error('Worker is not available.');
    }

    const requestId = requestIdRef.current;
    requestIdRef.current += 1;

    return new Promise((resolve, reject) => {
      pendingRef.current.set(requestId, { resolve, reject });
      worker.postMessage({ ...payload, requestId });
    });
  }

  async function boot(isStale?: () => boolean): Promise<void> {
    setBooting(true);
    setErrorText(null);
    setModelReady(false);
    setBootProgress(createInitialBootProgressState());

    try {
      const response = (await callWorker({
        kind: 'boot',
        dbUrl,
      })) as Extract<WorkerResponse, { kind: 'boot:ok' }>;

      if (isStale?.()) {
        return;
      }

      startTransition(() => {
        setBootStats(response.stats);
        setBootProgress((current) => ({
          pairs: {
            ...current.pairs,
            progress: 1,
            statusText: 'Ready',
            detail: `${response.stats.totalEntries.toLocaleString()} pairs loaded`,
          },
          model: {
            target: 'model',
            name: getDisplayModelName(response.stats.embeddingModelId),
            progress: 0,
            statusText: 'Warming cache',
            detail: response.stats.embeddingModelId,
          },
        }));
        setBooting(false);
      });
      void prepareModel();
    } catch (error) {
      if (isStale?.()) {
        return;
      }

      setBooting(false);
      setErrorText(error instanceof Error ? error.message : String(error));
    }
  }

  async function prepareModel(): Promise<void> {
    try {
      await callWorker({ kind: 'prepare-model' });
      setModelReady(true);
    } catch (error) {
      setBootProgress((current) => ({
        ...current,
        model: {
          ...current.model,
          statusText: 'Model load failed',
          detail: error instanceof Error ? error.message : String(error),
        },
      }));
    }
  }

  async function loadAtlasData(signal: AbortSignal): Promise<void> {
    setAtlasDataLoading(true);
    setAtlasDataErrorText(null);

    try {
      const response = await fetch(atlasDataUrl, { signal });
      if (!response.ok) {
        throw new Error(`Failed to download ${atlasDataUrl} (${response.status}).`);
      }

      const payload = (await response.json()) as SemanticLandscapeData;
      if (signal.aborted) {
        return;
      }

      startTransition(() => {
        setAtlasData(payload);
        setAtlasDataLoading(false);
      });
    } catch (error) {
      if (signal.aborted) {
        return;
      }

      setAtlasDataLoading(false);
      setAtlasDataErrorText(error instanceof Error ? error.message : String(error));
    }
  }

  async function runSearch(nextQuery = query): Promise<void> {
    const trimmed = nextQuery.trim();

    if (!trimmed) {
      latestSearchRef.current += 1;
      setResults([]);
      setSearchNote(null);
      return;
    }

    if (!bootStats) {
      setErrorText('The TM database is still loading.');
      return;
    }

    const sequence = latestSearchRef.current + 1;
    latestSearchRef.current = sequence;

    setSearching(true);
    setErrorText(null);
    setSearchNote(null);

    try {
      const response = (await callWorker({
        kind: 'search',
        query: trimmed,
        topK: DEFAULT_SEARCH_TOP_K,
        minLength: 0,
        minScore: 0,
      })) as Extract<WorkerResponse, { kind: 'search:ok' }>;

      if (latestSearchRef.current !== sequence) {
        return;
      }

      startTransition(() => {
        setResults(response.results);
        setSelectedEntryId(response.results[0]?.entryId ?? null);
        setSearchNote(response.note ?? null);
        setSearching(false);
      });
    } catch (error) {
      if (latestSearchRef.current !== sequence) {
        return;
      }

      setSearching(false);
      setErrorText(error instanceof Error ? error.message : String(error));
    }
  }

  function setTranscriptSelection(entryId: string): void {
    setSelectedEntryId(entryId);
    setTranscriptFocusEntryId(entryId);
  }

  function searchFromTranscriptLine(text: string): void {
    const trimmed = text.trim();
    if (!trimmed) {
      return;
    }

    closeTranscript();
    setQuery(trimmed);
    void runSearch(trimmed);
  }

  async function openTranscript(videoId: string, focusEntryId: string): Promise<void> {
    if (!bootStats) {
      setErrorText('The TM database is still loading.');
      return;
    }

    const sequence = latestTranscriptRef.current + 1;
    latestTranscriptRef.current = sequence;
    latestSearchRef.current += 1;

    setQuery('');
    setSearching(false);
    setErrorText(null);
    setSearchNote(null);
    setResults([]);
    setSelectedEntryId(focusEntryId);
    setTranscriptVideoId(videoId);
    setTranscriptItems([]);
    setTranscriptFocusEntryId(focusEntryId);
    setTranscriptLoading(true);
    setTranscriptErrorText(null);

    try {
      const response = (await callWorker({
        kind: 'transcript',
        videoId,
        focusEntryId,
      })) as Extract<WorkerResponse, { kind: 'transcript:ok' }>;

      if (latestTranscriptRef.current !== sequence) {
        return;
      }

      startTransition(() => {
        setTranscriptItems(response.items);
        setTranscriptLoading(false);
      });
    } catch (error) {
      if (latestTranscriptRef.current !== sequence) {
        return;
      }

      setTranscriptLoading(false);
      setTranscriptErrorText(error instanceof Error ? error.message : String(error));
    }
  }

  function closeTranscript(): void {
    latestTranscriptRef.current += 1;
    setTranscriptVideoId(null);
    setTranscriptItems([]);
    setTranscriptFocusEntryId(null);
    setTranscriptLoading(false);
    setTranscriptErrorText(null);
  }

  function clearAtlas(): void {
    latestSearchRef.current += 1;
    closeTranscript();
    setQuery('');
    setSearching(false);
    setErrorText(null);
    setSearchNote(null);
    setResults([]);
    setSelectedEntryId(null);
  }

  function toggleTheme(): void {
    setTheme((currentTheme) => (currentTheme === 'dark' ? 'light' : 'dark'));
  }

  const statusItems = [
    {
      label: 'Pairs',
      value: bootStats
        ? bootStats.totalEntries.toLocaleString()
        : formatProgressPercent(bootProgress.pairs.progress),
      detail: bootProgress.pairs.detail ?? bootProgress.pairs.statusText,
      progress: bootProgress.pairs.progress,
      ready: !booting && !!bootStats,
    },
    {
      label: 'Model',
      value: modelReady && bootStats
        ? getDisplayModelName(bootStats.embeddingModelId)
        : bootProgress.model.progress > 0
          ? formatProgressPercent(bootProgress.model.progress)
          : 'Idle',
      detail: bootProgress.model.detail ?? bootProgress.model.statusText,
      progress: bootProgress.model.progress,
      ready: modelReady,
    },
  ];

  return (
    <main className="app-shell">
      <TmAtlasPanel
        data={atlasData}
        dataLoading={atlasDataLoading}
        dataErrorText={atlasDataErrorText}
        theme={theme}
        statusItems={statusItems}
        query={query}
        searchReady={!booting && !!bootStats}
        searching={searching}
        searchResults={results}
        searchNote={searchNote}
        errorText={errorText}
        selectedEntryId={selectedEntryId}
        transcriptVideoId={transcriptVideoId}
        transcriptItems={transcriptItems}
        transcriptFocusEntryId={transcriptFocusEntryId}
        transcriptLoading={transcriptLoading}
        transcriptErrorText={transcriptErrorText}
        onQueryChange={setQuery}
        onSearch={() => {
          void runSearch();
        }}
        onSelectEntry={setSelectedEntryId}
        onOpenTranscript={(videoId, focusEntryId) => {
          void openTranscript(videoId, focusEntryId);
        }}
        onSelectTranscriptEntry={setTranscriptSelection}
        onSearchTranscriptLine={searchFromTranscriptLine}
        onCloseTranscript={closeTranscript}
        onClear={clearAtlas}
        onToggleTheme={toggleTheme}
      />
    </main>
  );
}

export default App;
