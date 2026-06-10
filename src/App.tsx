import { startTransition, useEffect, useLayoutEffect, useRef, useState } from 'react';
import { getDisplayModelName } from './format';
import type {
  BootProgressSnapshot,
  BootStats,
  WorkerPayload,
  WorkerResponse,
} from './search/protocol';
import TmAtlasPanel, { type AtlasNavigationState } from './atlas/TmAtlasPanel';
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
const EMPTY_TRANSCRIPT = {
  transcriptVideoId: null,
  transcriptItems: [],
  transcriptFocusEntryId: null,
  transcriptLoading: false,
  transcriptErrorText: null,
} satisfies Pick<
  AtlasNavigationState,
  | 'transcriptVideoId'
  | 'transcriptItems'
  | 'transcriptFocusEntryId'
  | 'transcriptLoading'
  | 'transcriptErrorText'
>;
const INITIAL_NAVIGATION: AtlasNavigationState = {
  query: '',
  searchResults: [],
  searchNote: null,
  errorText: null,
  selectedEntryId: null,
  ...EMPTY_TRANSCRIPT,
};

function withAssetVersion(path: string, version: string): string {
  return `${path}?v=${encodeURIComponent(version)}`;
}

function getInitialTheme(): Theme {
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
      progress: 0,
      statusText: 'Preparing TM snapshot',
    },
    model: {
      target: 'model',
      progress: 0,
      statusText: 'Queued after pairs',
      detail: 'Waiting for TM pairs',
    },
  };
}

function formatProgressPercent(progress: number): string {
  return `${Math.round(progress * 100)}%`;
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
  const [searching, setSearching] = useState(false);
  const [theme, setTheme] = useState<Theme>(getInitialTheme);
  const [navigation, setNavigation] = useState<AtlasNavigationState>(INITIAL_NAVIGATION);
  const [atlasData, setAtlasData] = useState<SemanticLandscapeData | null>(null);
  const [atlasDataErrorText, setAtlasDataErrorText] = useState<string | null>(null);
  const { query } = navigation;

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

      setNavigation((current) => ({
        ...current,
        errorText: event.message || 'The search worker crashed.',
      }));
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

  async function boot(isStale: () => boolean): Promise<void> {
    setBootStats(null);
    setNavigation((current) => ({ ...current, errorText: null }));
    setModelReady(false);
    setBootProgress(createInitialBootProgressState());

    try {
      const response = (await callWorker({
        kind: 'boot',
        dbUrl,
      })) as Extract<WorkerResponse, { kind: 'boot:ok' }>;

      if (isStale()) {
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
            progress: 0,
            statusText: 'Warming cache',
            detail: response.stats.embeddingModelId,
          },
        }));
      });
      void prepareModel();
    } catch (error) {
      if (isStale()) {
        return;
      }

      setNavigation((current) => ({
        ...current,
        errorText: error instanceof Error ? error.message : String(error),
      }));
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
      });
    } catch (error) {
      if (signal.aborted) {
        return;
      }

      setAtlasDataErrorText(error instanceof Error ? error.message : String(error));
    }
  }

  async function runSearch(nextQuery = query): Promise<void> {
    const trimmed = nextQuery.trim();

    if (!trimmed) {
      latestSearchRef.current += 1;
      setNavigation((current) => ({ ...current, searchResults: [], searchNote: null }));
      return;
    }

    if (!bootStats) {
      setNavigation((current) => ({ ...current, errorText: 'The TM database is still loading.' }));
      return;
    }

    const sequence = latestSearchRef.current + 1;
    latestSearchRef.current = sequence;
    latestTranscriptRef.current += 1;

    setSearching(true);
    setNavigation((current) => ({
      ...current,
      ...EMPTY_TRANSCRIPT,
      errorText: null,
      searchNote: null,
    }));

    try {
      const response = (await callWorker({
        kind: 'search',
        query: trimmed,
      })) as Extract<WorkerResponse, { kind: 'search:ok' }>;

      if (latestSearchRef.current !== sequence) {
        return;
      }

      startTransition(() => {
        setNavigation((current) => ({
          ...current,
          searchResults: response.results,
          selectedEntryId: response.results[0]?.entryId ?? null,
          searchNote: response.note ?? null,
        }));
        setSearching(false);
      });
    } catch (error) {
      if (latestSearchRef.current !== sequence) {
        return;
      }

      setSearching(false);
      setNavigation((current) => ({
        ...current,
        errorText: error instanceof Error ? error.message : String(error),
      }));
    }
  }

  function setTranscriptSelection(entryId: string): void {
    setNavigation((current) => ({
      ...current,
      selectedEntryId: entryId,
      transcriptFocusEntryId: entryId,
    }));
  }

  function searchFromLine(text: string): void {
    const trimmed = text.trim();
    if (!trimmed) {
      return;
    }

    setNavigation((current) => ({ ...current, query: trimmed }));
    void runSearch(trimmed);
  }

  async function openTranscript(videoId: string, focusEntryId: string): Promise<void> {
    if (!bootStats) {
      setNavigation((current) => ({ ...current, errorText: 'The TM database is still loading.' }));
      return;
    }

    const sequence = latestTranscriptRef.current + 1;
    latestTranscriptRef.current = sequence;
    latestSearchRef.current += 1;

    setSearching(false);
    setNavigation({
      query: '',
      searchResults: [],
      searchNote: null,
      errorText: null,
      selectedEntryId: focusEntryId,
      transcriptVideoId: videoId,
      transcriptItems: [],
      transcriptFocusEntryId: focusEntryId,
      transcriptLoading: true,
      transcriptErrorText: null,
    });

    try {
      const response = (await callWorker({
        kind: 'transcript',
        videoId,
      })) as Extract<WorkerResponse, { kind: 'transcript:ok' }>;

      if (latestTranscriptRef.current !== sequence) {
        return;
      }

      startTransition(() => {
        setNavigation((current) => ({
          ...current,
          transcriptItems: response.items,
          transcriptLoading: false,
        }));
      });
    } catch (error) {
      if (latestTranscriptRef.current !== sequence) {
        return;
      }

      setNavigation((current) => ({
        ...current,
        transcriptLoading: false,
        transcriptErrorText: error instanceof Error ? error.message : String(error),
      }));
    }
  }

  function closeTranscript(): void {
    latestTranscriptRef.current += 1;
    setNavigation((current) => ({ ...current, ...EMPTY_TRANSCRIPT }));
  }

  function clearAtlas(): void {
    latestSearchRef.current += 1;
    latestTranscriptRef.current += 1;
    setSearching(false);
    setNavigation(INITIAL_NAVIGATION);
  }

  function restoreNavigationState(state: AtlasNavigationState): void {
    latestSearchRef.current += 1;
    latestTranscriptRef.current += 1;
    setSearching(false);
    setNavigation(state);
  }

  const statusItems = [
    {
      label: 'Pairs',
      value: bootStats
        ? bootStats.totalEntries.toLocaleString()
        : formatProgressPercent(bootProgress.pairs.progress),
      detail: bootProgress.pairs.detail ?? bootProgress.pairs.statusText,
      progress: bootProgress.pairs.progress,
      ready: !!bootStats,
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
        dataLoading={!atlasData && !atlasDataErrorText}
        dataErrorText={atlasDataErrorText}
        theme={theme}
        statusItems={statusItems}
        navigation={navigation}
        searchReady={!!bootStats}
        searching={searching}
        onQueryChange={(nextQuery) => {
          setNavigation((current) => ({ ...current, query: nextQuery }));
        }}
        onSearch={() => {
          void runSearch();
        }}
        onSelectEntry={(entryId) => {
          setNavigation((current) => ({ ...current, selectedEntryId: entryId }));
        }}
        onOpenTranscript={(videoId, focusEntryId) => {
          void openTranscript(videoId, focusEntryId);
        }}
        onSelectTranscriptEntry={setTranscriptSelection}
        onSearchLine={searchFromLine}
        onCloseTranscript={closeTranscript}
        onRestoreNavigationState={restoreNavigationState}
        onClear={clearAtlas}
        onToggleTheme={() => {
          setTheme((currentTheme) => (currentTheme === 'dark' ? 'light' : 'dark'));
        }}
      />
    </main>
  );
}

export default App;
