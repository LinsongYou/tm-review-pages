import { startTransition, useEffect, useLayoutEffect, useRef, useState } from 'react';
import { classNames } from './classes';
import { getDisplayModelName } from './format';
import { handleSelectKey } from './keyboard';
import type {
  BootProgressSnapshot,
  BootStats,
  ContextItem,
  SearchResult,
  WorkerPayload,
  WorkerResponse,
} from './search/protocol';
import TmAtlasPanel from './startup/TmAtlasPanel';
import type { SemanticLandscapeData } from './startup/semantic-landscape';

type PendingRequest = {
  resolve: (value: WorkerResponse) => void;
  reject: (reason?: unknown) => void;
};

type Theme = 'dark' | 'light';
type HeaderLoadState = Record<BootProgressSnapshot['target'], BootProgressSnapshot>;

const DB_ASSET = 'data/tm_misha_minilm.db';
const STARTUP_DATA_ASSET = 'data/startup-visualizations.json';
const THEME_STORAGE_KEY = 'tm-review-theme';
const DEFAULT_CONTEXT_RADIUS = 4;
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

function formatCueTimestamp(ms: number | null): string {
  if (ms === null || !Number.isFinite(ms)) {
    return '--:--.---';
  }

  const totalMs = Math.max(0, Math.round(ms));
  const hours = Math.floor(totalMs / 3_600_000);
  const minutes = Math.floor((totalMs % 3_600_000) / 60_000);
  const seconds = Math.floor((totalMs % 60_000) / 1_000);
  const milliseconds = totalMs % 1_000;
  const secondFragment = `${seconds.toString().padStart(2, '0')}.${milliseconds
    .toString()
    .padStart(3, '0')}`;

  if (hours > 0) {
    return `${hours}:${minutes.toString().padStart(2, '0')}:${secondFragment}`;
  }

  return `${minutes.toString().padStart(2, '0')}:${secondFragment}`;
}

function formatCueRange(startMs: number | null, endMs: number | null): string {
  if (startMs === null && endMs === null) {
    return 'No timestamps';
  }

  if (startMs === null) {
    return `Ends ${formatCueTimestamp(endMs)}`;
  }

  if (endMs === null) {
    return `Starts ${formatCueTimestamp(startMs)}`;
  }

  return `${formatCueTimestamp(startMs)} - ${formatCueTimestamp(endMs)}`;
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
  const latestContextRef = useRef(0);
  const latestTranscriptRef = useRef(0);
  const transcriptBodyRef = useRef<HTMLDivElement | null>(null);
  const transcriptItemRefs = useRef(new Map<string, HTMLLIElement>());

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
  const [contextItems, setContextItems] = useState<ContextItem[]>([]);
  const [transcriptVideoId, setTranscriptVideoId] = useState<string | null>(null);
  const [transcriptItems, setTranscriptItems] = useState<ContextItem[]>([]);
  const [transcriptFocusEntryId, setTranscriptFocusEntryId] = useState<string | null>(null);
  const [transcriptLoading, setTranscriptLoading] = useState(false);
  const [transcriptErrorText, setTranscriptErrorText] = useState<string | null>(null);
  const [startupData, setStartupData] = useState<SemanticLandscapeData | null>(null);
  const [startupDataLoading, setStartupDataLoading] = useState(true);
  const [startupDataErrorText, setStartupDataErrorText] = useState<string | null>(null);

  const dbUrl = `${import.meta.env.BASE_URL}${withAssetVersion(DB_ASSET, __TM_DB_VERSION__)}`;
  const startupDataUrl = `${import.meta.env.BASE_URL}${withAssetVersion(
    STARTUP_DATA_ASSET,
    __TM_STARTUP_DATA_VERSION__,
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
    void loadStartupData(controller.signal);
    return () => controller.abort();
  }, [startupDataUrl]);

  useEffect(() => {
    if (!selectedEntryId || !bootStats) {
      setContextItems([]);
      return;
    }

    const sequence = latestContextRef.current + 1;
    latestContextRef.current = sequence;

    void (async () => {
      try {
        const response = (await callWorker({
          kind: 'context',
          entryId: selectedEntryId,
          radius: DEFAULT_CONTEXT_RADIUS,
        })) as Extract<WorkerResponse, { kind: 'context:ok' }>;

        if (latestContextRef.current !== sequence) {
          return;
        }

        startTransition(() => {
          setContextItems(response.context);
        });
      } catch (error) {
        if (latestContextRef.current !== sequence) {
          return;
        }

        setErrorText(error instanceof Error ? error.message : String(error));
      }
    })();
  }, [bootStats, selectedEntryId]);

  useEffect(() => {
    if (!transcriptVideoId) {
      return;
    }

    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = 'hidden';

    const handleEscape = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        closeTranscript();
      }
    };

    window.addEventListener('keydown', handleEscape);
    return () => {
      document.body.style.overflow = previousOverflow;
      window.removeEventListener('keydown', handleEscape);
    };
  }, [transcriptVideoId]);

  useEffect(() => {
    if (!transcriptVideoId || transcriptItems.length === 0) {
      return;
    }

    const nextEntryId =
      transcriptFocusEntryId ??
      transcriptItems.find((item) => item.entryId === selectedEntryId)?.entryId ??
      transcriptItems[0]?.entryId ??
      null;

    if (!nextEntryId) {
      return;
    }

    const frameId = window.requestAnimationFrame(() => {
      const container = transcriptBodyRef.current;
      const entry = transcriptItemRefs.current.get(nextEntryId);

      if (!container || !entry) {
        return;
      }

      const containerBounds = container.getBoundingClientRect();
      const entryBounds = entry.getBoundingClientRect();
      const top =
        container.scrollTop +
        (entryBounds.top - containerBounds.top) -
        (container.clientHeight - entryBounds.height) / 2;
      container.scrollTo({
        top: Math.max(0, top),
        behavior: 'auto',
      });
    });

    return () => window.cancelAnimationFrame(frameId);
  }, [selectedEntryId, transcriptFocusEntryId, transcriptItems, transcriptVideoId]);

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

  async function loadStartupData(signal: AbortSignal): Promise<void> {
    setStartupDataLoading(true);
    setStartupDataErrorText(null);

    try {
      const response = await fetch(startupDataUrl, { signal });
      if (!response.ok) {
        throw new Error(`Failed to download ${startupDataUrl} (${response.status}).`);
      }

      const payload = (await response.json()) as SemanticLandscapeData;
      if (signal.aborted) {
        return;
      }

      startTransition(() => {
        setStartupData(payload);
        setStartupDataLoading(false);
      });
    } catch (error) {
      if (signal.aborted) {
        return;
      }

      setStartupDataLoading(false);
      setStartupDataErrorText(error instanceof Error ? error.message : String(error));
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

  function setTranscriptItemRef(entryId: string, element: HTMLLIElement | null): void {
    if (element) {
      transcriptItemRefs.current.set(entryId, element);
      return;
    }

    transcriptItemRefs.current.delete(entryId);
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
    transcriptItemRefs.current.clear();

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
    transcriptItemRefs.current.clear();
    setTranscriptVideoId(null);
    setTranscriptItems([]);
    setTranscriptFocusEntryId(null);
    setTranscriptLoading(false);
    setTranscriptErrorText(null);
  }

  function clearAtlas(): void {
    latestSearchRef.current += 1;
    latestContextRef.current += 1;
    closeTranscript();
    setQuery('');
    setSearching(false);
    setErrorText(null);
    setSearchNote(null);
    setResults([]);
    setSelectedEntryId(null);
    setContextItems([]);
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

  const transcriptHasTimestamps = transcriptItems.some(
    (item) => item.startMs !== null || item.endMs !== null,
  );

  return (
    <main className="app-shell">
      <TmAtlasPanel
        data={startupData}
        dataLoading={startupDataLoading}
        dataErrorText={startupDataErrorText}
        theme={theme}
        statusItems={statusItems}
        query={query}
        searchReady={!booting && !!bootStats}
        searching={searching}
        searchResults={results}
        searchNote={searchNote}
        errorText={errorText}
        selectedEntryId={selectedEntryId}
        contextItems={contextItems}
        onQueryChange={setQuery}
        onSearch={() => {
          void runSearch();
        }}
        onSelectEntry={setSelectedEntryId}
        onOpenTranscript={(videoId, focusEntryId) => {
          void openTranscript(videoId, focusEntryId);
        }}
        onClear={clearAtlas}
        onToggleTheme={toggleTheme}
      />

      {transcriptVideoId ? (
        <div className="modal-backdrop" role="presentation" onClick={closeTranscript}>
          <section
            aria-labelledby="transcript-dialog-title"
            aria-modal="true"
            className="transcript-modal"
            role="dialog"
            onClick={(event) => event.stopPropagation()}
          >
            <div className="transcript-modal-header">
              <div className="transcript-heading">
                <span>YouTube ID</span>
                <h2 id="transcript-dialog-title">{transcriptVideoId}</h2>
                <p>
                  {transcriptLoading
                    ? 'Loading transcript cues...'
                    : `${transcriptItems.length.toLocaleString()} cues${
                        transcriptHasTimestamps ? ' with cue timestamps' : ''
                      }`}
                </p>
              </div>

              <button className="modal-close" type="button" onClick={closeTranscript}>
                Close
              </button>
            </div>

            {transcriptLoading ? (
              <div className="empty-state">
                <p>Loading the full transcript...</p>
              </div>
            ) : transcriptErrorText ? (
              <div className="empty-state">
                <p>{transcriptErrorText}</p>
              </div>
            ) : (
              <div ref={transcriptBodyRef} className="transcript-modal-body">
                <ol className="transcript-row-list">
                  {transcriptItems.map((item) => {
                    const isSelected = item.entryId === selectedEntryId;

                    return (
                      <li
                        key={item.entryId}
                        ref={(element) => setTranscriptItemRef(item.entryId, element)}
                        className="transcript-row"
                      >
                        <button
                          aria-label={`Jump to ${item.videoId} cue ${item.segIndex}. ${formatCueRange(
                            item.startMs,
                            item.endMs,
                          )}.`}
                          className={classNames('transcript-time', isSelected && 'is-selected')}
                          type="button"
                          onClick={() => setTranscriptSelection(item.entryId)}
                        >
                          <span>#{item.segIndex}</span>
                          <strong>{formatCueTimestamp(item.startMs)}</strong>
                          <em>{formatCueTimestamp(item.endMs)}</em>
                        </button>

                        <article
                          className={classNames('transcript-entry', isSelected && 'is-selected')}
                          role="button"
                          tabIndex={0}
                          onClick={() => setTranscriptSelection(item.entryId)}
                          onKeyDown={(event) =>
                            handleSelectKey(event, () => setTranscriptSelection(item.entryId))
                          }
                        >
                          <div className="transcript-entry-meta">
                            <span>{item.videoId}#{item.segIndex}</span>
                            <span>{item.blockName || 'no block'}</span>
                          </div>
                          <div className="transcript-entry-copy-line">
                            <p className="transcript-entry-line transcript-entry-line--en">{item.en}</p>
                            <button
                              aria-label={`Search using cue ${item.segIndex} English text`}
                              className="transcript-entry-action"
                              type="button"
                              disabled={!item.en.trim()}
                              onClick={(event) => {
                                event.stopPropagation();
                                searchFromTranscriptLine(item.en);
                              }}
                            >
                              Search
                            </button>
                          </div>
                          <p className="transcript-entry-line transcript-entry-line--zh">{item.zh}</p>
                        </article>
                      </li>
                    );
                  })}
                </ol>
              </div>
            )}
          </section>
        </div>
      ) : null}
    </main>
  );
}

export default App;
