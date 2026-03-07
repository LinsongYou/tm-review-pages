import {
  FormEvent,
  KeyboardEvent as ReactKeyboardEvent,
  startTransition,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
} from 'react';
import type {
  BootStats,
  ContextItem,
  SearchResult,
  WorkerPayload,
  WorkerResponse,
} from './search/protocol';
import SemanticLandscapePanel from './startup/SemanticLandscapePanel';
import type { SemanticLandscapeData } from './startup/semantic-landscape';

type PendingRequest = {
  resolve: (value: WorkerResponse) => void;
  reject: (reason?: unknown) => void;
};

type Theme = 'dark' | 'light';

const VECTOR_MODEL_ID = 'sentence-transformers/all-MiniLM-L6-v2';
const QUERY_MODEL_ID = 'Xenova/all-MiniLM-L6-v2';
const DB_ASSET = 'data/tm_misha_minilm.db';
const LANDSCAPE_ASSET = 'data/semantic-landscape.json';
const THEME_STORAGE_KEY = 'tm-review-theme';
const DEFAULT_CONTEXT_RADIUS = 3;

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

function formatStat(value: number): string {
  if (!Number.isFinite(value)) {
    return '0';
  }

  const rounded = Math.round(value);
  const isWhole = Math.abs(value - rounded) < 0.001;

  return value.toLocaleString(undefined, {
    minimumFractionDigits: isWhole ? 0 : 1,
    maximumFractionDigits: isWhole ? 0 : 1,
  });
}

function App() {
  const workerRef = useRef<Worker | null>(null);
  const pendingRef = useRef(new Map<number, PendingRequest>());
  const requestIdRef = useRef(1);
  const latestSearchRef = useRef(0);
  const latestContextRef = useRef(0);
  const latestTranscriptRef = useRef(0);

  const [bootStats, setBootStats] = useState<BootStats | null>(null);
  const [booting, setBooting] = useState(true);
  const [searching, setSearching] = useState(false);
  const [errorText, setErrorText] = useState<string | null>(null);
  const [searchNote, setSearchNote] = useState<string | null>(null);
  const [query, setQuery] = useState('');
  const [topK, setTopK] = useState(10);
  const [minLength, setMinLength] = useState(0);
  const [minScore, setMinScore] = useState(0);
  const [contextRadius, setContextRadius] = useState(DEFAULT_CONTEXT_RADIUS);
  const [theme, setTheme] = useState<Theme>(getInitialTheme);
  const [hasSearched, setHasSearched] = useState(false);
  const [results, setResults] = useState<SearchResult[]>([]);
  const [selectedEntryId, setSelectedEntryId] = useState<string | null>(null);
  const [contextItems, setContextItems] = useState<ContextItem[]>([]);
  const [transcriptVideoId, setTranscriptVideoId] = useState<string | null>(null);
  const [transcriptItems, setTranscriptItems] = useState<ContextItem[]>([]);
  const [transcriptLoading, setTranscriptLoading] = useState(false);
  const [transcriptErrorText, setTranscriptErrorText] = useState<string | null>(null);
  const [landscapeData, setLandscapeData] = useState<SemanticLandscapeData | null>(null);
  const [landscapeLoading, setLandscapeLoading] = useState(true);
  const [landscapeErrorText, setLandscapeErrorText] = useState<string | null>(null);

  const dbUrl = useMemo(() => `${import.meta.env.BASE_URL}${DB_ASSET}`, []);
  const landscapeUrl = useMemo(() => `${import.meta.env.BASE_URL}${LANDSCAPE_ASSET}`, []);

  useLayoutEffect(() => {
    document.documentElement.dataset.theme = theme;
    document.documentElement.style.colorScheme = theme;
    window.localStorage.setItem(THEME_STORAGE_KEY, theme);
  }, [theme]);

  useEffect(() => {
    const worker = new Worker(new URL('./search/search.worker.ts', import.meta.url), {
      type: 'module',
    });
    workerRef.current = worker;

    worker.addEventListener('message', (event: MessageEvent<WorkerResponse>) => {
      const message = event.data;

      if (message.kind === 'status') {
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
      setErrorText(event.message || 'The search worker crashed.');
      setBooting(false);
      setSearching(false);
    });

    void boot();

    return () => {
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

    void loadSemanticLandscape(controller.signal);

    return () => {
      controller.abort();
    };
  }, [landscapeUrl]);

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
          radius: contextRadius,
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
  }, [bootStats, contextRadius, selectedEntryId]);

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

  async function callWorker(payload: WorkerPayload): Promise<WorkerResponse> {
    const worker = workerRef.current;
    if (!worker) {
      throw new Error('Worker is not available.');
    }

    const requestId = requestIdRef.current;
    requestIdRef.current += 1;

    const request = { ...payload, requestId };

    return new Promise((resolve, reject) => {
      pendingRef.current.set(requestId, { resolve, reject });
      worker.postMessage(request);
    });
  }

  async function boot(): Promise<void> {
    setBooting(true);
    setErrorText(null);

    try {
      const response = (await callWorker({
        kind: 'boot',
        dbUrl,
        vectorModelId: VECTOR_MODEL_ID,
        queryModelId: QUERY_MODEL_ID,
      })) as Extract<WorkerResponse, { kind: 'boot:ok' }>;

      startTransition(() => {
        setBootStats(response.stats);
        setBooting(false);
      });
    } catch (error) {
      setBooting(false);
      setErrorText(error instanceof Error ? error.message : String(error));
    }
  }

  async function loadSemanticLandscape(signal: AbortSignal): Promise<void> {
    setLandscapeLoading(true);
    setLandscapeErrorText(null);

    try {
      const response = await fetch(landscapeUrl, { cache: 'force-cache', signal });
      if (!response.ok) {
        throw new Error(`Failed to download ${landscapeUrl} (${response.status}).`);
      }

      const payload = (await response.json()) as SemanticLandscapeData;
      if (signal.aborted) {
        return;
      }

      startTransition(() => {
        setLandscapeData(payload);
        setLandscapeLoading(false);
      });
    } catch (error) {
      if (signal.aborted) {
        return;
      }

      setLandscapeLoading(false);
      setLandscapeErrorText(error instanceof Error ? error.message : String(error));
    }
  }

  async function runSearch(
    nextQuery = query,
    nextTopK = topK,
    nextMinLength = minLength,
    nextMinScore = minScore,
  ): Promise<void> {
    const trimmed = nextQuery.trim();

    if (!trimmed || !bootStats) {
      setHasSearched(false);
      setResults([]);
      setSelectedEntryId(null);
      setContextItems([]);
      setSearchNote(null);
      return;
    }

    const sequence = latestSearchRef.current + 1;
    latestSearchRef.current = sequence;

    setSearching(true);
    setErrorText(null);
    setSearchNote(null);
    setHasSearched(true);

    try {
      const response = (await callWorker({
        kind: 'search',
        query: trimmed,
        mode: 'semantic',
        language: 'en',
        topK: nextTopK,
        minLength: nextMinLength,
        minScore: nextMinScore,
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

  function handleSubmit(event: FormEvent<HTMLFormElement>): void {
    event.preventDefault();
    void runSearch();
  }

  function handleTopKChange(value: string): void {
    const nextTopK = Math.max(1, Number(value) || 1);
    setTopK(nextTopK);

    if (hasSearched && query.trim()) {
      void runSearch(query, nextTopK, minLength, minScore);
    }
  }

  function handleMinLengthChange(value: string): void {
    const nextMinLength = Math.max(0, Number(value) || 0);
    setMinLength(nextMinLength);

    if (hasSearched && query.trim()) {
      void runSearch(query, topK, nextMinLength, minScore);
    }
  }

  function handleMinScoreChange(value: string): void {
    const nextMinScore = Math.max(0, Number(value) || 0);
    setMinScore(nextMinScore);

    if (hasSearched && query.trim()) {
      void runSearch(query, topK, minLength, nextMinScore);
    }
  }

  function handleContextRadiusChange(value: string): void {
    const nextRadius = Math.max(0, Number(value) || 0);
    setContextRadius(nextRadius);
  }

  function handleResultCardKeyDown(
    event: ReactKeyboardEvent<HTMLElement>,
    entryId: string,
  ): void {
    if (event.target !== event.currentTarget) {
      return;
    }

    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      setSelectedEntryId(entryId);
    }
  }

  async function openTranscript(videoId: string, focusEntryId: string): Promise<void> {
    const sequence = latestTranscriptRef.current + 1;
    latestTranscriptRef.current = sequence;

    setSelectedEntryId(focusEntryId);
    setTranscriptVideoId(videoId);
    setTranscriptItems([]);
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
    setTranscriptLoading(false);
    setTranscriptErrorText(null);
  }

  function goHome(): void {
    latestSearchRef.current += 1;
    latestContextRef.current += 1;
    closeTranscript();
    setQuery('');
    setTopK(10);
    setMinLength(0);
    setMinScore(0);
    setContextRadius(DEFAULT_CONTEXT_RADIUS);
    setHasSearched(false);
    setSearching(false);
    setErrorText(null);
    setSearchNote(null);
    setResults([]);
    setSelectedEntryId(null);
    setContextItems([]);
    window.scrollTo({ top: 0, behavior: 'smooth' });
  }

  function toggleTheme(): void {
    setTheme((currentTheme) => (currentTheme === 'dark' ? 'light' : 'dark'));
  }

  const canSearch = !booting && !!query.trim();

  return (
    <main className="app-shell">
      <section className="hero">
        <h1 className="page-title">
          <button className="title-home" type="button" onClick={goHome}>
            Translation Memory
          </button>
        </h1>
        <div className="hero-meta">
          <span className="hero-chip">
            {bootStats ? (
              <>
                <strong className="hero-chip-value">{bootStats.totalEntries.toLocaleString()}</strong>
                <span className="hero-chip-label">English/中文 Pairs</span>
              </>
            ) : (
              <span className="hero-chip-label">Loading English/中文 Pairs</span>
            )}
          </span>
          <button
            aria-label={`Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`}
            className={theme === 'dark' ? 'theme-toggle is-dark' : 'theme-toggle is-light'}
            type="button"
            onClick={toggleTheme}
          >
            <span
              aria-hidden="true"
              className={theme === 'dark' ? 'theme-option is-active' : 'theme-option'}
            >
              ☾
            </span>
            <span
              aria-hidden="true"
              className={theme === 'light' ? 'theme-option is-active' : 'theme-option'}
            >
              ☼
            </span>
          </button>
        </div>
      </section>

      <section className="panel controls-panel">
        <form className="search-form" onSubmit={handleSubmit}>
          <div className="field query-field">
            <input
              aria-label="Search English subtitle lines"
              autoFocus
              type="text"
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              placeholder="Search English subtitle lines"
            />
          </div>

          <button className="search-button" type="submit" disabled={!canSearch || searching}>
            {searching ? 'Searching…' : 'Search'}
          </button>
        </form>
      </section>

      {errorText ? (
        <section className="panel message error-message">
          <strong>Error</strong>
          <p>{errorText}</p>
        </section>
      ) : null}

      {searchNote ? (
        <section className="panel message note-message">
          <strong>Note</strong>
          <p>{searchNote}</p>
        </section>
      ) : null}

      {!hasSearched ? (
        <>
          <section className="startup-grid" aria-label="Dataset overview">
            <section className="panel startup-panel">
              <div className="panel-header startup-header">
                <div className="results-heading">
                  <h2>Rows Per Video</h2>
                  <span>How densely the current dataset is distributed across videos.</span>
                </div>
              </div>

              {bootStats ? (
                <>
                  <div className="startup-metrics">
                    <article className="startup-metric-card">
                      <span className="startup-metric-label">Average</span>
                      <strong className="startup-metric-value">
                        {formatStat(bootStats.avgRowsPerVideo)}
                      </strong>
                      <span className="startup-metric-note">rows per video</span>
                    </article>

                    <article className="startup-metric-card">
                      <span className="startup-metric-label">Median</span>
                      <strong className="startup-metric-value">
                        {formatStat(bootStats.medianRowsPerVideo)}
                      </strong>
                      <span className="startup-metric-note">rows per video</span>
                    </article>

                    <article className="startup-metric-card">
                      <span className="startup-metric-label">Max</span>
                      <strong className="startup-metric-value">
                        {formatStat(bootStats.maxRowsPerVideo)}
                      </strong>
                      <span className="startup-metric-note">rows in one video</span>
                    </article>
                  </div>

                  <p className="startup-panel-note">
                    Based on {bootStats.videoCount.toLocaleString()} videos in the current SQLite
                    snapshot.
                  </p>
                </>
              ) : (
                <div className="empty-state">
                  <p>Loading video distribution…</p>
                </div>
              )}
            </section>

            <section className="panel startup-panel">
              <div className="panel-header startup-header">
                <div className="results-heading">
                  <h2>Line Length Medians</h2>
                  <span>Median character counts for English and Chinese subtitle lines.</span>
                </div>
              </div>

              {bootStats ? (
                <>
                  <div className="startup-metrics startup-metrics--two-up">
                    <article className="startup-metric-card">
                      <span className="startup-metric-label">English</span>
                      <strong className="startup-metric-value">
                        {formatStat(bootStats.medianEnLength)}
                      </strong>
                      <span className="startup-metric-note">median characters</span>
                    </article>

                    <article className="startup-metric-card">
                      <span className="startup-metric-label">Chinese</span>
                      <strong className="startup-metric-value">
                        {formatStat(bootStats.medianZhLength)}
                      </strong>
                      <span className="startup-metric-note">median characters</span>
                    </article>
                  </div>

                  <p className="startup-panel-note">
                    Useful for setting expectations around scan density and minimum-length filters.
                  </p>
                </>
              ) : (
                <div className="empty-state">
                  <p>Loading line length summary…</p>
                </div>
              )}
            </section>
          </section>

          {landscapeErrorText ? (
            <section className="panel message error-message">
              <strong>Semantic Landscape</strong>
              <p>{landscapeErrorText}</p>
            </section>
          ) : landscapeLoading ? (
            <section className="panel startup-panel semantic-panel semantic-panel--loading">
              <div className="panel-header semantic-panel-header">
                <div className="results-heading">
                  <h2>Semantic Landscape</h2>
                  <span>Preparing the precomputed startup distribution…</span>
                </div>
              </div>
              <div className="empty-state">
                <p>Loading all-entry semantic coordinates…</p>
              </div>
            </section>
          ) : landscapeData ? (
            <SemanticLandscapePanel
              data={landscapeData}
              theme={theme}
              onOpenTranscript={(videoId, focusEntryId) => {
                void openTranscript(videoId, focusEntryId);
              }}
            />
          ) : null}
        </>
      ) : null}

      {hasSearched ? (
        <section className="workspace">
          <div className="panel results-panel">
            <div className="panel-header results-header">
              <div className="results-heading">
                <h2>Results</h2>
              </div>

              <div className="results-tools">
                <label className="field compact-inline-field">
                  <span>Top K</span>
                  <input
                    type="number"
                    min={1}
                    max={100}
                    value={topK}
                    onChange={(event) => handleTopKChange(event.target.value)}
                  />
                </label>

                <label className="field compact-inline-field">
                  <span>Min Chars</span>
                  <input
                    type="number"
                    min={0}
                    max={500}
                    value={minLength}
                    onChange={(event) => handleMinLengthChange(event.target.value)}
                  />
                </label>

                <label className="field compact-inline-field">
                  <span>Score</span>
                  <input
                    type="number"
                    min={0}
                    max={2}
                    step={0.05}
                    value={minScore}
                    onChange={(event) => handleMinScoreChange(event.target.value)}
                  />
                </label>
              </div>
            </div>

            {results.length === 0 ? (
              <div className="empty-state">
                <p>No matches found for this query.</p>
              </div>
            ) : (
              <ol className="results-list">
                {results.map((result) => (
                  <li key={result.entryId}>
                    <article
                      className={
                        result.entryId === selectedEntryId ? 'result-card is-active' : 'result-card'
                      }
                      role="button"
                      tabIndex={0}
                      onClick={() => setSelectedEntryId(result.entryId)}
                      onKeyDown={(event) => handleResultCardKeyDown(event, result.entryId)}
                    >
                      <div className="result-header">
                        <span className="result-metric">
                          <span className="result-metric-label">YouTube ID</span>
                          <button
                            className="video-id-button"
                            type="button"
                            onClick={(event) => {
                              event.stopPropagation();
                              void openTranscript(result.videoId, result.entryId);
                            }}
                          >
                            {result.videoId}
                          </button>
                        </span>
                        <span className="result-metric">
                          <span className="result-metric-label">Entry</span>
                          <strong>#{result.segIndex}</strong>
                        </span>
                        <span className="result-metric result-score">
                          <span className="result-metric-label">Score</span>
                          <strong>{result.score.toFixed(4)}</strong>
                        </span>
                      </div>
                      <div className="result-copy-group">
                        <div className="result-copy">
                          <div className="result-copy-line">
                            <p className="result-en">{result.en}</p>
                            <span className="result-char-count">{result.textLength} chars</span>
                          </div>
                        </div>
                        <div className="result-copy">
                          <p className="result-zh">{result.zh}</p>
                        </div>
                      </div>
                      <p className="result-block">Block ID {result.blockName || 'No block'}</p>
                    </article>
                  </li>
                ))}
              </ol>
            )}
          </div>

          <div className="detail-column">
            <section className="panel context-panel">
              <div className="panel-header">
                <h2>Context</h2>
                <label className="field compact-inline-field">
                  <span>Radius</span>
                  <input
                    type="number"
                    min={0}
                    max={20}
                    value={contextRadius}
                    onChange={(event) => handleContextRadiusChange(event.target.value)}
                  />
                </label>
              </div>

              {contextItems.length === 0 ? (
                <div className="empty-state">
                  <p>Context appears here after you select a result.</p>
                </div>
              ) : (
                <ol className="context-list">
                  {contextItems.map((item) => (
                    <li
                      key={item.entryId}
                      className={item.isFocus ? 'context-item is-focus' : 'context-item'}
                    >
                      <div className="context-meta">
                        <span>{item.videoId}#{item.segIndex}</span>
                        <span>{item.blockName || 'no block'}</span>
                      </div>
                      <p>{item.en}</p>
                      <p>{item.zh}</p>
                    </li>
                  ))}
                </ol>
              )}
            </section>
          </div>
        </section>
      ) : null}

      {transcriptVideoId ? (
        <div className="modal-backdrop" role="presentation" onClick={closeTranscript}>
          <section
            aria-labelledby="transcript-dialog-title"
            aria-modal="true"
            className="panel transcript-modal"
            role="dialog"
            onClick={(event) => event.stopPropagation()}
          >
            <div className="transcript-modal-header">
              <div className="transcript-heading">
                <h2 id="transcript-dialog-title">Video Transcript</h2>
                <span>{transcriptVideoId}</span>
              </div>

              <button className="modal-close" type="button" onClick={closeTranscript}>
                Close
              </button>
            </div>

            {transcriptLoading ? (
              <div className="empty-state">
                <p>Loading the full transcript…</p>
              </div>
            ) : transcriptErrorText ? (
              <div className="empty-state">
                <p>{transcriptErrorText}</p>
              </div>
            ) : (
              <div className="transcript-modal-body">
                <ol className="context-list transcript-list">
                  {transcriptItems.map((item) => (
                    <li
                      key={item.entryId}
                      className={item.isFocus ? 'context-item is-focus' : 'context-item'}
                    >
                      <div className="context-meta">
                        <span>{item.videoId}#{item.segIndex}</span>
                        <span>{item.blockName || 'no block'}</span>
                      </div>
                      <p>{item.en}</p>
                      <p>{item.zh}</p>
                    </li>
                  ))}
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
