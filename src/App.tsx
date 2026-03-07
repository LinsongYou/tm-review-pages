import { FormEvent, startTransition, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react';
import type {
  BootStats,
  ContextItem,
  SearchResult,
  WorkerPayload,
  WorkerResponse,
} from './search/protocol';

type PendingRequest = {
  resolve: (value: WorkerResponse) => void;
  reject: (reason?: unknown) => void;
};

type Theme = 'dark' | 'light';

const VECTOR_MODEL_ID = 'sentence-transformers/all-MiniLM-L6-v2';
const QUERY_MODEL_ID = 'Xenova/all-MiniLM-L6-v2';
const DB_ASSET = 'data/tm_misha_minilm.db';
const THEME_STORAGE_KEY = 'tm-review-theme';

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

function App() {
  const workerRef = useRef<Worker | null>(null);
  const pendingRef = useRef(new Map<number, PendingRequest>());
  const requestIdRef = useRef(1);
  const latestSearchRef = useRef(0);
  const latestContextRef = useRef(0);

  const [bootStats, setBootStats] = useState<BootStats | null>(null);
  const [booting, setBooting] = useState(true);
  const [searching, setSearching] = useState(false);
  const [errorText, setErrorText] = useState<string | null>(null);
  const [searchNote, setSearchNote] = useState<string | null>(null);
  const [query, setQuery] = useState('');
  const [topK, setTopK] = useState(10);
  const [minLength, setMinLength] = useState(0);
  const [theme, setTheme] = useState<Theme>(getInitialTheme);
  const [hasSearched, setHasSearched] = useState(false);
  const [results, setResults] = useState<SearchResult[]>([]);
  const [selectedEntryId, setSelectedEntryId] = useState<string | null>(null);
  const [contextItems, setContextItems] = useState<ContextItem[]>([]);

  const dbUrl = useMemo(() => `${import.meta.env.BASE_URL}${DB_ASSET}`, []);

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
          radius: 2,
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

  async function runSearch(
    nextQuery = query,
    nextTopK = topK,
    nextMinLength = minLength,
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
      void runSearch(query, nextTopK, minLength);
    }
  }

  function handleMinLengthChange(value: string): void {
    const nextMinLength = Math.max(0, Number(value) || 0);
    setMinLength(nextMinLength);

    if (hasSearched && query.trim()) {
      void runSearch(query, topK, nextMinLength);
    }
  }

  function toggleTheme(): void {
    setTheme((currentTheme) => (currentTheme === 'dark' ? 'light' : 'dark'));
  }

  const canSearch = !booting && !!query.trim();
  const selectedResult = selectedEntryId
    ? results.find((result) => result.entryId === selectedEntryId) ?? null
    : null;

  return (
    <main className="app-shell">
      <section className="hero">
        <p className="eyebrow">Translation Memory Review</p>
        <div className="hero-meta">
          <span className="hero-chip">
            {bootStats
              ? `${bootStats.totalEntries.toLocaleString()} EN/中文 pairs`
              : 'Loading EN/中文 pairs'}
          </span>
          <span className="hero-chip">{QUERY_MODEL_ID}</span>
          <button
            aria-label={`Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`}
            className="theme-toggle"
            type="button"
            onClick={toggleTheme}
          >
            <span
              aria-hidden="true"
              className={theme === 'dark' ? 'theme-option is-active' : 'theme-option'}
            >
              🌙
            </span>
            <span
              aria-hidden="true"
              className={theme === 'light' ? 'theme-option is-active' : 'theme-option'}
            >
              ☀️
            </span>
          </button>
        </div>
      </section>

      <section className="panel controls-panel">
        <form className="search-form" onSubmit={handleSubmit}>
          <label className="field query-field">
            <span>Search</span>
            <input
              autoFocus
              type="text"
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              placeholder="Search English subtitle lines"
            />
          </label>

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

      {hasSearched ? (
        <section className="workspace">
          <div className="panel results-panel">
            <div className="panel-header results-header">
              <div className="results-heading">
                <h2>Results</h2>
                <span>{results.length.toLocaleString()} shown</span>
              </div>

              <div className="results-tools">
                <label className="field compact-field">
                  <span>Top K</span>
                  <input
                    type="number"
                    min={1}
                    max={100}
                    value={topK}
                    onChange={(event) => handleTopKChange(event.target.value)}
                  />
                </label>

                <label className="field compact-field">
                  <span>Min Chars</span>
                  <input
                    type="number"
                    min={0}
                    max={500}
                    value={minLength}
                    onChange={(event) => handleMinLengthChange(event.target.value)}
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
                    <button
                      className={
                        result.entryId === selectedEntryId ? 'result-card is-active' : 'result-card'
                      }
                      type="button"
                      onClick={() => setSelectedEntryId(result.entryId)}
                    >
                      <div className="result-meta">
                        <span>{result.videoId}#{result.segIndex}</span>
                        <span>{result.score.toFixed(4)}</span>
                      </div>
                      <div className="result-flags">
                        <span>{result.textLength} chars</span>
                        <span>{result.hasVector ? 'vector' : 'text-only'}</span>
                      </div>
                      <p className="result-en">{result.en}</p>
                      <p className="result-zh">{result.zh}</p>
                    </button>
                  </li>
                ))}
              </ol>
            )}
          </div>

          <div className="detail-column">
            <section className="panel detail-panel">
              <div className="panel-header">
                <h2>Selection</h2>
                <span>{selectedResult?.entryId ?? 'None'}</span>
              </div>

              {selectedResult ? (
                <div className="detail-body">
                  <div className="detail-grid">
                    <div>
                      <span className="detail-label">Score</span>
                      <strong>{selectedResult.score.toFixed(4)}</strong>
                    </div>
                    <div>
                      <span className="detail-label">Block</span>
                      <strong>{selectedResult.blockName || '-'}</strong>
                    </div>
                    <div>
                      <span className="detail-label">Layer</span>
                      <strong>{selectedResult.layer}</strong>
                    </div>
                    <div>
                      <span className="detail-label">Updated</span>
                      <strong>{selectedResult.updatedAt || '-'}</strong>
                    </div>
                  </div>

                  <div className="detail-copy">
                    <h3>English</h3>
                    <p>{selectedResult.en}</p>
                  </div>

                  <div className="detail-copy">
                    <h3>Chinese</h3>
                    <p>{selectedResult.zh}</p>
                  </div>
                </div>
              ) : (
                <div className="empty-state">
                  <p>Select a result to inspect its metadata and nearby context.</p>
                </div>
              )}
            </section>

            <section className="panel context-panel">
              <div className="panel-header">
                <h2>Context</h2>
                <span>±2 rows in the same video</span>
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
    </main>
  );
}

export default App;
