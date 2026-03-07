import { FormEvent, startTransition, useEffect, useMemo, useRef, useState } from 'react';
import type {
  BootStats,
  ContextItem,
  ModelStatus,
  SearchLanguage,
  SearchResult,
  WorkerPayload,
  WorkerResponse,
} from './search/protocol';

type PendingRequest = {
  resolve: (value: WorkerResponse) => void;
  reject: (reason?: unknown) => void;
};

const SEMANTIC_MODEL_ID = 'sentence-transformers/all-MiniLM-L6-v2';
const DB_ASSET = 'data/tm_misha_minilm.db';

function formatBytes(bytes: number): string {
  if (bytes === 0) {
    return '0 B';
  }

  const units = ['B', 'KB', 'MB', 'GB'];
  let value = bytes;
  let unitIndex = 0;

  while (value >= 1024 && unitIndex < units.length - 1) {
    value /= 1024;
    unitIndex += 1;
  }

  return `${value.toFixed(unitIndex === 0 ? 0 : 1)} ${units[unitIndex]}`;
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
  const [statusText, setStatusText] = useState('Starting worker.');
  const [modelStatus, setModelStatus] = useState<ModelStatus>('idle');
  const [errorText, setErrorText] = useState<string | null>(null);
  const [searchNote, setSearchNote] = useState<string | null>(null);
  const [query, setQuery] = useState('');
  const [language, setLanguage] = useState<SearchLanguage>('en');
  const [topK, setTopK] = useState(10);
  const [minLength, setMinLength] = useState(0);
  const [results, setResults] = useState<SearchResult[]>([]);
  const [selectedEntryId, setSelectedEntryId] = useState<string | null>(null);
  const [contextItems, setContextItems] = useState<ContextItem[]>([]);

  const dbUrl = useMemo(() => `${import.meta.env.BASE_URL}${DB_ASSET}`, []);

  useEffect(() => {
    const worker = new Worker(new URL('./search/search.worker.ts', import.meta.url), {
      type: 'module',
    });
    workerRef.current = worker;

    worker.addEventListener('message', (event: MessageEvent<WorkerResponse>) => {
      const message = event.data;

      if (message.kind === 'status') {
        setStatusText(message.message);
        if (message.modelStatus) {
          setModelStatus(message.modelStatus);
        }
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
        modelId: SEMANTIC_MODEL_ID,
      })) as Extract<WorkerResponse, { kind: 'boot:ok' }>;

      startTransition(() => {
        setBootStats(response.stats);
        setBooting(false);
        setStatusText('TM database ready.');
      });
    } catch (error) {
      setBooting(false);
      setErrorText(error instanceof Error ? error.message : String(error));
    }
  }

  async function runSearch(): Promise<void> {
    const trimmed = query.trim();

    if (!trimmed || !bootStats) {
      setResults([]);
      setSelectedEntryId(null);
      setSearchNote(null);
      return;
    }

    const sequence = latestSearchRef.current + 1;
    latestSearchRef.current = sequence;

    setSearching(true);
    setErrorText(null);
    setSearchNote(null);

    try {
      const mode = language === 'en' ? 'semantic' : 'lexical';
      const response = (await callWorker({
        kind: 'search',
        query: trimmed,
        mode,
        language,
        topK,
        minLength,
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

  const canSearch = !booting && !!query.trim();
  const selectedResult = selectedEntryId
    ? results.find((result) => result.entryId === selectedEntryId) ?? null
    : null;

  return (
    <main className="app-shell">
      <section className="hero">
        <div className="hero-copy">
          <p className="eyebrow">Translation Memory Review</p>
        </div>
        <div className="hero-status">
          <div className="status-card">
            <span>Semantic Model</span>
            <strong>{modelStatus}</strong>
            <small>{booting ? statusText : SEMANTIC_MODEL_ID}</small>
          </div>
          <div className="status-card">
            <span>Corpus</span>
            <strong>
              {bootStats ? `${bootStats.totalEntries.toLocaleString()} rows` : 'Loading'}
            </strong>
            <small>
              {bootStats ? `${formatBytes(bootStats.dbSizeBytes)} static SQLite asset` : statusText}
            </small>
          </div>
        </div>
      </section>

      <section className="panel controls-panel">
        <form className="search-form" onSubmit={handleSubmit}>
          <label className="field query-field">
            <span>Query</span>
            <input
              autoFocus
              type="text"
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              placeholder="Search English or Chinese TM text"
            />
          </label>

          <div className="control-row">
            <label className="field">
              <span>Language</span>
              <select
                value={language}
                onChange={(event) => setLanguage(event.target.value as SearchLanguage)}
              >
                <option value="en">English</option>
                <option value="zh">Chinese</option>
              </select>
            </label>

            <label className="field small-field">
              <span>Top K</span>
              <input
                type="number"
                min={1}
                max={100}
                value={topK}
                onChange={(event) => setTopK(Math.max(1, Number(event.target.value) || 1))}
              />
            </label>

            <label className="field small-field">
              <span>Min Length</span>
              <input
                type="number"
                min={0}
                max={500}
                value={minLength}
                onChange={(event) => setMinLength(Math.max(0, Number(event.target.value) || 0))}
              />
            </label>

            <button className="search-button" type="submit" disabled={!canSearch || searching}>
              {searching ? 'Searching…' : 'Search'}
            </button>
          </div>
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

      <section className="workspace">
        <div className="panel results-panel">
          <div className="panel-header">
            <h2>Results</h2>
            <span>{results.length.toLocaleString()} shown</span>
          </div>

          {results.length === 0 ? (
            <div className="empty-state">
              <p>Run a search to inspect TM rows, scores, and local context.</p>
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
    </main>
  );
}

export default App;
