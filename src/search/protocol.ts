export type SearchMode = 'lexical' | 'semantic';
export type SearchLanguage = 'en' | 'zh';
export type ModelStatus = 'idle' | 'loading' | 'ready' | 'error';

export interface EntrySummary {
  entryId: string;
  videoId: string;
  segIndex: number;
  en: string;
  zh: string;
  blockName: string;
  layer: number;
  updatedAt: string;
  hasVector: boolean;
}

export interface SearchResult extends EntrySummary {
  score: number;
  textLength: number;
}

export interface ContextItem extends EntrySummary {
  isFocus: boolean;
}

export interface BootStats {
  dbUrl: string;
  dbSizeBytes: number;
  totalEntries: number;
  vectorEntries: number;
  vectorCoverage: number;
  vectorDim: number;
  vectorModelId: string;
  queryModelId: string;
  loadMs: number;
  semanticLanguageSupport: 'en-only';
}

export interface BootRequest {
  kind: 'boot';
  requestId: number;
  dbUrl: string;
  vectorModelId: string;
  queryModelId: string;
}

export interface SearchRequest {
  kind: 'search';
  requestId: number;
  query: string;
  mode: SearchMode;
  language: SearchLanguage;
  topK: number;
  minLength: number;
  minScore: number;
}

export interface ContextRequest {
  kind: 'context';
  requestId: number;
  entryId: string;
  radius?: number;
}

export interface TranscriptRequest {
  kind: 'transcript';
  requestId: number;
  videoId: string;
  focusEntryId?: string;
}

export type WorkerRequest = BootRequest | SearchRequest | ContextRequest | TranscriptRequest;
export type WorkerPayload =
  | Omit<BootRequest, 'requestId'>
  | Omit<SearchRequest, 'requestId'>
  | Omit<ContextRequest, 'requestId'>
  | Omit<TranscriptRequest, 'requestId'>;

export interface StatusResponse {
  kind: 'status';
  requestId: number;
  scope: 'boot' | 'model';
  message: string;
  modelStatus?: ModelStatus;
}

export interface BootOkResponse {
  kind: 'boot:ok';
  requestId: number;
  stats: BootStats;
}

export interface SearchOkResponse {
  kind: 'search:ok';
  requestId: number;
  results: SearchResult[];
  note?: string;
}

export interface ContextOkResponse {
  kind: 'context:ok';
  requestId: number;
  context: ContextItem[];
}

export interface TranscriptOkResponse {
  kind: 'transcript:ok';
  requestId: number;
  videoId: string;
  items: ContextItem[];
}

export interface ErrorResponse {
  kind: 'error';
  requestId: number;
  message: string;
}

export type WorkerResponse =
  | StatusResponse
  | BootOkResponse
  | SearchOkResponse
  | ContextOkResponse
  | TranscriptOkResponse
  | ErrorResponse;
