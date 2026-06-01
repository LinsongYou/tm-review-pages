export interface EntrySummary {
  entryId: string;
  videoId: string;
  segIndex: number;
  en: string;
  zh: string;
  blockName: string;
  startMs: number | null;
  endMs: number | null;
}

export interface SearchResult extends EntrySummary {
  score: number;
  textLength: number;
}

export interface ContextItem extends EntrySummary {
  isFocus: boolean;
}

export type BootProgressTarget = 'pairs' | 'model';

export interface BootProgressSnapshot {
  target: BootProgressTarget;
  name: string;
  progress: number;
  statusText: string;
  detail?: string;
}

export interface BootStats {
  totalEntries: number;
  embeddingModelId: string;
}

export interface BootRequest {
  kind: 'boot';
  requestId: number;
  dbUrl: string;
}

export interface PrepareModelRequest {
  kind: 'prepare-model';
  requestId: number;
}

export interface SearchRequest {
  kind: 'search';
  requestId: number;
  query: string;
  topK: number;
  minLength: number;
  minScore: number;
}

export interface TranscriptRequest {
  kind: 'transcript';
  requestId: number;
  videoId: string;
  focusEntryId?: string;
}

export type WorkerRequest = BootRequest | PrepareModelRequest | SearchRequest | TranscriptRequest;
export type WorkerPayload =
  | Omit<BootRequest, 'requestId'>
  | Omit<PrepareModelRequest, 'requestId'>
  | Omit<SearchRequest, 'requestId'>
  | Omit<TranscriptRequest, 'requestId'>;

export interface BootOkResponse {
  kind: 'boot:ok';
  requestId: number;
  stats: BootStats;
}

export interface BootProgressResponse {
  kind: 'boot:progress';
  requestId: number;
  progress: BootProgressSnapshot;
}

export interface PrepareModelOkResponse {
  kind: 'prepare-model:ok';
  requestId: number;
}

export interface SearchOkResponse {
  kind: 'search:ok';
  requestId: number;
  results: SearchResult[];
  note?: string;
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
  | BootOkResponse
  | BootProgressResponse
  | PrepareModelOkResponse
  | SearchOkResponse
  | TranscriptOkResponse
  | ErrorResponse;
