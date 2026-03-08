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

export interface CueTimeDistribution {
  binCount: number;
  bins: number[];
  timedEntryCount: number;
  totalEntryCount: number;
  timedVideoCount: number;
  totalVideoCount: number;
  averageCueDurationMs: number;
  medianVideoSpanMs: number;
  averageCoverage: number;
  peakCoverage: number;
  peakRangeStart: number;
  peakRangeEnd: number;
}

export interface BootStats {
  totalEntries: number;
  cueTimeDistribution: CueTimeDistribution | null;
}

export interface BootRequest {
  kind: 'boot';
  requestId: number;
  dbUrl: string;
}

export interface SearchRequest {
  kind: 'search';
  requestId: number;
  query: string;
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
  | BootOkResponse
  | SearchOkResponse
  | ContextOkResponse
  | TranscriptOkResponse
  | ErrorResponse;
