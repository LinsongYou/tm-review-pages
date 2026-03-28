export interface SemanticLandscapeSample {
  entryId: string;
  videoId: string;
  segIndex: number;
  en: string;
  zh: string;
}

export type SemanticLandscapeLabelMode = 'theme' | 'provisional' | 'descriptive';

export interface SemanticLandscapeCluster {
  id: number;
  label: string;
  labelMode: SemanticLandscapeLabelMode;
  labelConfidence: number;
  color: string;
  size: number;
  videoCount: number;
  x: number;
  y: number;
  keywords: string[];
  topPhrases: string[];
  medoidEntryId: string;
  representativeEntryIds: string[];
  samples: SemanticLandscapeSample[];
}

export interface SemanticLandscapePoint {
  entryId: string;
  videoId: string;
  segIndex: number;
  en: string;
  zh: string;
  x: number;
  y: number;
  clusterId: number;
}

export interface VideoFingerprintVideo {
  videoId: string;
  firstEntryId: string;
  entryCount: number;
  timedEntryCount: number;
  dominantClusterId: number;
  dominantShare: number;
  clusterCounts: number[];
  bins: number[];
  densities: number[];
  x: number;
  y: number;
}

export interface VideoFingerprintWall {
  binCount: number;
  sort: string;
  videos: VideoFingerprintVideo[];
}

export interface SemanticLandscapeData {
  version: number;
  projection: string;
  clusterAlgorithm: string;
  generatedAt: string;
  sourceDb: string;
  modelId: string;
  pointCount: number;
  vectorDim: number;
  clusters: SemanticLandscapeCluster[];
  points: SemanticLandscapePoint[];
  videoFingerprintWall: VideoFingerprintWall;
}
