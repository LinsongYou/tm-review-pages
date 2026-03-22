export interface SemanticLandscapeSample {
  entryId: string;
  videoId: string;
  segIndex: number;
  en: string;
  zh: string;
}

export interface SemanticLandscapeCluster {
  id: number;
  label: string;
  color: string;
  size: number;
  x: number;
  y: number;
  keywords: string[];
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
