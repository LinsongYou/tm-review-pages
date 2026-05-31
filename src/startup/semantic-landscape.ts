export interface SemanticLandscapeSample {
  entryId: string;
  videoId: string;
  segIndex: number;
  en: string;
  zh: string;
  blockName?: string;
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
  z: number;
  x3d: number;
  y3d: number;
  z3d: number;
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
  blockName: string;
  startMs: number | null;
  endMs: number | null;
  x: number;
  y: number;
  z: number;
  x3d: number;
  y3d: number;
  z3d: number;
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

export interface SemanticLandscapeClusterSelection {
  minCount: number;
  maxCount: number;
  sampleSize: number;
  finalistCount: number;
  selectedCount: number;
  score: number;
  cohesion: number;
  margin: number;
  centerSeparation: number;
  maxNeighborSimilarity: number;
  minClusterShare: number;
  interpretabilityScore: number;
  themeShare: number;
  descriptiveShare: number;
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
  clusterCount?: number;
  umap?: {
    neighbors: number;
    minDist: number;
    spread: number;
    epochs: number;
    randomSeed: number;
  };
  clusterSelection?: SemanticLandscapeClusterSelection;
  clusters: SemanticLandscapeCluster[];
  points: SemanticLandscapePoint[];
  videoFingerprintWall?: VideoFingerprintWall;
}
