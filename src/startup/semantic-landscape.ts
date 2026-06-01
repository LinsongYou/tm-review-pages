export interface SemanticLandscapeCluster {
  id: number;
  label: string;
  description: string;
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
  compactness: number;
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
  color: string;
  clusterId: number;
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
  clusters: SemanticLandscapeCluster[];
  points: SemanticLandscapePoint[];
}
