export interface SemanticLandscapeCluster {
  id: number;
  label: string;
  description: string;
  color: string;
  size: number;
  videoCount: number;
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
  startMs: number | null;
  endMs: number | null;
  x3d: number;
  y3d: number;
  z3d: number;
  color: string;
  clusterId: number;
}

export interface InitialView {
  rotateX: number;
  rotateY: number;
  zoom: number;
  offsetX: number;
  offsetY: number;
}

export interface SemanticLandscapeData {
  pointCount: number;
  initialView?: InitialView;
  clusters: SemanticLandscapeCluster[];
  points: SemanticLandscapePoint[];
}
