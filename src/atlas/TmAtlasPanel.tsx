import { type CSSProperties, type FormEvent, type ReactNode, useEffect, useMemo, useRef, useState } from 'react';
import { classNames } from '../classes';
import type { EntrySummary, SearchResult } from '../search/protocol';
import type {
  SemanticLandscapeCluster,
  SemanticLandscapeData,
  SemanticLandscapePoint,
} from './semantic-landscape';
import { hexToRgba, blendHexColors, isLightHex } from './colors';

type ThemeMode = 'dark' | 'light';
export type AtlasViewMode = 'atlas' | 'text';

interface StatusItem {
  label: string;
  value: string;
  detail: string;
  progress: number;
  ready: boolean;
}

interface TmAtlasPanelProps {
  data: SemanticLandscapeData | null;
  dataLoading: boolean;
  dataErrorText: string | null;
  theme: ThemeMode;
  viewMode: AtlasViewMode;
  statusItems: StatusItem[];
  navigation: AtlasNavigationState;
  searchReady: boolean;
  searching: boolean;
  onQueryChange: (query: string) => void;
  onSearch: () => void;
  onSelectEntry: (entryId: string | null) => void;
  onOpenTranscript: (videoId: string, focusEntryId: string) => void;
  onSelectTranscriptEntry: (entryId: string) => void;
  onSearchLine: (line: string) => void;
  onCloseTranscript: () => void;
  onRestoreNavigationState: (state: AtlasNavigationState) => void;
  onClear: () => void;
  onToggleTheme: () => void;
  onToggleViewMode: () => void;
}

export interface AtlasNavigationState {
  query: string;
  searchResults: SearchResult[];
  searchNote: string | null;
  errorText: string | null;
  selectedEntryId: string | null;
  transcriptVideoId: string | null;
  transcriptItems: EntrySummary[];
  transcriptFocusEntryId: string | null;
  transcriptLoading: boolean;
  transcriptErrorText: string | null;
}

interface View3d {
  rotateX: number;
  rotateY: number;
  zoom: number;
  offsetX: number;
  offsetY: number;
}

interface VisualGeometry {
  centerX: number;
  centerY: number;
  centerZ: number;
  radius: number;
}

interface VisualFocus {
  key: string;
  centerX: number;
  centerY: number;
  centerZ: number;
  zoom: number;
}

interface ProjectedPoint {
  point: SemanticLandscapePoint;
  rawX: number;
  rawY: number;
  x: number;
  y: number;
  depth: number;
  culled: boolean;
}

interface ProjectedIsland {
  cluster: SemanticLandscapeCluster;
  x: number;
  y: number;
  depth: number;
}

interface ProjectedFrame {
  points: ProjectedPoint[];
  pointByEntryId: Map<string, ProjectedPoint>;
  islandById: Map<number, ProjectedIsland>;
}

interface ProjectionCache {
  sourcePoints: SemanticLandscapePoint[];
  clusterById: Map<number, SemanticLandscapeCluster>;
  projectedPoints: ProjectedPoint[];
  sortedPoints: ProjectedPoint[];
  radiusScratch: number[];
  pointByEntryId: Map<string, ProjectedPoint>;
  islandTotals: Map<number, { x: number; y: number; depth: number; count: number }>;
  islandById: Map<number, ProjectedIsland>;
  frame: ProjectedFrame;
}

interface IslandFlow {
  fromClusterId: number;
  toClusterId: number;
  count: number;
}

interface HoverState {
  entryId: string;
  x: number;
  y: number;
}

interface IslandPanelData {
  cluster: SemanticLandscapeCluster;
  entries: SemanticLandscapePoint[];
  share: number;
  zoom: number;
}

interface DragState {
  mode: 'rotate3d' | 'pan3d';
  pointerId: number;
  lastX: number;
  lastY: number;
  startX: number;
  startY: number;
  moved: boolean;
}

type SidebarMode = 'transcript' | 'island' | 'entry' | 'search' | 'idle';

interface NavState extends AtlasNavigationState {
  selectedIslandId: number | null;
}

const INITIAL_VIEW_3D: View3d = {
  rotateX: -0.05,
  rotateY: 0.6,
  zoom: 1.18,
  offsetX: 0,
  offsetY: 0,
};
const FOCUSED_ISLAND_FLOW_LIMIT = 16;
const DESKTOP_SIDEBAR_WIDTH = 400;
const DESKTOP_ATLAS_MIN_WIDTH = 1180;
const MOBILE_ATLAS_MAX_WIDTH = 640;
const MOBILE_HUD_SAFE_TOP = 96;
const CAMERA_CENTER_LERP = 0.12;
const CAMERA_ZOOM_LERP = 0.16;
const CAMERA_ENTRY_ZOOM_LERP = 0.24;
const CAMERA_OFFSET_LERP = 0.16;
const CAMERA_CENTER_EPSILON = 0.0005;
const CAMERA_ZOOM_EPSILON = 0.002;
const CAMERA_OFFSET_EPSILON = 0.25;
const ENTRY_FOCUS_ZOOM = 4.35;
const SEARCH_FOCUS_RESULT_LIMIT = 8;
const SEARCH_FOCUS_MIN_ZOOM = 3.4;
const SEARCH_FOCUS_MAX_ZOOM = 8;
const EMPTY_PROJECTED_FRAME: ProjectedFrame = {
  points: [],
  pointByEntryId: new Map(),
  islandById: new Map(),
};

function clamp(value: number, minimum: number, maximum: number): number {
  return Math.min(maximum, Math.max(minimum, value));
}

function percentile(values: number[], share: number): number {
  if (values.length === 0) {
    return 1;
  }

  const sorted = [...values].sort((left, right) => left - right);
  const index = Math.min(sorted.length - 1, Math.max(0, Math.round((sorted.length - 1) * share)));
  return sorted[index]!;
}

function createVisualGeometry(points: SemanticLandscapePoint[]): VisualGeometry {
  const count = Math.max(1, points.length);
  const centerX = points.reduce((total, point) => total + point.x3d, 0) / count;
  const centerY = points.reduce((total, point) => total + point.y3d, 0) / count;
  const centerZ = points.reduce((total, point) => total + point.z3d, 0) / count;
  const radius = Math.max(
    percentile(points.map((point) => Math.abs(point.x3d - centerX)), 0.98),
    percentile(points.map((point) => Math.abs(point.y3d - centerY)), 0.98),
    percentile(points.map((point) => Math.abs(point.z3d - centerZ)), 0.98),
    1,
  );

  return {
    centerX,
    centerY,
    centerZ,
    radius,
  };
}

function getIslandZoom(
  cluster: SemanticLandscapeCluster,
  entries: SemanticLandscapePoint[],
  geometry: VisualGeometry,
): number {
  const radius = Math.max(
    percentile(
      entries.map((point) =>
        Math.hypot(point.x3d - cluster.x3d, point.y3d - cluster.y3d, point.z3d - cluster.z3d),
      ),
      0.92,
    ),
    geometry.radius * 0.06,
  );
  const islandSpan = radius * 2;
  const atlasSpan = geometry.radius * 2;

  return clamp((atlasSpan / islandSpan) * 0.72, 1.45, 8);
}

function getWeightedCenter(points: SemanticLandscapePoint[]) {
  let totalWeight = 0;
  let centerX = 0;
  let centerY = 0;
  let centerZ = 0;

  for (let index = 0; index < points.length; index += 1) {
    const point = points[index]!;
    const weight = points.length - index;
    totalWeight += weight;
    centerX += point.x3d * weight;
    centerY += point.y3d * weight;
    centerZ += point.z3d * weight;
  }

  return {
    centerX: centerX / totalWeight,
    centerY: centerY / totalWeight,
    centerZ: centerZ / totalWeight,
  };
}

function getSearchFocus(
  results: SearchResult[],
  pointsById: Map<string, SemanticLandscapePoint>,
  geometry: VisualGeometry,
  selectedEntryId: string | null,
): VisualFocus | null {
  const points = results
    .slice(0, SEARCH_FOCUS_RESULT_LIMIT)
    .map((result) => pointsById.get(result.entryId))
    .filter((point): point is SemanticLandscapePoint => !!point);

  if (!points.length) {
    return null;
  }

  const selectedPoint = selectedEntryId && results.some((result) => result.entryId === selectedEntryId)
    ? pointsById.get(selectedEntryId) ?? null
    : null;
  const focusPoints = selectedPoint && !points.some((point) => point.entryId === selectedPoint.entryId)
    ? [selectedPoint, ...points]
    : points;
  const center = selectedPoint
    ? { centerX: selectedPoint.x3d, centerY: selectedPoint.y3d, centerZ: selectedPoint.z3d }
    : getWeightedCenter(points);
  const radius = Math.max(
    percentile(
      focusPoints.map((point) =>
        Math.hypot(point.x3d - center.centerX, point.y3d - center.centerY, point.z3d - center.centerZ),
      ),
      0.85,
    ),
    geometry.radius * 0.05,
  );

  return {
    key: `search:${results.slice(0, SEARCH_FOCUS_RESULT_LIMIT).map((result) => result.entryId).join('|')}:${selectedEntryId ?? ''}`,
    centerX: center.centerX,
    centerY: center.centerY,
    centerZ: center.centerZ,
    zoom: clamp((geometry.radius / radius) * 0.82, SEARCH_FOCUS_MIN_ZOOM, SEARCH_FOCUS_MAX_ZOOM),
  };
}

interface TrigCache {
  cosX: number;
  sinX: number;
  cosY: number;
  sinY: number;
}

function computeTrig(view: View3d): TrigCache {
  return {
    cosX: Math.cos(view.rotateX),
    sinX: Math.sin(view.rotateX),
    cosY: Math.cos(view.rotateY),
    sinY: Math.sin(view.rotateY),
  };
}

function createProjectionCache(
  points: SemanticLandscapePoint[],
  clusterById: Map<number, SemanticLandscapeCluster>,
): ProjectionCache {
  const projectedPoints = points.map((point): ProjectedPoint => ({
    point,
    rawX: 0,
    rawY: 0,
    x: 0,
    y: 0,
    depth: 0,
    culled: false,
  }));
  const sortedPoints = [...projectedPoints];
  const pointByEntryId = new Map<string, ProjectedPoint>();
  const islandById = new Map<number, ProjectedIsland>();

  for (const item of projectedPoints) {
    pointByEntryId.set(item.point.entryId, item);
  }

  return {
    sourcePoints: points,
    clusterById,
    projectedPoints,
    sortedPoints,
    radiusScratch: new Array(points.length),
    pointByEntryId,
    islandTotals: new Map(),
    islandById,
    frame: {
      points: sortedPoints,
      pointByEntryId,
      islandById,
    },
  };
}

function project3dRawInto(
  point: SemanticLandscapePoint,
  geometry: VisualGeometry,
  trig: TrigCache,
  target: ProjectedPoint,
): void {
  const x = (point.x3d - geometry.centerX) / (geometry.radius * 2);
  const y = (geometry.centerY - point.y3d) / (geometry.radius * 2);
  const z = (point.z3d - geometry.centerZ) / (geometry.radius * 2);
  const x1 = x * trig.cosY + z * trig.sinY;
  const z1 = -x * trig.sinY + z * trig.cosY;
  const y1 = y * trig.cosX - z1 * trig.sinX;
  const z2 = y * trig.sinX + z1 * trig.cosX;
  const perspective = 1 / (1 + z2 * 0.72);

  target.rawX = x1 * perspective;
  target.rawY = y1 * perspective;
  target.depth = z2;
  target.culled = z2 < -1.1;
}

function percentileScratch(values: number[], count: number, share: number): number {
  if (count <= 0) {
    return 1;
  }

  values.length = count;
  values.sort((left, right) => left - right);
  const index = Math.min(count - 1, Math.max(0, Math.round((count - 1) * share)));
  return values[index]!;
}

function fitProjected3d(items: ProjectedPoint[], radiusScratch: number[], width: number, height: number) {
  let visibleCount = 0;
  let centerX = 0;
  let centerY = 0;

  for (const item of items) {
    if (item.culled) {
      continue;
    }

    visibleCount += 1;
    centerX += item.rawX;
    centerY += item.rawY;
  }

  const count = Math.max(1, visibleCount);
  centerX /= count;
  centerY /= count;

  let radiusCount = 0;
  for (const item of items) {
    if (item.culled) {
      continue;
    }

    radiusScratch[radiusCount] = Math.hypot(item.rawX - centerX, item.rawY - centerY);
    radiusCount += 1;
  }

  const radius = Math.max(percentileScratch(radiusScratch, radiusCount, 0.98), 0.01);
  const scale = (Math.min(width, height) * 0.44) / radius;

  return { centerX, centerY, scale };
}

function projectAtlasFrame(
  cache: ProjectionCache,
  projectionGeometry: VisualGeometry,
  view: View3d,
  width: number,
  height: number,
  visualFocus: VisualFocus | null,
): ProjectedFrame {
  const trig = computeTrig(view);
  for (let index = 0; index < cache.projectedPoints.length; index += 1) {
    const item = cache.projectedPoints[index]!;
    project3dRawInto(item.point, projectionGeometry, trig, item);
    cache.sortedPoints[index] = item;
  }

  const sidebarGutter = width >= DESKTOP_ATLAS_MIN_WIDTH ? DESKTOP_SIDEBAR_WIDTH : 0;
  const topGutter = width <= MOBILE_ATLAS_MAX_WIDTH ? MOBILE_HUD_SAFE_TOP : 0;
  const plotWidth = Math.max(1, width - sidebarGutter);
  const plotHeight = Math.max(1, height - topGutter);
  const fitted = fitProjected3d(cache.projectedPoints, cache.radiusScratch, plotWidth, plotHeight);
  const centerX = visualFocus ? 0 : fitted.centerX;
  const centerY = visualFocus ? 0 : fitted.centerY;

  for (const item of cache.projectedPoints) {
    item.x = (item.rawX - centerX) * fitted.scale * view.zoom + plotWidth / 2 + view.offsetX;
    item.y = (item.rawY - centerY) * fitted.scale * view.zoom + topGutter + plotHeight / 2 + view.offsetY;
  }

  cache.sortedPoints.sort((left, right) => right.depth - left.depth);
  cache.islandTotals.clear();
  cache.islandById.clear();

  for (const item of cache.sortedPoints) {
    if (item.culled) {
      continue;
    }

    const total = cache.islandTotals.get(item.point.clusterId);
    if (total) {
      total.x += item.x;
      total.y += item.y;
      total.depth += item.depth;
      total.count += 1;
    } else {
      cache.islandTotals.set(item.point.clusterId, { x: item.x, y: item.y, depth: item.depth, count: 1 });
    }
  }

  for (const [clusterId, total] of cache.islandTotals) {
    const cluster = cache.clusterById.get(clusterId);
    if (!cluster) {
      continue;
    }

    cache.islandById.set(clusterId, {
      cluster,
      x: total.x / total.count,
      y: total.y / total.count,
      depth: total.depth / total.count,
    });
  }

  return cache.frame;
}

function getDepthAlphaScale(depth: number): number {
  return clamp(1 - depth * 0.78, 0.45, 1.36);
}

function buildIslandFlows(points: SemanticLandscapePoint[]): IslandFlow[] {
  const pointsByVideo = new Map<string, SemanticLandscapePoint[]>();
  const flowCounts = new Map<string, IslandFlow>();

  for (const point of points) {
    const videoPoints = pointsByVideo.get(point.videoId);
    if (videoPoints) {
      videoPoints.push(point);
    } else {
      pointsByVideo.set(point.videoId, [point]);
    }
  }

  for (const videoPoints of pointsByVideo.values()) {
    videoPoints.sort(
      (left, right) =>
        left.segIndex - right.segIndex ||
        (left.startMs ?? Number.MAX_SAFE_INTEGER) - (right.startMs ?? Number.MAX_SAFE_INTEGER) ||
        left.entryId.localeCompare(right.entryId),
    );

    for (let index = 0; index < videoPoints.length - 1; index += 1) {
      const fromClusterId = videoPoints[index]!.clusterId;
      const toClusterId = videoPoints[index + 1]!.clusterId;
      if (fromClusterId === toClusterId) {
        continue;
      }

      const key = `${fromClusterId}:${toClusterId}`;
      const existing = flowCounts.get(key);
      if (existing) {
        existing.count += 1;
      } else {
        flowCounts.set(key, { fromClusterId, toClusterId, count: 1 });
      }
    }
  }

  return [...flowCounts.values()].sort(
    (left, right) =>
      right.count - left.count ||
      left.fromClusterId - right.fromClusterId ||
      left.toClusterId - right.toClusterId,
  );
}

function drawIslandFlowArc(
  context: CanvasRenderingContext2D,
  flow: IslandFlow,
  from: ProjectedIsland,
  to: ProjectedIsland,
  maxCount: number,
  theme: ThemeMode,
  focused: boolean,
): void {
  const dx = to.x - from.x;
  const dy = to.y - from.y;
  const distance = Math.hypot(dx, dy);
  if (distance < 8) {
    return;
  }

  const strength = Math.sqrt(flow.count / Math.max(1, maxCount));
  const depthAlpha = clamp(getDepthAlphaScale((from.depth + to.depth) / 2), 0.55, 1.14);
  const baseAlpha = focused ? 0.035 + strength * 0.13 : 0.025 + strength * 0.1;
  const alpha = clamp(baseAlpha * depthAlpha * (theme === 'light' ? 0.72 : 1), 0.025, 0.22);
  const lineWidth = focused ? 0.55 + strength * 1.65 : 0.45 + strength * 1.25;
  const reciprocalOffset = flow.fromClusterId < flow.toClusterId ? 1 : -1;
  const curve = clamp(distance * 0.2, 28, focused ? 130 : 95) * reciprocalOffset;
  const controlX = (from.x + to.x) / 2 - (dy / distance) * curve;
  const controlY = (from.y + to.y) / 2 + (dx / distance) * curve;
  const gradient = context.createLinearGradient(from.x, from.y, to.x, to.y);

  gradient.addColorStop(0, hexToRgba(from.cluster.color, alpha * 0.48));
  gradient.addColorStop(1, hexToRgba(to.cluster.color, alpha));

  context.save();
  context.globalCompositeOperation = 'source-over';
  context.strokeStyle = gradient;
  context.lineWidth = lineWidth;
  context.lineCap = 'round';
  context.lineJoin = 'round';
  context.shadowColor = hexToRgba(to.cluster.color, alpha * 0.55);
  context.shadowBlur = focused ? 4 : 2;
  context.beginPath();
  context.moveTo(from.x, from.y);
  context.quadraticCurveTo(controlX, controlY, to.x, to.y);
  context.stroke();

  if (strength > 0.62) {
    const t = 0.64;
    const arrowX = (1 - t) ** 2 * from.x + 2 * (1 - t) * t * controlX + t ** 2 * to.x;
    const arrowY = (1 - t) ** 2 * from.y + 2 * (1 - t) * t * controlY + t ** 2 * to.y;
    const tangentX = 2 * (1 - t) * (controlX - from.x) + 2 * t * (to.x - controlX);
    const tangentY = 2 * (1 - t) * (controlY - from.y) + 2 * t * (to.y - controlY);
    const arrowSize = clamp(lineWidth * 2 + 1.6, 4, 7.5);

    context.translate(arrowX, arrowY);
    context.rotate(Math.atan2(tangentY, tangentX));
    context.fillStyle = hexToRgba(to.cluster.color, alpha * 0.7);
    context.shadowBlur = focused ? 3 : 1;
    context.beginPath();
    context.moveTo(arrowSize, 0);
    context.lineTo(-arrowSize * 0.58, -arrowSize * 0.42);
    context.lineTo(-arrowSize * 0.58, arrowSize * 0.42);
    context.closePath();
    context.fill();
  }

  context.restore();
}

function formatCueTimestamp(ms: number | null): string {
  if (ms === null || !Number.isFinite(ms)) {
    return '--:--.---';
  }

  const totalMs = Math.max(0, Math.round(ms));
  const hours = Math.floor(totalMs / 3_600_000);
  const minutes = Math.floor((totalMs % 3_600_000) / 60_000);
  const seconds = Math.floor((totalMs % 60_000) / 1_000);
  const milliseconds = totalMs % 1_000;
  const secondFragment = `${seconds.toString().padStart(2, '0')}.${milliseconds
    .toString()
    .padStart(3, '0')}`;

  if (hours > 0) {
    return `${hours}:${minutes.toString().padStart(2, '0')}:${secondFragment}`;
  }

  return `${minutes.toString().padStart(2, '0')}:${secondFragment}`;
}


interface PairCardProps {
  entry: EntrySummary;
  isFocus?: boolean;
  onSelect: (entryId: string) => void;
  onOpenTranscript: (videoId: string, entryId: string) => void;
  onSearchLine: (text: string) => void;
  score?: number;
  clusterColor?: string;
  clusterLabel?: string;
  onClusterClick?: () => void;
}

type CssVars = CSSProperties & Record<`--${string}`, string>;
type IconName = 'atlas' | 'back' | 'moon' | 'reset' | 'sun' | 'text';

const ICONS: Record<IconName, ReactNode> = {
  atlas: (
    <>
      <circle cx="6" cy="7" r="1.5" />
      <circle cx="17" cy="5" r="1.5" />
      <circle cx="14" cy="16" r="1.5" />
      <circle cx="5" cy="18" r="1.5" />
      <path d="m7.4 6.8 8.1-1.5M6.9 8.2l6.2 6.6m2.2-.2 1.4-8M6.5 17.5l6-1.2" />
    </>
  ),
  back: (
    <>
      <path d="M19 12H5" />
      <path d="m12 5-7 7 7 7" />
    </>
  ),
  moon: <path d="M20 14.5A8 8 0 0 1 9.5 4a7 7 0 1 0 10.5 10.5Z" />,
  reset: (
    <>
      <path d="M5 12a7 7 0 1 0 2.1-5H4" />
      <path d="M4 3v4h4" />
    </>
  ),
  sun: (
    <>
      <circle cx="12" cy="12" r="4" />
      <path d="M12 2v2M12 20v2M2 12h2M20 12h2" />
      <path d="m4.9 4.9 1.4 1.4m11.4 11.4 1.4 1.4m-14.2 0 1.4-1.4m11.4-11.4 1.4-1.4" />
    </>
  ),
  text: <path d="M5 6h14M5 11h14M5 16h9" />,
};

function Icon({ name }: { name: IconName }) {
  return (
    <svg className="atlas-icon" aria-hidden="true" viewBox="0 0 24 24">
      {ICONS[name]}
    </svg>
  );
}

function SectionHeader({ title, count }: { title: string; count: ReactNode }) {
  return (
    <div className="atlas-section-header">
      <strong>{title}</strong>
      <span>{count}</span>
    </div>
  );
}

function EmptyState({ children }: { children: ReactNode }) {
  return (
    <div className="empty-state">
      <p>{children}</p>
    </div>
  );
}

function islandColorStyle(color: string): CssVars {
  const light = isLightHex(color);
  return {
    '--cluster-color': color,
    '--island-text': light ? '#020a06' : '#e2f5ea',
    '--island-text-muted': light ? 'rgba(2, 10, 6, 0.6)' : 'rgba(226, 245, 234, 0.6)',
    '--island-text-strong': light ? 'rgba(2, 10, 6, 0.75)' : 'rgba(226, 245, 234, 0.75)',
    '--island-overlay': light ? 'rgba(0, 0, 0, 0.06)' : 'rgba(255, 255, 255, 0.1)',
    '--island-overlay-border': light ? 'rgba(0, 0, 0, 0.1)' : 'rgba(255, 255, 255, 0.15)',
  };
}

function islandPanelStyle(panel: IslandPanelData): CssVars {
  return islandColorStyle(panel.cluster.color);
}

function PairCard({
  entry,
  isFocus,
  onSelect,
  onOpenTranscript,
  onSearchLine,
  score,
  clusterColor,
  clusterLabel,
  onClusterClick,
}: PairCardProps) {
  const { entryId, videoId, segIndex, en, zh, startMs, endMs } = entry;

  return (
    <article
      className={classNames('pair-card', isFocus && 'is-focus')}
      role="button"
      tabIndex={0}
      onClick={() => onSelect(entryId)}
      onKeyDown={(event) => {
        if (event.target === event.currentTarget && (event.key === 'Enter' || event.key === ' ')) {
          event.preventDefault();
          onSelect(entryId);
        }
      }}
    >
      {onClusterClick && (
        <button
          className="pair-card-cluster"
          style={{ ['--cluster-color' as string]: clusterColor }}
          type="button"
          onClick={(event) => {
            event.stopPropagation();
            onClusterClick();
          }}
        >
          {clusterLabel}
        </button>
      )}

      <div className="pair-card-header">
        <div className="pair-card-meta">
          <button
            className="video-id-button"
            type="button"
            onClick={(event) => {
              event.stopPropagation();
              onOpenTranscript(videoId, entryId);
            }}
          >
            {videoId}
          </button>
          <span className="pair-card-seg">#{segIndex}</span>
          {startMs != null && (
            <span className="pair-card-time">
              {formatCueTimestamp(startMs)}-{formatCueTimestamp(endMs ?? null)}
            </span>
          )}
        </div>

        {score !== undefined && <span className="pair-card-score">{score.toFixed(3)}</span>}
      </div>

      <button
        className="pair-card-en"
        type="button"
        title="Search for similar lines"
        disabled={!en.trim()}
        onClick={(event) => {
          event.stopPropagation();
          onSearchLine(en);
        }}
      >
        {en}
      </button>
      <p className="pair-card-zh">{zh}</p>
    </article>
  );
}

export default function TmAtlasPanel({
  data,
  dataLoading,
  dataErrorText,
  theme,
  viewMode,
  statusItems,
  navigation,
  searchReady,
  searching,
  onQueryChange,
  onSearch,
  onSelectEntry,
  onOpenTranscript,
  onSelectTranscriptEntry,
  onSearchLine,
  onCloseTranscript,
  onRestoreNavigationState,
  onClear,
  onToggleTheme,
  onToggleViewMode,
}: TmAtlasPanelProps) {
  const {
    query,
    searchResults,
    searchNote,
    errorText,
    selectedEntryId,
    transcriptVideoId,
    transcriptItems,
    transcriptFocusEntryId,
    transcriptLoading,
    transcriptErrorText,
  } = navigation;
  const wrapRef = useRef<HTMLDivElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const dragRef = useRef<DragState | null>(null);
  const sidebarBodyRef = useRef<HTMLDivElement | null>(null);
  const transcriptBodyRef = useRef<HTMLDivElement | null>(null);
  const transcriptItemRefs = useRef(new Map<string, HTMLLIElement>());
  const cameraAnimRef = useRef(0);
  const canvasRenderFrameRef = useRef(0);
  const manualZoomOverrideRef = useRef(false);
  const targetCenterRef = useRef<{ x: number; y: number; z: number } | null>(null);
  const animatedCenterRef = useRef<{ x: number; y: number; z: number } | null>(null);
  const projectionCacheRef = useRef<ProjectionCache | null>(null);
  const projectedFrameRef = useRef<ProjectedFrame>(EMPTY_PROJECTED_FRAME);
  const renderAtlasRef = useRef<() => void>(() => {});
  const view3dRef = useRef<View3d>({ ...INITIAL_VIEW_3D });
  const historyRef = useRef<NavState[]>([]);
  const [size, setSize] = useState({ width: 1, height: 1 });
  const effectiveInitialView = useMemo<View3d>(
    () => data?.initialView ?? INITIAL_VIEW_3D,
    [data?.initialView],
  );
  const appliedDataViewRef = useRef(false);
  const isTextMode = viewMode === 'text';

  useEffect(() => {
    if (!isTextMode && data?.initialView && !appliedDataViewRef.current) {
      appliedDataViewRef.current = true;
      view3dRef.current = { ...data.initialView };
      requestAtlasRender();
    }
  }, [data?.initialView, isTextMode]);
  const [hoverState, setHoverState] = useState<HoverState | null>(null);
  const [selectedIslandId, setSelectedIslandId] = useState<number | null>(null);

  const clusterById = useMemo(
    () => new Map(data?.clusters.map((cluster) => [cluster.id, cluster]) ?? []),
    [data],
  );
  const pointById = useMemo(
    () => new Map(data?.points.map((point) => [point.entryId, point]) ?? []),
    [data],
  );
  const searchHitIds = useMemo(
    () => new Set(searchResults.map((result) => result.entryId)),
    [searchResults],
  );
  const searchResultById = useMemo(
    () => new Map(searchResults.map((result) => [result.entryId, result])),
    [searchResults],
  );
  const rankedIslands = useMemo(
    () =>
      [...(data?.clusters ?? [])].sort(
        (left, right) => right.size - left.size || left.label.localeCompare(right.label),
      ),
    [data],
  );
  const visualGeometry = useMemo(
    () => createVisualGeometry(data?.points ?? []),
    [data],
  );
  const islandEntriesById = useMemo(() => {
    const entriesById = new Map<number, SemanticLandscapePoint[]>();

    if (!data) {
      return entriesById;
    }

    for (const point of data.points) {
      const entries = entriesById.get(point.clusterId);
      if (entries) {
        entries.push(point);
      } else {
        entriesById.set(point.clusterId, [point]);
      }
    }

    for (const entries of entriesById.values()) {
      entries.sort((left, right) => left.videoId.localeCompare(right.videoId) || left.segIndex - right.segIndex);
    }

    return entriesById;
  }, [data]);
  const islandFlows = useMemo(
    () => buildIslandFlows(data?.points ?? []),
    [data],
  );
  const videoCount = useMemo(
    () => new Set(data?.points.map((point) => point.videoId) ?? []).size,
    [data],
  );

  const selectedPoint = selectedEntryId ? pointById.get(selectedEntryId) ?? null : null;
  const selectedEntry = selectedPoint ?? (selectedEntryId ? searchResultById.get(selectedEntryId) ?? null : null);
  const selectedCluster = selectedPoint ? clusterById.get(selectedPoint.clusterId)! : null;
  const showTranscriptPanel = !!transcriptVideoId;
  const selectedIsland = selectedIslandId !== null ? clusterById.get(selectedIslandId)! : null;
  const selectedIslandEntries = selectedIsland ? islandEntriesById.get(selectedIsland.id)! : [];
  const isInSearchResults = selectedEntryId ? searchResultById.has(selectedEntryId) : false;
  const topSearchResults = useMemo(() => searchResults.slice(0, 12), [searchResults]);
  const selectedIslandPanel = useMemo<IslandPanelData | null>(() => {
    if (!selectedIsland || !data) {
      return null;
    }

    return {
      cluster: selectedIsland,
      entries: selectedIslandEntries,
      share: (selectedIsland.size / data.pointCount) * 100,
      zoom: getIslandZoom(selectedIsland, selectedIslandEntries, visualGeometry),
    };
  }, [data, selectedIsland, selectedIslandEntries, visualGeometry]);
  const searchFocus = useMemo<VisualFocus | null>(() => {
    if (showTranscriptPanel || selectedIslandPanel || !searchResults.length || (selectedEntryId && !isInSearchResults)) {
      return null;
    }

    return getSearchFocus(topSearchResults, pointById, visualGeometry, selectedEntryId);
  }, [
    isInSearchResults,
    pointById,
    searchResults.length,
    selectedEntryId,
    selectedIslandPanel,
    showTranscriptPanel,
    topSearchResults,
    visualGeometry,
  ]);
  const hoveredPoint = hoverState ? pointById.get(hoverState.entryId)! : null;
  const visualFocus = useMemo<VisualFocus | null>(() => {
    if (selectedIslandPanel) {
      const cluster = selectedIslandPanel.cluster;
      return {
        key: `island:${cluster.id}`,
        centerX: cluster.x3d,
        centerY: cluster.y3d,
        centerZ: cluster.z3d,
        zoom: selectedIslandPanel.zoom,
      };
    }

    if (searchFocus) {
      return searchFocus;
    }

    if (selectedPoint) {
      return {
        key: `entry:${selectedPoint.entryId}`,
        centerX: selectedPoint.x3d,
        centerY: selectedPoint.y3d,
        centerZ: selectedPoint.z3d,
        zoom: ENTRY_FOCUS_ZOOM,
      };
    }

    return null;
  }, [searchFocus, selectedIslandPanel, selectedPoint]);
  const cameraZoomLerp = visualFocus?.key.startsWith('entry:') ? CAMERA_ENTRY_ZOOM_LERP : CAMERA_ZOOM_LERP;

  function getProjectionGeometry(): VisualGeometry {
    const center = animatedCenterRef.current;
    if (!center) {
      return visualFocus
        ? {
            ...visualGeometry,
            centerX: visualFocus.centerX,
            centerY: visualFocus.centerY,
            centerZ: visualFocus.centerZ,
          }
        : visualGeometry;
    }
    return { ...visualGeometry, centerX: center.x, centerY: center.y, centerZ: center.z };
  }

  const transcriptHasTimestamps = transcriptItems.some((item) => item.startMs !== null || item.endMs !== null);

  const sidebarMode: SidebarMode = showTranscriptPanel
    ? 'transcript'
    : selectedIslandPanel
      ? 'island'
      : selectedEntry && (!searchResults.length || !isInSearchResults)
        ? 'entry'
        : searchResults.length > 0
          ? 'search'
          : 'idle';

  const showIslandBrowser = !!data && !query.trim();
  const isIdle = sidebarMode === 'idle' && !errorText && !searchNote;
  const canGoBack = sidebarMode !== 'idle' || historyRef.current.length > 0;
  const sidebarScrollKey =
    sidebarMode === 'search'
      ? topSearchResults.map((result) => result.entryId).join('|')
      : sidebarMode === 'island'
        ? String(selectedIslandId ?? '')
        : sidebarMode === 'entry'
          ? selectedEntryId ?? ''
          : transcriptVideoId ?? '';

  useEffect(() => {
    const container = wrapRef.current;
    if (!container) {
      return;
    }

    const observer = new ResizeObserver((entries) => {
      const entry = entries[0];
      if (!entry) {
        return;
      }

      const nextSize = {
        width: Math.max(1, Math.floor(entry.contentRect.width)),
        height: Math.max(1, Math.floor(entry.contentRect.height)),
      };

      setSize((currentSize) =>
        currentSize.width === nextSize.width && currentSize.height === nextSize.height ? currentSize : nextSize,
      );
    });

    observer.observe(container);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    if (searchResults.length > 0) {
      setSelectedIslandId(null);
    }
  }, [searchResults.length]);

  useEffect(() => {
    sidebarBodyRef.current?.scrollTo({ top: 0, behavior: 'auto' });
  }, [sidebarMode, sidebarScrollKey]);

  useEffect(() => {
    if (!transcriptVideoId || transcriptItems.length === 0) {
      return;
    }

    const nextEntryId =
      transcriptFocusEntryId ??
      transcriptItems.find((item) => item.entryId === selectedEntryId)?.entryId ??
      transcriptItems[0]?.entryId ??
      null;

    if (!nextEntryId) {
      return;
    }

    const frameId = window.requestAnimationFrame(() => {
      const container = transcriptBodyRef.current;
      const entry = transcriptItemRefs.current.get(nextEntryId);

      if (!container || !entry) {
        return;
      }

      const containerBounds = container.getBoundingClientRect();
      const entryBounds = entry.getBoundingClientRect();
      const top =
        container.scrollTop +
        (entryBounds.top - containerBounds.top) -
        (container.clientHeight - entryBounds.height) / 2;
      container.scrollTo({ top: Math.max(0, top), behavior: 'auto' });
    });

    return () => window.cancelAnimationFrame(frameId);
  }, [selectedEntryId, transcriptFocusEntryId, transcriptItems, transcriptVideoId]);

  useEffect(() => {
    if (isTextMode || !data?.points.length) {
      cancelAnimationFrame(cameraAnimRef.current);
      return;
    }

    const targetZoom = visualFocus ? visualFocus.zoom : effectiveInitialView.zoom;
    manualZoomOverrideRef.current = false;
    const target = visualFocus
      ? { x: visualFocus.centerX, y: visualFocus.centerY, z: visualFocus.centerZ }
      : { x: visualGeometry.centerX, y: visualGeometry.centerY, z: visualGeometry.centerZ };
    targetCenterRef.current = target;
    animatedCenterRef.current ??= { ...target };

    const lerp = (a: number, b: number, t: number) => a + (b - a) * t;
    let running = true;

    function step() {
      if (!running) return;

      const center = animatedCenterRef.current;
      const target = targetCenterRef.current;
      if (center && target) {
        center.x = lerp(center.x, target.x, CAMERA_CENTER_LERP);
        center.y = lerp(center.y, target.y, CAMERA_CENTER_LERP);
        center.z = lerp(center.z, target.z, CAMERA_CENTER_LERP);
      }

      const current = view3dRef.current;
      const offsetDistance = Math.hypot(current.offsetX, current.offsetY);
      const shouldZoom = !manualZoomOverrideRef.current;
      const zoomDistance = shouldZoom ? Math.abs(current.zoom - targetZoom) : 0;
      const centerDistance = center && target
        ? Math.hypot(center.x - target.x, center.y - target.y, center.z - target.z) / Math.max(1, visualGeometry.radius)
        : 0;
      if (
        offsetDistance < CAMERA_OFFSET_EPSILON &&
        zoomDistance < CAMERA_ZOOM_EPSILON &&
        centerDistance < CAMERA_CENTER_EPSILON
      ) {
        manualZoomOverrideRef.current = false;
        current.zoom = shouldZoom ? targetZoom : current.zoom;
        current.offsetX = 0;
        current.offsetY = 0;
        renderAtlasNow();
        return;
      }

      current.zoom = shouldZoom ? lerp(current.zoom, targetZoom, cameraZoomLerp) : current.zoom;
      current.offsetX = lerp(current.offsetX, 0, CAMERA_OFFSET_LERP);
      current.offsetY = lerp(current.offsetY, 0, CAMERA_OFFSET_LERP);
      renderAtlasNow();
      cameraAnimRef.current = requestAnimationFrame(step);
    }

    cameraAnimRef.current = requestAnimationFrame(step);
    return () => {
      running = false;
      cancelAnimationFrame(cameraAnimRef.current);
    };
  }, [
    data?.points.length,
    visualFocus?.centerX,
    visualFocus?.centerY,
    visualFocus?.centerZ,
    visualFocus?.key,
    visualFocus?.zoom,
    visualGeometry,
    cameraZoomLerp,
    isTextMode,
  ]);

  function requestAtlasRender(): void {
    if (canvasRenderFrameRef.current) {
      return;
    }

    canvasRenderFrameRef.current = window.requestAnimationFrame(() => {
      canvasRenderFrameRef.current = 0;
      renderAtlasRef.current();
    });
  }

  function renderAtlasNow(): void {
    if (canvasRenderFrameRef.current) {
      window.cancelAnimationFrame(canvasRenderFrameRef.current);
      canvasRenderFrameRef.current = 0;
    }
    renderAtlasRef.current();
  }

  useEffect(() => {
    renderAtlasRef.current = () => {
      const canvas = canvasRef.current;
      if (isTextMode || !canvas || !data) {
        if (isTextMode) {
          projectionCacheRef.current = null;
        }
        projectedFrameRef.current = EMPTY_PROJECTED_FRAME;
        return;
      }

      const context = canvas.getContext('2d');
      if (!context) {
        return;
      }

      const dpr = window.devicePixelRatio || 1;
      const canvasWidth = Math.round(size.width * dpr);
      const canvasHeight = Math.round(size.height * dpr);
      const styleWidth = `${size.width}px`;
      const styleHeight = `${size.height}px`;

      if (canvas.width !== canvasWidth || canvas.height !== canvasHeight) {
        canvas.width = canvasWidth;
        canvas.height = canvasHeight;
      }
      if (canvas.style.width !== styleWidth) {
        canvas.style.width = styleWidth;
      }
      if (canvas.style.height !== styleHeight) {
        canvas.style.height = styleHeight;
      }
      context.setTransform(dpr, 0, 0, dpr, 0, 0);
      context.clearRect(0, 0, size.width, size.height);

      const background = theme === 'dark' ? '#06100b' : '#f5f0e8';
      const grid = theme === 'dark' ? 'rgba(197, 228, 203, 0.08)' : 'rgba(51, 104, 72, 0.12)';
      const muted = theme === 'dark' ? '#9fb2a5' : '#617266';
      const text = theme === 'dark' ? '#edf8ef' : '#17251b';
      const selected = theme === 'dark' ? '#4ade80' : '#16a34a';
      let projectionCache = projectionCacheRef.current;
      if (
        !projectionCache ||
        projectionCache.sourcePoints !== data.points ||
        projectionCache.clusterById !== clusterById ||
        projectionCache.projectedPoints.length !== data.points.length
      ) {
        projectionCache = createProjectionCache(data.points, clusterById);
        projectionCacheRef.current = projectionCache;
      }

      const view3d = view3dRef.current;
      const projectedFrame = projectAtlasFrame(
        projectionCache,
        getProjectionGeometry(),
        view3d,
        size.width,
        size.height,
        visualFocus,
      );
      projectedFrameRef.current = projectedFrame;
      const projectedPoints = projectedFrame.points;
      const projectedPointByEntryId = projectedFrame.pointByEntryId;
      const projectedIslandById = projectedFrame.islandById;

      context.fillStyle = background;
      context.fillRect(0, 0, size.width, size.height);

      context.strokeStyle = grid;
      context.lineWidth = 1;
      const gridStep = Math.max(80, Math.round(Math.min(size.width, size.height) / 7));
      context.beginPath();
      for (let x = (view3d.offsetX % gridStep) - gridStep; x < size.width + gridStep; x += gridStep) {
        context.moveTo(x, 0);
        context.lineTo(x, size.height);
      }
      for (let y = (view3d.offsetY % gridStep) - gridStep; y < size.height + gridStep; y += gridStep) {
        context.moveTo(0, y);
        context.lineTo(size.width, y);
      }
      context.stroke();

      if (sidebarMode === 'island' && selectedIslandId !== null && islandFlows.length > 0) {
        const drawableFlows: IslandFlow[] = [];
        let maxFlowCount = 1;

        for (const flow of islandFlows) {
          if (
            flow.fromClusterId !== selectedIslandId &&
            flow.toClusterId !== selectedIslandId
          ) {
            continue;
          }

          if (!projectedIslandById.has(flow.fromClusterId) || !projectedIslandById.has(flow.toClusterId)) {
            continue;
          }

          drawableFlows.push(flow);
          maxFlowCount = Math.max(maxFlowCount, flow.count);
          if (drawableFlows.length >= FOCUSED_ISLAND_FLOW_LIMIT) {
            break;
          }
        }

        for (let index = drawableFlows.length - 1; index >= 0; index -= 1) {
          const flow = drawableFlows[index]!;
          const from = projectedIslandById.get(flow.fromClusterId);
          const to = projectedIslandById.get(flow.toClusterId);
          if (from && to) {
            drawIslandFlowArc(context, flow, from, to, maxFlowCount, theme, true);
          }
        }
      }

      const rankedSearchHits = topSearchResults
        .map((result) => projectedPointByEntryId.get(result.entryId))
        .filter((item): item is ProjectedPoint => !!item && !item.culled);
      const allVideoPathPoints = transcriptVideoId
        ? transcriptItems
            .map((item) => projectedPointByEntryId.get(item.entryId))
            .filter((item): item is ProjectedPoint => !!item && !item.culled)
        : [];
      const selectedVideoPathIndex = allVideoPathPoints.findIndex((item) => item.point.entryId === selectedEntryId);
      const videoPathWindowRadius = 5;
      const videoPathWindowStart = selectedVideoPathIndex >= 0
        ? Math.max(0, selectedVideoPathIndex - videoPathWindowRadius)
        : 0;
      const videoPathWindowEnd = selectedVideoPathIndex >= 0
        ? Math.min(allVideoPathPoints.length, selectedVideoPathIndex + videoPathWindowRadius + 1)
        : allVideoPathPoints.length;
      const videoPathPoints = allVideoPathPoints.slice(videoPathWindowStart, videoPathWindowEnd);
      const videoPathEntryIds = new Set(videoPathPoints.map((item) => item.point.entryId));
      const videoPathFocusColor = selectedPoint?.color ?? videoPathPoints[0]?.point.color ?? selected;
      const localSelectedIndex = selectedVideoPathIndex >= 0 ? selectedVideoPathIndex - videoPathWindowStart : -1;
      const emphasizedVideoPathIds = localSelectedIndex >= 0
        ? new Set(
            [localSelectedIndex - 1, localSelectedIndex, localSelectedIndex + 1]
              .filter((index) => index >= 0 && index < videoPathPoints.length)
              .map((index) => videoPathPoints[index]!.point.entryId),
          )
        : new Set<string>();

      if (sidebarMode === 'search' && rankedSearchHits.length > 1) {
        context.save();
        context.strokeStyle = hexToRgba(selected, 0.44);
        context.lineWidth = 1.25;
        context.setLineDash([5, 7]);
        context.beginPath();
        const firstHit = rankedSearchHits[0]!;
        context.moveTo(firstHit.x, firstHit.y);
        for (const item of rankedSearchHits.slice(1)) {
          context.lineTo(item.x, item.y);
        }
        context.stroke();
        context.restore();
      }

      if (videoPathPoints.length > 1) {
        context.save();
        context.strokeStyle = hexToRgba(muted, 0.2);
        context.lineWidth = 1;
        context.beginPath();
        const firstPoint = videoPathPoints[0]!;
        context.moveTo(firstPoint.x, firstPoint.y);
        for (const item of videoPathPoints.slice(1)) {
          context.lineTo(item.x, item.y);
        }
        context.stroke();

        if (localSelectedIndex >= 0) {
          context.lineWidth = 2;
          const highlightedSegments: Array<[number, number]> = [
            [localSelectedIndex - 1, localSelectedIndex],
            [localSelectedIndex, localSelectedIndex + 1],
          ];
          for (const [fromIndex, toIndex] of highlightedSegments) {
            const fromPoint = videoPathPoints[fromIndex];
            const toPoint = videoPathPoints[toIndex];
            if (fromPoint && toPoint) {
              const blended = blendHexColors(fromPoint.point.color, toPoint.point.color);
              context.strokeStyle = `rgba(${blended}, 0.72)`;
              context.beginPath();
              context.moveTo(fromPoint.x, fromPoint.y);
              context.lineTo(toPoint.x, toPoint.y);
              context.stroke();
            }
          }
        }
        context.restore();
      }

      const hasSearch = searchHitIds.size > 0 && sidebarMode === 'search';
      const hasVideoPath = videoPathEntryIds.size > 0;
      const pointRadiusScale = clamp(Math.min(size.width, size.height) / 720, 0.5, 1);

      for (const item of projectedPoints) {
        if (item.culled) {
          continue;
        }

        const isSelected = item.point.entryId === selectedEntryId;
        const isHovered = item.point.entryId === hoverState?.entryId;
        const isSearchHit = searchHitIds.has(item.point.entryId);
        const isVideoPathPoint = videoPathEntryIds.has(item.point.entryId);
        const isEmphasizedVideoPathPoint = emphasizedVideoPathIds.has(item.point.entryId);
        const isOutsideSelectedIsland = sidebarMode === 'island' && selectedIslandId !== null && item.point.clusterId !== selectedIslandId;
        const depthRadiusScale = clamp(1 - item.depth * 0.9, 0.55, 1.45);
        const depthAlphaScale = getDepthAlphaScale(item.depth);
        const baseRadius = isSelected
          ? 5.2
          : isHovered
            ? 4.4
            : isSearchHit || isEmphasizedVideoPathPoint
              ? 3.4
              : isVideoPathPoint
                ? 2.7
                : 2.35;
        const radius = Math.max(
          0.75,
          baseRadius * pointRadiusScale * (isSelected || isHovered ? clamp(depthRadiusScale, 0.92, 1.14) : depthRadiusScale),
        );
        const baseAlpha = isSelected
          ? 1
          : isHovered
            ? 0.95
            : isSearchHit
              ? 0.86
              : isEmphasizedVideoPathPoint
                ? 0.78
              : isVideoPathPoint
                ? 0.38
              : hasSearch
                ? 0.16
                : hasVideoPath
                  ? 0.1
                : isOutsideSelectedIsland
                  ? 0.08
                  : 0.68;
        const alpha = isSelected ? 1 : clamp(baseAlpha * depthAlphaScale, 0.035, 1);

        context.fillStyle = hexToRgba(isSearchHit && !isSelected ? selected : item.point.color, alpha);
        context.beginPath();
        context.arc(item.x, item.y, radius, 0, Math.PI * 2);
        context.fill();

        if (isEmphasizedVideoPathPoint && !isSelected && !isHovered) {
          context.strokeStyle = hexToRgba(videoPathFocusColor, 0.45);
          context.lineWidth = Math.max(0.7, pointRadiusScale);
          context.beginPath();
          context.arc(item.x, item.y, radius + 2.5 * pointRadiusScale, 0, Math.PI * 2);
          context.stroke();
        }

        if (isSelected || isHovered) {
          context.strokeStyle = hexToRgba(text, isSelected ? 0.92 : 0.7);
          context.lineWidth = Math.max(0.8, (isSelected ? 1.8 : 1.2) * pointRadiusScale);
          context.beginPath();
          context.arc(item.x, item.y, radius + 3 * pointRadiusScale, 0, Math.PI * 2);
          context.stroke();
        }
      }

      if (sidebarMode === 'search' && rankedSearchHits.length > 0) {
        context.save();
        context.strokeStyle = hexToRgba(selected, 0.9);
        context.lineWidth = Math.max(0.8, 1.5 * pointRadiusScale);
        for (let index = 0; index < rankedSearchHits.length; index += 1) {
          const item = rankedSearchHits[index]!;
          context.beginPath();
          context.arc(item.x, item.y, (8 + Math.min(index, 5) * 0.55) * pointRadiusScale, 0, Math.PI * 2);
          context.stroke();
        }
        context.restore();
      }
    };

    requestAtlasRender();
    return () => {
      if (canvasRenderFrameRef.current) {
        window.cancelAnimationFrame(canvasRenderFrameRef.current);
        canvasRenderFrameRef.current = 0;
      }
    };
  }, [
    clusterById,
    data,
    hoverState?.entryId,
    islandFlows,
    isTextMode,
    searchHitIds,
    selectedEntryId,
    selectedIslandId,
    selectedPoint?.color,
    sidebarMode,
    size.height,
    size.width,
    theme,
    topSearchResults,
    transcriptItems,
    transcriptVideoId,
    visualFocus,
    visualGeometry,
  ]);

  function findPoint(clientX: number, clientY: number): ProjectedPoint | null {
    const canvas = canvasRef.current;
    if (!canvas) {
      return null;
    }

    const bounds = canvas.getBoundingClientRect();
    const x = clientX - bounds.left;
    const y = clientY - bounds.top;
    let nearest: ProjectedPoint | null = null;
    let nearestDistance = 100;

    const points = projectedFrameRef.current.points;
    for (let index = points.length - 1; index >= 0; index -= 1) {
      const item = points[index]!;
      if (item.culled) {
        continue;
      }

      const dx = item.x - x;
      const dy = item.y - y;
      const distance = dx * dx + dy * dy;
      if (distance < nearestDistance) {
        nearest = item;
        nearestDistance = distance;
      }
    }

    return nearest;
  }

  function handleSubmit(event: FormEvent<HTMLFormElement>): void {
    event.preventDefault();
    if (!query.trim()) {
      return;
    }

    pushHistory();
    setSelectedIslandId(null);
    onSearch();
  }

  function selectEntry(entryId: string | null): void {
    const clickedPoint = entryId ? pointById.get(entryId) : null;
    const stayInIsland = selectedIslandId !== null && clickedPoint?.clusterId === selectedIslandId;
    if (!stayInIsland) {
      setSelectedIslandId(null);
    }
    const differentVideo = transcriptVideoId && clickedPoint?.videoId !== transcriptVideoId;
    if (differentVideo) {
      onCloseTranscript();
    }
    onSelectEntry(entryId);
  }

  function pushHistory(): void {
    historyRef.current.push({
      query,
      searchResults,
      searchNote,
      errorText,
      selectedEntryId,
      selectedIslandId,
      transcriptVideoId,
      transcriptItems,
      transcriptFocusEntryId,
      transcriptLoading,
      transcriptErrorText,
    });
  }

  function selectIsland(cluster: SemanticLandscapeCluster): void {
    pushHistory();
    setSelectedIslandId(cluster.id);
    onSelectEntry(cluster.medoidEntryId);
  }

  function openTranscript(videoId: string, entryId: string): void {
    pushHistory();
    setSelectedIslandId(null);
    onOpenTranscript(videoId, entryId);
  }

  function searchLine(text: string): void {
    if (!text.trim()) {
      return;
    }

    pushHistory();
    setSelectedIslandId(null);
    onSearchLine(text);
  }

  function clearAtlas(): void {
    historyRef.current = [];
    setSelectedIslandId(null);
    onClear();
  }

  function handleBack(): void {
    const prev = historyRef.current.pop();
    if (prev) {
      const { selectedIslandId: prevIslandId, ...appState } = prev;
      setSelectedIslandId(prevIslandId);
      onRestoreNavigationState(appState);
    } else if (transcriptVideoId) {
      onCloseTranscript();
    } else {
      clearAtlas();
    }
  }

  function selectTranscriptCard(entryId: string): void {
    pushHistory();
    onSelectTranscriptEntry(entryId);
  }

  function handlePointerDown(event: React.PointerEvent<HTMLCanvasElement>): void {
    if (event.button !== 0) {
      return;
    }

    const canvas = event.currentTarget;
    canvas.setPointerCapture(event.pointerId);
    dragRef.current = {
      mode: event.ctrlKey || event.metaKey ? 'pan3d' : 'rotate3d',
      pointerId: event.pointerId,
      lastX: event.clientX,
      lastY: event.clientY,
      startX: event.clientX,
      startY: event.clientY,
      moved: false,
    };
    canvas.classList.add('is-dragging');
  }

  function handlePointerMove(event: React.PointerEvent<HTMLCanvasElement>): void {
    const drag = dragRef.current;
    if (drag) {
      const dx = event.clientX - drag.lastX;
      const dy = event.clientY - drag.lastY;
      drag.lastX = event.clientX;
      drag.lastY = event.clientY;
      drag.moved = drag.moved || Math.hypot(event.clientX - drag.startX, event.clientY - drag.startY) > 4;

      if (drag.mode === 'pan3d') {
        const current = view3dRef.current;
        current.offsetX += dx;
        current.offsetY += dy;
      } else {
        const current = view3dRef.current;
        current.rotateX = clamp(current.rotateX + dy * 0.005, -1.45, 1.45);
        current.rotateY += dx * 0.005;
      }
      requestAtlasRender();
      return;
    }

    const nearest = findPoint(event.clientX, event.clientY);
    const bounds = event.currentTarget.getBoundingClientRect();
    setHoverState(
      nearest
        ? {
            entryId: nearest.point.entryId,
            x: event.clientX - bounds.left,
            y: event.clientY - bounds.top,
          }
        : null,
    );
  }

  function handlePointerUp(event: React.PointerEvent<HTMLCanvasElement>): void {
    const drag = dragRef.current;
    event.currentTarget.classList.remove('is-dragging');

    if (!drag || drag.pointerId !== event.pointerId) {
      dragRef.current = null;
      return;
    }

    dragRef.current = null;
    if (!drag.moved) {
      const nearest = findPoint(event.clientX, event.clientY);
      if (nearest) {
        pushHistory();
        selectEntry(nearest.point.entryId);
      } else if (selectedEntryId || selectedIslandId) {
        pushHistory();
        selectEntry(null);
      }
    } else {
      setHoverState(null);
    }
  }

  function handlePointerLeave(): void {
    if (!dragRef.current) {
      setHoverState(null);
    }
  }

  function handleWheel(event: React.WheelEvent<HTMLCanvasElement>): void {
    event.preventDefault();
    manualZoomOverrideRef.current = true;
    const zoomFactor = Math.exp(-event.deltaY * 0.001);
    const current = view3dRef.current;
    current.zoom = clamp(current.zoom * zoomFactor, 0.42, 12);
    requestAtlasRender();
  }

  function resetView(): void {
    view3dRef.current = { ...effectiveInitialView };
    requestAtlasRender();
    setHoverState(null);
    clearAtlas();
  }

  function renderPairCard(
    entry: EntrySummary,
    props: Omit<Partial<PairCardProps>, 'entry' | 'onOpenTranscript' | 'onSearchLine'> = {},
  ) {
    return (
      <PairCard
        entry={entry}
        isFocus={entry.entryId === selectedEntryId}
        onSelect={selectEntry}
        onOpenTranscript={openTranscript}
        onSearchLine={searchLine}
        {...props}
      />
    );
  }

  return (
    <section
      className={classNames('atlas-shell', isTextMode && 'is-text-mode')}
      aria-label={isTextMode ? 'Translation memory text browser' : 'Translation memory atlas'}
    >
      <div ref={wrapRef} className="atlas-canvas-wrap">
        {!isTextMode && data ? (
          <canvas
            ref={canvasRef}
            className="atlas-canvas"
            onPointerDown={handlePointerDown}
            onPointerLeave={handlePointerLeave}
            onPointerMove={handlePointerMove}
            onPointerUp={handlePointerUp}
            onWheel={handleWheel}
          />
        ) : !isTextMode ? (
          <div className="atlas-loading">
            <strong>{dataErrorText ? 'Atlas unavailable' : dataLoading ? 'Loading UMAP atlas' : 'No atlas data'}</strong>
            <span>{dataErrorText ?? 'Preparing the browser visualization.'}</span>
          </div>
        ) : null}

        <div className="atlas-hud">
          <button className="atlas-title" type="button" onClick={clearAtlas} aria-label="Translation Memory">
            <span>Translation</span>
            <span>Memory</span>
          </button>

          <div className="atlas-hud-status" aria-label="Load status">
            {statusItems.map((item) => (
              <div
                key={item.label}
                className={classNames('atlas-status-item', item.ready && 'is-ready')}
                title={item.detail}
                style={{ ['--progress' as string]: String(item.progress) }}
              >
                <span>{item.label}</span>
                <strong>{item.value}</strong>
              </div>
            ))}
            {data ? (
              <div className="atlas-status-item is-ready">
                <span>Videos</span>
                <strong>{videoCount.toLocaleString()}</strong>
              </div>
            ) : null}
          </div>

          <button
            className="atlas-control"
            type="button"
            onClick={onToggleViewMode}
            aria-label={`Switch to ${isTextMode ? '3D atlas' : 'text-only'} view`}
            title={`Switch to ${isTextMode ? '3D atlas' : 'text-only'} view`}
          >
            <Icon name={isTextMode ? 'atlas' : 'text'} />
            <span>{isTextMode ? 'Atlas' : 'Text'}</span>
          </button>

          <button
            className="atlas-control"
            type="button"
            onClick={isTextMode ? clearAtlas : resetView}
            title={isTextMode ? 'Return to the island browser' : 'Reset atlas'}
            aria-label={isTextMode ? 'Return to the island browser' : 'Reset atlas'}
          >
            <Icon name="reset" />
            <span>{isTextMode ? 'Home' : 'Reset'}</span>
          </button>

          <button
            className="atlas-control"
            type="button"
            onClick={onToggleTheme}
            aria-label={`Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`}
            title={`Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`}
          >
            <Icon name={theme === 'dark' ? 'sun' : 'moon'} />
            <span>{theme === 'dark' ? 'Light' : 'Dark'}</span>
          </button>
        </div>

        {!isTextMode && hoveredPoint ? (
          <div
            className="atlas-tooltip"
            style={{
              left: `${Math.max(12, Math.min(hoverState!.x + 14, size.width - 372))}px`,
              top: `${Math.max(12, Math.min(hoverState!.y + 14, size.height - 118))}px`,
            }}
          >
            <strong>{hoveredPoint.videoId}#{hoveredPoint.segIndex}</strong>
            <span>{hoveredPoint.en}</span>
          </div>
        ) : null}
      </div>

      <aside className={classNames('atlas-sidebar', isIdle && 'is-idle', sidebarMode === 'transcript' && 'is-transcript')}>
        <form className="atlas-search" onSubmit={handleSubmit}>
          {canGoBack && (
            <button className="atlas-back-btn" type="button" onClick={handleBack} aria-label="Go back">
              <Icon name="back" />
            </button>
          )}
          <input
            aria-label="Search English subtitle lines"
            type="text"
            value={query}
            placeholder="Search English lines"
            onChange={(event) => {
              setSelectedIslandId(null);
              onQueryChange(event.target.value);
            }}
          />
          <button type="submit" disabled={searching || !searchReady || !query.trim()}>
            {searching ? '...' : 'Search'}
          </button>
        </form>

        <div ref={sidebarBodyRef} className="atlas-sidebar-body">
          {errorText ? <p className="atlas-message atlas-message--error">{errorText}</p> : null}
          {searchNote ? <p className="atlas-message">{searchNote}</p> : null}

          {sidebarMode === 'transcript' && (
            <section className="atlas-section atlas-video-transcript">
              <div className="transcript-heading">
                <span>YouTube ID</span>
                <h2>{transcriptVideoId}</h2>
                <p>
                  {transcriptLoading
                    ? 'Loading transcript cues...'
                    : `${transcriptItems.length.toLocaleString()} cues${
                        transcriptHasTimestamps ? ' with timestamps' : ''
                      }`}
                </p>
              </div>

              {transcriptLoading ? (
                <EmptyState>Loading the full transcript...</EmptyState>
              ) : transcriptErrorText ? (
                <EmptyState>{transcriptErrorText}</EmptyState>
              ) : (
                <div ref={transcriptBodyRef} className="transcript-panel-body">
                  <ol className="atlas-card-list">
                    {transcriptItems.map((item) => (
                      <li
                        key={item.entryId}
                        ref={(element) => {
                          if (element) transcriptItemRefs.current.set(item.entryId, element);
                          else transcriptItemRefs.current.delete(item.entryId);
                        }}
                      >
                        {renderPairCard(item, { onSelect: selectTranscriptCard })}
                      </li>
                    ))}
                  </ol>
                </div>
              )}
            </section>
          )}

          {sidebarMode === 'island' && selectedIslandPanel && (
            <section
              className="atlas-section atlas-island-focus"
              style={islandPanelStyle(selectedIslandPanel)}
            >
              <article className="atlas-island-card">
                <strong className="atlas-island-title">{selectedIslandPanel.cluster.label}</strong>

                <dl className="atlas-island-metrics">
                  {[
                    ['Lines', selectedIslandPanel.cluster.size.toLocaleString()],
                    ['Videos', selectedIslandPanel.cluster.videoCount.toLocaleString()],
                    ['Share', `${selectedIslandPanel.share.toFixed(1)}%`],
                    ['Compact', `${selectedIslandPanel.cluster.compactness}%`],
                  ].map(([label, value]) => (
                    <div key={label}>
                      <dt>{label}</dt>
                      <dd>{value}</dd>
                    </div>
                  ))}
                </dl>

                <div className="atlas-phrase-list">
                  {selectedIslandPanel.cluster.topPhrases.map((phrase) => (
                    <span key={phrase}>{phrase}</span>
                  ))}
                </div>
              </article>

              <SectionHeader title="Island Lines" count={selectedIslandPanel.entries.length.toLocaleString()} />
              <ol className="atlas-card-list is-scroll">
                {selectedIslandPanel.entries.map((entry) => (
                  <li key={entry.entryId}>
                    {renderPairCard(entry)}
                  </li>
                ))}
              </ol>
            </section>
          )}

          {sidebarMode === 'entry' && selectedEntry && (
            renderPairCard(selectedEntry, {
              isFocus: true,
              clusterColor: selectedCluster?.color,
              clusterLabel: selectedCluster?.label,
              onClusterClick: selectedCluster ? () => selectIsland(selectedCluster) : undefined,
            })
          )}

          {sidebarMode === 'search' && (
            <section className="atlas-section">
              <SectionHeader title="Semantic Matches" count={searchResults.length} />
              <ol className="atlas-card-list">
                {topSearchResults.map((result) => {
                  const point = pointById.get(result.entryId);
                  const cluster = point ? clusterById.get(point.clusterId) : null;
                  return (
                    <li key={result.entryId}>
                      {renderPairCard(result, {
                        score: result.score,
                        clusterColor: cluster?.color,
                        clusterLabel: cluster?.label,
                        onClusterClick: cluster ? () => selectIsland(cluster) : undefined,
                      })}
                    </li>
                  );
                })}
              </ol>
            </section>
          )}

          {sidebarMode === 'idle' && showIslandBrowser && (
            <section className="atlas-section atlas-island-section">
              <SectionHeader title="Islands" count={rankedIslands.length} />
              <ol className="atlas-island-list">
                {rankedIslands.map((cluster) => (
                  <li key={cluster.id}>
                    <button
                      className={classNames(
                        'atlas-island-row',
                        selectedIslandId === cluster.id && 'is-active',
                      )}
                      style={islandColorStyle(cluster.color)}
                      type="button"
                      onClick={() => selectIsland(cluster)}
                    >
                      <span className="atlas-island-dot" aria-hidden="true" />
                      <span className="atlas-island-copy">
                        <strong>{cluster.label}</strong>
                        <em>{cluster.description}</em>
                      </span>
                      <span className="atlas-island-count">{cluster.size.toLocaleString()}</span>
                    </button>
                  </li>
                ))}
              </ol>
            </section>
          )}
        </div>
      </aside>
    </section>
  );
}
