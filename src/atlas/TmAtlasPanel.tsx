import { FormEvent, useEffect, useMemo, useRef, useState } from 'react';
import { classNames } from '../classes';
import { handleSelectKey } from '../keyboard';
import type { ContextItem, SearchResult } from '../search/protocol';
import type {
  SemanticLandscapeCluster,
  SemanticLandscapeData,
  SemanticLandscapePoint,
} from './semantic-landscape';
import { hexToRgba, blendHexColors, isLightHex } from './colors';

type ThemeMode = 'dark' | 'light';

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
  statusItems: StatusItem[];
  query: string;
  searchReady: boolean;
  searching: boolean;
  searchResults: SearchResult[];
  searchNote: string | null;
  errorText: string | null;
  selectedEntryId: string | null;
  transcriptVideoId: string | null;
  transcriptItems: ContextItem[];
  transcriptFocusEntryId: string | null;
  transcriptLoading: boolean;
  transcriptErrorText: string | null;
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
}

export interface AtlasNavigationState {
  query: string;
  searchResults: SearchResult[];
  searchNote: string | null;
  errorText: string | null;
  selectedEntryId: string | null;
  transcriptVideoId: string | null;
  transcriptItems: ContextItem[];
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

interface Spatial3d {
  x3d: number;
  y3d: number;
  z3d: number;
}

interface RawProjected3d {
  point: SemanticLandscapePoint;
  cluster: SemanticLandscapeCluster;
  rawX: number;
  rawY: number;
  depth: number;
  culled: boolean;
}

interface ProjectedPoint {
  point: SemanticLandscapePoint;
  cluster: SemanticLandscapeCluster;
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
  active: boolean;
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

interface IconProps {
  className?: string;
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

function clamp(value: number, minimum: number, maximum: number): number {
  return Math.min(maximum, Math.max(minimum, value));
}

function percentile(values: number[], share: number): number {
  const sorted = [...values].sort((left, right) => left - right);
  const index = Math.min(sorted.length - 1, Math.max(0, Math.round((sorted.length - 1) * share)));
  return sorted[index] ?? 1;
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

function geometryWithFocus(geometry: VisualGeometry, focus: VisualFocus | null): VisualGeometry {
  if (!focus) {
    return geometry;
  }

  return {
    ...geometry,
    centerX: focus.centerX,
    centerY: focus.centerY,
    centerZ: focus.centerZ,
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

function project3dRaw(
  point: Spatial3d,
  geometry: VisualGeometry,
  trig: TrigCache,
) {
  const x = (point.x3d - geometry.centerX) / (geometry.radius * 2);
  const y = (geometry.centerY - point.y3d) / (geometry.radius * 2);
  const z = (point.z3d - geometry.centerZ) / (geometry.radius * 2);
  const x1 = x * trig.cosY + z * trig.sinY;
  const z1 = -x * trig.sinY + z * trig.cosY;
  const y1 = y * trig.cosX - z1 * trig.sinX;
  const z2 = y * trig.sinX + z1 * trig.cosX;
  const perspective = 1 / (1 + z2 * 0.72);

  return {
    rawX: x1 * perspective,
    rawY: y1 * perspective,
    depth: z2,
    culled: z2 < -1.1,
  };
}

function fitProjected3d(items: RawProjected3d[], width: number, height: number) {
  const visible = items.filter((item) => !item.culled);
  const count = Math.max(1, visible.length);
  const centerX = visible.reduce((total, item) => total + item.rawX, 0) / count;
  const centerY = visible.reduce((total, item) => total + item.rawY, 0) / count;
  const radius = Math.max(
    percentile(
      visible.map((item) => Math.hypot(item.rawX - centerX, item.rawY - centerY)),
      0.98,
    ),
    0.01,
  );
  const scale = (Math.min(width, height) * 0.44) / radius;

  return { centerX, centerY, scale };
}

function getDepthRadiusScale(depth: number): number {
  return clamp(1 - depth * 0.9, 0.55, 1.45);
}

function getCanvasRadiusScale(width: number, height: number): number {
  return clamp(Math.min(width, height) / 720, 0.5, 1);
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


function ResetIcon({ className }: IconProps) {
  return (
    <svg className={className} aria-hidden="true" viewBox="0 0 24 24">
      <path d="M5 12a7 7 0 1 0 2.1-5H4" />
      <path d="M4 3v4h4" />
    </svg>
  );
}


function SunIcon({ className }: IconProps) {
  return (
    <svg className={className} aria-hidden="true" viewBox="0 0 24 24">
      <circle cx="12" cy="12" r="4" />
      <path d="M12 2v2" />
      <path d="M12 20v2" />
      <path d="m4.9 4.9 1.4 1.4" />
      <path d="m17.7 17.7 1.4 1.4" />
      <path d="M2 12h2" />
      <path d="M20 12h2" />
      <path d="m4.9 19.1 1.4-1.4" />
      <path d="m17.7 6.3 1.4-1.4" />
    </svg>
  );
}

function MoonIcon({ className }: IconProps) {
  return (
    <svg className={className} aria-hidden="true" viewBox="0 0 24 24">
      <path d="M20 14.5A8 8 0 0 1 9.5 4a7 7 0 1 0 10.5 10.5Z" />
    </svg>
  );
}

function BackArrowIcon({ className }: IconProps) {
  return (
    <svg className={className} aria-hidden="true" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M19 12H5" />
      <path d="m12 5-7 7 7 7" />
    </svg>
  );
}

interface PairCardProps {
  entryId: string;
  videoId: string;
  segIndex: number;
  en: string;
  zh: string;
  isFocus?: boolean;
  onSelect: (entryId: string) => void;
  onOpenTranscript: (videoId: string, entryId: string) => void;
  onSearchLine: (text: string) => void;
  startMs?: number | null;
  endMs?: number | null;
  score?: number;
  clusterColor?: string;
  clusterLabel?: string;
  onClusterClick?: () => void;
}

function PairCard({
  entryId,
  videoId,
  segIndex,
  en,
  zh,
  isFocus,
  onSelect,
  onOpenTranscript,
  onSearchLine,
  startMs,
  endMs,
  score,
  clusterColor,
  clusterLabel,
  onClusterClick,
}: PairCardProps) {
  const hasTimestamps = startMs != null;
  const hasScore = score !== undefined;
  const hasCluster = onClusterClick !== undefined;

  return (
    <article
      className={classNames('pair-card', isFocus && 'is-focus')}
      role="button"
      tabIndex={0}
      onClick={() => onSelect(entryId)}
      onKeyDown={(event) => handleSelectKey(event, () => onSelect(entryId))}
    >
      {hasCluster && (
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

      {hasTimestamps && (
        <div className="pair-card-timestamps">
          <span>{formatCueTimestamp(startMs ?? null)}</span>
          <span className="ts-sep">–</span>
          <span>{formatCueTimestamp(endMs ?? null)}</span>
        </div>
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
        </div>

        {hasScore && <span className="pair-card-score">{score.toFixed(3)}</span>}
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
  statusItems,
  query,
  searchReady,
  searching,
  searchResults,
  searchNote,
  errorText,
  selectedEntryId,
  transcriptVideoId,
  transcriptItems,
  transcriptFocusEntryId,
  transcriptLoading,
  transcriptErrorText,
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
}: TmAtlasPanelProps) {
  const wrapRef = useRef<HTMLDivElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const dragRef = useRef<DragState | null>(null);
  const transcriptBodyRef = useRef<HTMLDivElement | null>(null);
  const transcriptItemRefs = useRef(new Map<string, HTMLLIElement>());
  const cameraAnimRef = useRef(0);
  const userNavigatingRef = useRef(false);
  const targetCenterRef = useRef<{ x: number; y: number; z: number } | null>(null);
  const animatedCenterRef = useRef<{ x: number; y: number; z: number } | null>(null);
  const historyRef = useRef<NavState[]>([]);
  const [size, setSize] = useState({ width: 1, height: 1 });
  const effectiveInitialView = useMemo<View3d>(
    () => data?.initialView ?? INITIAL_VIEW_3D,
    [data?.initialView],
  );
  const [view3d, setView3d] = useState<View3d>(INITIAL_VIEW_3D);
  const appliedDataViewRef = useRef(false);

  useEffect(() => {
    if (data?.initialView && !appliedDataViewRef.current) {
      appliedDataViewRef.current = true;
      setView3d(data.initialView);
    }
  }, [data?.initialView]);
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
  const hoveredPoint = hoverState ? pointById.get(hoverState.entryId)! : null;
  const visualFocus = useMemo<VisualFocus | null>(() => {
    if (showTranscriptPanel && selectedPoint) {
      return {
        key: `entry:${selectedPoint.entryId}`,
        centerX: selectedPoint.x3d,
        centerY: selectedPoint.y3d,
        centerZ: selectedPoint.z3d,
        zoom: 3.25,
      };
    }

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

    if (selectedPoint) {
      return {
        key: `entry:${selectedPoint.entryId}`,
        centerX: selectedPoint.x3d,
        centerY: selectedPoint.y3d,
        centerZ: selectedPoint.z3d,
        zoom: 3.25,
      };
    }

    return null;
  }, [selectedIslandPanel, selectedPoint, showTranscriptPanel]);

  function updateTargetCenter(focus: VisualFocus | null, base: VisualGeometry): void {
    const target = focus
      ? { x: focus.centerX, y: focus.centerY, z: focus.centerZ }
      : { x: base.centerX, y: base.centerY, z: base.centerZ };
    targetCenterRef.current = target;
    if (!animatedCenterRef.current) {
      animatedCenterRef.current = { ...target };
    }
  }

  const projectionGeometry = useMemo(() => {
    const center = animatedCenterRef.current;
    if (!center) {
      return geometryWithFocus(visualGeometry, visualFocus);
    }
    return { ...visualGeometry, centerX: center.x, centerY: center.y, centerZ: center.z };
  }, [visualGeometry, visualFocus, view3d]);
  const projectedPoints = useMemo<ProjectedPoint[]>(() => {
    if (!data) {
      return [];
    }

    const trig = computeTrig(view3d);
    const rawItems = data.points.map((point): RawProjected3d => ({
      point,
      cluster: clusterById.get(point.clusterId)!,
      ...project3dRaw(point, projectionGeometry, trig),
    }));
    const sidebarGutter = size.width >= DESKTOP_ATLAS_MIN_WIDTH ? DESKTOP_SIDEBAR_WIDTH : 0;
    const topGutter = size.width <= MOBILE_ATLAS_MAX_WIDTH ? MOBILE_HUD_SAFE_TOP : 0;
    const plotWidth = Math.max(1, size.width - sidebarGutter);
    const plotHeight = Math.max(1, size.height - topGutter);
    const fitted = fitProjected3d(rawItems, plotWidth, plotHeight);
    const centerX = visualFocus ? 0 : fitted.centerX;
    const centerY = visualFocus ? 0 : fitted.centerY;
    const points = rawItems.map((item) => ({
      point: item.point,
      cluster: item.cluster,
      x: (item.rawX - centerX) * fitted.scale * view3d.zoom + plotWidth / 2 + view3d.offsetX,
      y: (item.rawY - centerY) * fitted.scale * view3d.zoom + topGutter + plotHeight / 2 + view3d.offsetY,
      depth: item.depth,
      culled: item.culled,
    }));

    points.sort((left, right) => right.depth - left.depth);
    return points;
  }, [clusterById, data, projectionGeometry, size.height, size.width, view3d, visualFocus]);
  const projectedIslandById = useMemo(() => {
    const totals = new Map<number, { x: number; y: number; depth: number; count: number }>();

    for (const item of projectedPoints) {
      if (item.culled) {
        continue;
      }

      const total = totals.get(item.point.clusterId);
      if (total) {
        total.x += item.x;
        total.y += item.y;
        total.depth += item.depth;
        total.count += 1;
      } else {
        totals.set(item.point.clusterId, { x: item.x, y: item.y, depth: item.depth, count: 1 });
      }
    }

    const islands = new Map<number, ProjectedIsland>();
    for (const [clusterId, total] of totals) {
      const cluster = clusterById.get(clusterId);
      if (!cluster) {
        continue;
      }

      islands.set(clusterId, {
        cluster,
        x: total.x / total.count,
        y: total.y / total.count,
        depth: total.depth / total.count,
      });
    }

    return islands;
  }, [clusterById, projectedPoints]);
  const transcriptHasTimestamps = transcriptItems.some((item) => item.startMs !== null || item.endMs !== null);

  const isInSearchResults = selectedEntryId ? searchResultById.has(selectedEntryId) : false;
  const sidebarMode: SidebarMode = showTranscriptPanel
    ? 'transcript'
    : selectedIslandPanel
      ? 'island'
      : selectedEntry && (!searchResults.length || !isInSearchResults)
        ? 'entry'
        : searchResults.length > 0
          ? 'search'
          : 'idle';
  const topSearchResults = useMemo(() => searchResults.slice(0, 12), [searchResults]);

  const showIslandBrowser = !!data && !query.trim();
  const isIdle = sidebarMode === 'idle' && !errorText && !searchNote;
  const canGoBack = sidebarMode !== 'idle' || historyRef.current.length > 0;

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

      setSize({
        width: Math.max(1, Math.floor(entry.contentRect.width)),
        height: Math.max(1, Math.floor(entry.contentRect.height)),
      });
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
    if (!data?.points.length) {
      return;
    }

    const targetZoom = visualFocus ? visualFocus.zoom : effectiveInitialView.zoom;
    updateTargetCenter(visualFocus, visualGeometry);

    const lerp = (a: number, b: number, t: number) => a + (b - a) * t;
    const epsilon = 0.001;
    let running = true;
    let settled = false;

    function step() {
      if (!running || settled) return;

      const center = animatedCenterRef.current;
      const target = targetCenterRef.current;
      if (center && target) {
        center.x = lerp(center.x, target.x, 0.1);
        center.y = lerp(center.y, target.y, 0.1);
        center.z = lerp(center.z, target.z, 0.1);
      }

      setView3d((current) => {
        const dx = Math.abs(current.offsetX) + Math.abs(current.offsetY);
        const zooming = userNavigatingRef.current;
        const dz = zooming ? 0 : Math.abs(current.zoom - targetZoom);
        const cx = center && target
          ? Math.abs(center.x - target.x) + Math.abs(center.y - target.y) + Math.abs(center.z - target.z)
          : 0;
        if (dx + dz + cx < epsilon) {
          settled = true;
          userNavigatingRef.current = false;
          return { ...current, zoom: zooming ? current.zoom : targetZoom, offsetX: 0, offsetY: 0 };
        }
        return {
          ...current,
          zoom: zooming ? current.zoom : lerp(current.zoom, targetZoom, 0.1),
          offsetX: lerp(current.offsetX, 0, 0.1),
          offsetY: lerp(current.offsetY, 0, 0.1),
        };
      });

      if (!settled) {
        cameraAnimRef.current = requestAnimationFrame(step);
      }
    }

    settled = false;
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
  ]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !data) {
      return;
    }

    const context = canvas.getContext('2d');
    if (!context) {
      return;
    }

    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.round(size.width * dpr);
    canvas.height = Math.round(size.height * dpr);
    canvas.style.width = `${size.width}px`;
    canvas.style.height = `${size.height}px`;
    context.setTransform(dpr, 0, 0, dpr, 0, 0);
    context.clearRect(0, 0, size.width, size.height);

    const background = theme === 'dark' ? '#06100b' : '#f5f0e8';
    const grid = theme === 'dark' ? 'rgba(197, 228, 203, 0.08)' : 'rgba(51, 104, 72, 0.12)';
    const muted = theme === 'dark' ? '#9fb2a5' : '#617266';
    const text = theme === 'dark' ? '#edf8ef' : '#17251b';
    const selected = theme === 'dark' ? '#4ade80' : '#16a34a';

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

    const projectedByEntryId = new Map(projectedPoints.map((item) => [item.point.entryId, item]));
    const rankedSearchHits = topSearchResults
      .map((result) => projectedByEntryId.get(result.entryId))
      .filter((item): item is ProjectedPoint => !!item && !item.culled);
    const allVideoPathPoints = transcriptVideoId
      ? transcriptItems
          .map((item) => projectedByEntryId.get(item.entryId))
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
    const pointRadiusScale = getCanvasRadiusScale(size.width, size.height);

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
      const depthRadiusScale = getDepthRadiusScale(item.depth);
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
  }, [
    data,
    hoverState?.entryId,
    islandFlows,
    projectedIslandById,
    projectedPoints,
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
    view3d.offsetX,
    view3d.offsetY,
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

    for (let index = projectedPoints.length - 1; index >= 0; index -= 1) {
      const item = projectedPoints[index]!;
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

  function createHistoryState(): NavState {
    return {
      query,
      searchResults: [...searchResults],
      searchNote,
      errorText,
      selectedEntryId,
      selectedIslandId,
      transcriptVideoId,
      transcriptItems: [...transcriptItems],
      transcriptFocusEntryId,
      transcriptLoading,
      transcriptErrorText,
    };
  }

  function pushHistory(): void {
    historyRef.current.push(createHistoryState());
  }

  function restoreHistoryState(state: NavState): void {
    const { selectedIslandId: nextSelectedIslandId, ...appState } = state;
    setSelectedIslandId(nextSelectedIslandId);
    onRestoreNavigationState(appState);
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
      restoreHistoryState(prev);
    } else if (transcriptVideoId) {
      onCloseTranscript();
    } else {
      clearAtlas();
    }
  }

  function setTranscriptItemRef(entryId: string, element: HTMLLIElement | null): void {
    if (element) {
      transcriptItemRefs.current.set(entryId, element);
      return;
    }

    transcriptItemRefs.current.delete(entryId);
  }

  function handlePointerDown(event: React.PointerEvent<HTMLCanvasElement>): void {
    if (event.button !== 0) {
      return;
    }

    const canvas = event.currentTarget;
    canvas.setPointerCapture(event.pointerId);
    dragRef.current = {
      active: true,
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
    if (drag?.active) {
      const dx = event.clientX - drag.lastX;
      const dy = event.clientY - drag.lastY;
      drag.lastX = event.clientX;
      drag.lastY = event.clientY;
      drag.moved = drag.moved || Math.hypot(event.clientX - drag.startX, event.clientY - drag.startY) > 4;
      userNavigatingRef.current = true;

      if (drag.mode === 'pan3d') {
        setView3d((current) => ({
          ...current,
          offsetX: current.offsetX + dx,
          offsetY: current.offsetY + dy,
        }));
      } else {
        setView3d((current) => ({
          ...current,
          rotateX: clamp(current.rotateX + dy * 0.005, -1.45, 1.45),
          rotateY: current.rotateY + dx * 0.005,
        }));
      }
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

    if (!drag?.active || drag.pointerId !== event.pointerId) {
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
    if (!dragRef.current?.active) {
      setHoverState(null);
    }
  }

  function handleWheel(event: React.WheelEvent<HTMLCanvasElement>): void {
    event.preventDefault();
    userNavigatingRef.current = true;
    const zoomFactor = Math.exp(-event.deltaY * 0.001);

    setView3d((current) => ({
      ...current,
      zoom: clamp(current.zoom * zoomFactor, 0.42, 12),
    }));
  }

  function resetView(): void {
    setView3d(effectiveInitialView);
    setHoverState(null);
    clearAtlas();
  }

  return (
    <section className="atlas-shell" aria-label="Translation memory atlas">
      <div ref={wrapRef} className="atlas-canvas-wrap">
        {data ? (
          <canvas
            ref={canvasRef}
            className="atlas-canvas"
            onPointerDown={handlePointerDown}
            onPointerLeave={handlePointerLeave}
            onPointerMove={handlePointerMove}
            onPointerUp={handlePointerUp}
            onWheel={handleWheel}
          />
        ) : (
          <div className="atlas-loading">
            <strong>{dataErrorText ? 'Atlas unavailable' : dataLoading ? 'Loading UMAP atlas' : 'No atlas data'}</strong>
            <span>{dataErrorText ?? 'Preparing the browser visualization.'}</span>
          </div>
        )}

        <div className="atlas-hud">
          <button className="atlas-title" type="button" onClick={clearAtlas}>
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

          <button className="atlas-icon-button" type="button" onClick={resetView} title="Reset atlas" aria-label="Reset atlas">
            <ResetIcon className="atlas-button-icon" />
            <span>Reset</span>
          </button>

          <button
            className={classNames('atlas-theme-toggle', theme === 'dark' && 'is-dark')}
            type="button"
            onClick={onToggleTheme}
            role="switch"
            aria-checked={theme === 'dark'}
            aria-label={`Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`}
            title={`Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`}
          >
            <span className="atlas-theme-toggle-track" aria-hidden="true">
              <SunIcon className="atlas-theme-icon atlas-theme-icon--sun" />
              <MoonIcon className="atlas-theme-icon atlas-theme-icon--moon" />
            </span>
          </button>
        </div>

        {hoveredPoint ? (
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
              <BackArrowIcon />
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

        {errorText ? <p className="atlas-message atlas-message--error">{errorText}</p> : null}
        {searchNote ? <p className="atlas-message">{searchNote}</p> : null}

        {sidebarMode === 'transcript' && (
          <section className="atlas-section atlas-video-transcript">
            <div className="atlas-video-header">
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
            </div>

            {transcriptLoading ? (
              <div className="empty-state">
                <p>Loading the full transcript...</p>
              </div>
            ) : transcriptErrorText ? (
              <div className="empty-state">
                <p>{transcriptErrorText}</p>
              </div>
            ) : (
              <div ref={transcriptBodyRef} className="transcript-panel-body">
                <ol className="transcript-row-list">
                  {transcriptItems.map((item) => (
                    <li
                      key={item.entryId}
                      ref={(element) => setTranscriptItemRef(item.entryId, element)}
                    >
                      <PairCard
                        entryId={item.entryId}
                        videoId={item.videoId}
                        segIndex={item.segIndex}
                        en={item.en}
                        zh={item.zh}
                        isFocus={item.entryId === selectedEntryId}
                        onSelect={(entryId) => {
                          pushHistory();
                          onSelectTranscriptEntry(entryId);
                        }}
                        onOpenTranscript={openTranscript}
                        onSearchLine={searchLine}
                        startMs={item.startMs}
                        endMs={item.endMs}
                      />
                    </li>
                  ))}
                </ol>
              </div>
            )}
          </section>
        )}

        {sidebarMode === 'island' && selectedIslandPanel && (() => {
          const light = isLightHex(selectedIslandPanel.cluster.color);
          return (
          <section
            className="atlas-section atlas-island-focus"
            style={{
              ['--cluster-color' as string]: selectedIslandPanel.cluster.color,
              ['--island-text' as string]: light ? '#020a06' : '#e2f5ea',
              ['--island-text-muted' as string]: light ? 'rgba(2, 10, 6, 0.6)' : 'rgba(226, 245, 234, 0.6)',
              ['--island-text-strong' as string]: light ? 'rgba(2, 10, 6, 0.75)' : 'rgba(226, 245, 234, 0.75)',
              ['--island-overlay' as string]: light ? 'rgba(0, 0, 0, 0.06)' : 'rgba(255, 255, 255, 0.1)',
              ['--island-overlay-border' as string]: light ? 'rgba(0, 0, 0, 0.1)' : 'rgba(255, 255, 255, 0.15)',
            }}
          >
            <div className="atlas-island-sticky">
              <article className="atlas-island-card">
                <div className="atlas-island-heading">
                  <strong>{selectedIslandPanel.cluster.label}</strong>
                </div>
                <p className="atlas-region-description">{selectedIslandPanel.cluster.description}</p>

                <dl className="atlas-island-metrics">
                  <div>
                    <dt>Lines</dt>
                    <dd>{selectedIslandPanel.cluster.size.toLocaleString()}</dd>
                  </div>
                  <div>
                    <dt>Videos</dt>
                    <dd>{selectedIslandPanel.cluster.videoCount.toLocaleString()}</dd>
                  </div>
                  <div>
                    <dt>Share</dt>
                    <dd>{selectedIslandPanel.share.toFixed(1)}%</dd>
                  </div>
                  <div>
                    <dt>Compact</dt>
                    <dd>{selectedIslandPanel.cluster.compactness}%</dd>
                  </div>
                </dl>

                <div className="atlas-phrase-list">
                  {selectedIslandPanel.cluster.topPhrases.map((phrase) => (
                    <span key={phrase}>{phrase}</span>
                  ))}
                </div>
              </article>
            </div>

            <div className="atlas-section-header">
              <strong>Island Lines</strong>
              <span>{selectedIslandPanel.entries.length.toLocaleString()}</span>
            </div>
            <ol className="atlas-island-entry-list">
              {selectedIslandPanel.entries.map((entry) => (
                <li key={entry.entryId}>
                  <PairCard
                    entryId={entry.entryId}
                    videoId={entry.videoId}
                    segIndex={entry.segIndex}
                    en={entry.en}
                    zh={entry.zh}
                    isFocus={entry.entryId === selectedEntryId}
                    onSelect={selectEntry}
                    onOpenTranscript={openTranscript}
                    onSearchLine={searchLine}
                  />
                </li>
              ))}
            </ol>
          </section>
          );
        })()}

        {sidebarMode === 'entry' && selectedEntry && (
          <div className="atlas-detail">
            <PairCard
              entryId={selectedEntry.entryId}
              videoId={selectedEntry.videoId}
              segIndex={selectedEntry.segIndex}
              en={selectedEntry.en}
              zh={selectedEntry.zh}
              isFocus
              onSelect={selectEntry}
              onOpenTranscript={openTranscript}
              onSearchLine={searchLine}
              clusterColor={selectedCluster?.color}
              clusterLabel={selectedCluster?.label}
              onClusterClick={selectedCluster ? () => selectIsland(selectedCluster) : undefined}
            />
          </div>
        )}

        {sidebarMode === 'search' && (
          <section className="atlas-section">
            <div className="atlas-section-header">
              <strong>Semantic Matches</strong>
              <span>{searchResults.length}</span>
            </div>
            <ol className="atlas-result-list">
              {topSearchResults.map((result) => {
                const point = pointById.get(result.entryId);
                const cluster = point ? clusterById.get(point.clusterId) : null;
                return (
                  <li key={result.entryId}>
                    <PairCard
                      entryId={result.entryId}
                      videoId={result.videoId}
                      segIndex={result.segIndex}
                      en={result.en}
                      zh={result.zh}
                      isFocus={result.entryId === selectedEntryId}
                      onSelect={selectEntry}
                      onOpenTranscript={openTranscript}
                      onSearchLine={searchLine}
                      score={result.score}
                      clusterColor={cluster?.color}
                      clusterLabel={cluster?.label}
                      onClusterClick={cluster ? () => selectIsland(cluster) : undefined}
                    />
                  </li>
                );
              })}
            </ol>
          </section>
        )}

        {sidebarMode === 'idle' && showIslandBrowser && (
          <section className="atlas-section atlas-island-section">
            <div className="atlas-section-header">
              <strong>Islands</strong>
              <span>{rankedIslands.length}</span>
            </div>
            <ol className="atlas-island-list">
              {rankedIslands.map((cluster) => (
                <li key={cluster.id}>
                  <button
                    className={classNames(
                      'atlas-island-row',
                      selectedIslandId === cluster.id && 'is-active',
                    )}
                    style={{ ['--cluster-color' as string]: cluster.color }}
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
      </aside>
    </section>
  );
}
