import { FormEvent, useEffect, useMemo, useRef, useState } from 'react';
import { classNames } from '../classes';
import { handleSelectKey } from '../keyboard';
import type { ContextItem, SearchResult } from '../search/protocol';
import type {
  SemanticLandscapeCluster,
  SemanticLandscapeData,
  SemanticLandscapePoint,
} from './semantic-landscape';
import { hexToRgba } from './colors';

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
  contextItems: ContextItem[];
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
  onSearchTranscriptLine: (line: string) => void;
  onCloseTranscript: () => void;
  onClear: () => void;
  onToggleTheme: () => void;
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

function project3dRaw(
  point: SemanticLandscapePoint,
  view: View3d,
  geometry: VisualGeometry,
) {
  const x = (point.x3d - geometry.centerX) / (geometry.radius * 2);
  const y = (geometry.centerY - point.y3d) / (geometry.radius * 2);
  const z = (point.z3d - geometry.centerZ) / (geometry.radius * 2);
  const cosY = Math.cos(view.rotateY);
  const sinY = Math.sin(view.rotateY);
  const cosX = Math.cos(view.rotateX);
  const sinX = Math.sin(view.rotateX);
  const x1 = x * cosY + z * sinY;
  const z1 = -x * sinY + z * cosY;
  const y1 = y * cosX - z1 * sinX;
  const z2 = y * sinX + z1 * cosX;
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

function getEntryText(entry: SemanticLandscapePoint | SearchResult | null): string {
  return entry?.en.trim() ?? '';
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

function formatCueRange(startMs: number | null, endMs: number | null): string {
  if (startMs === null && endMs === null) {
    return 'No timestamps';
  }

  if (startMs === null) {
    return `Ends ${formatCueTimestamp(endMs)}`;
  }

  if (endMs === null) {
    return `Starts ${formatCueTimestamp(startMs)}`;
  }

  return `${formatCueTimestamp(startMs)} - ${formatCueTimestamp(endMs)}`;
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
  contextItems,
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
  onSearchTranscriptLine,
  onCloseTranscript,
  onClear,
  onToggleTheme,
}: TmAtlasPanelProps) {
  const wrapRef = useRef<HTMLDivElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const dragRef = useRef<DragState | null>(null);
  const transcriptBodyRef = useRef<HTMLDivElement | null>(null);
  const transcriptItemRefs = useRef(new Map<string, HTMLLIElement>());
  const [size, setSize] = useState({ width: 1, height: 1 });
  const [view3d, setView3d] = useState<View3d>(INITIAL_VIEW_3D);
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
  const videoCount = useMemo(
    () => new Set(data?.points.map((point) => point.videoId) ?? []).size,
    [data],
  );

  const selectedPoint = selectedEntryId ? pointById.get(selectedEntryId) ?? null : null;
  const selectedSearchResult = selectedEntryId ? searchResultById.get(selectedEntryId) ?? null : null;
  const selectedEntry = selectedPoint ?? selectedSearchResult ?? null;
  const selectedCluster = selectedPoint ? clusterById.get(selectedPoint.clusterId)! : null;
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
  }, [selectedIslandPanel, selectedPoint]);
  const projectionGeometry = useMemo(
    () => geometryWithFocus(visualGeometry, visualFocus),
    [visualFocus, visualGeometry],
  );
  const projectedPoints = useMemo<ProjectedPoint[]>(() => {
    if (!data) {
      return [];
    }

    const rawItems = data.points.map((point): RawProjected3d => ({
      point,
      cluster: clusterById.get(point.clusterId)!,
      ...project3dRaw(point, view3d, projectionGeometry),
    }));
    const fitted = fitProjected3d(rawItems, size.width, size.height);
    const centerX = visualFocus ? 0 : fitted.centerX;
    const centerY = visualFocus ? 0 : fitted.centerY;
    const points = rawItems.map((item) => ({
      point: item.point,
      cluster: item.cluster,
      x: (item.rawX - centerX) * fitted.scale * view3d.zoom + size.width / 2 + view3d.offsetX,
      y: (item.rawY - centerY) * fitted.scale * view3d.zoom + size.height / 2 + view3d.offsetY,
      depth: item.depth,
      culled: item.culled,
    }));

    points.sort((left, right) => left.depth - right.depth);
    return points;
  }, [clusterById, data, projectionGeometry, size.height, size.width, view3d, visualFocus]);
  const showTranscriptPanel = !!transcriptVideoId;
  const transcriptHasTimestamps = transcriptItems.some((item) => item.startMs !== null || item.endMs !== null);
  const showIslandBrowser =
    !!data && searchResults.length === 0 && !query.trim() && !selectedIslandPanel && !showTranscriptPanel;
  const showLocalContext = !showTranscriptPanel && !selectedIslandPanel && (!showIslandBrowser || !!selectedEntry);
  const idleSidebar = showIslandBrowser && !selectedEntry && !errorText && !searchNote;

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
    if (!visualFocus) {
      setView3d((current) => ({
        ...current,
        zoom: INITIAL_VIEW_3D.zoom,
        offsetX: 0,
        offsetY: 0,
      }));
      return;
    }

    setView3d((current) => ({
      ...current,
      zoom: visualFocus.zoom,
      offsetX: 0,
      offsetY: 0,
    }));
  }, [visualFocus?.key]);

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

    const background = theme === 'dark' ? '#06100b' : '#f2f8f1';
    const grid = theme === 'dark' ? 'rgba(197, 228, 203, 0.08)' : 'rgba(51, 104, 72, 0.12)';
    const muted = theme === 'dark' ? '#9fb2a5' : '#617266';
    const text = theme === 'dark' ? '#edf8ef' : '#17251b';
    const computedStyle = window.getComputedStyle(document.documentElement);
    const selected = computedStyle.getPropertyValue('--accent').trim();

    context.fillStyle = background;
    context.fillRect(0, 0, size.width, size.height);

    context.strokeStyle = grid;
    context.lineWidth = 1;
    const gridStep = Math.max(80, Math.round(Math.min(size.width, size.height) / 7));
    for (let x = (view3d.offsetX % gridStep) - gridStep; x < size.width + gridStep; x += gridStep) {
      context.beginPath();
      context.moveTo(x, 0);
      context.lineTo(x, size.height);
      context.stroke();
    }
    for (let y = (view3d.offsetY % gridStep) - gridStep; y < size.height + gridStep; y += gridStep) {
      context.beginPath();
      context.moveTo(0, y);
      context.lineTo(size.width, y);
      context.stroke();
    }

    const projectedByEntryId = new Map(projectedPoints.map((item) => [item.point.entryId, item]));
    const rankedSearchHits = searchResults
      .slice(0, 12)
      .map((result) => projectedByEntryId.get(result.entryId))
      .filter((item): item is ProjectedPoint => !!item && !item.culled);
    const videoPathPoints = transcriptVideoId
      ? transcriptItems
          .map((item) => projectedByEntryId.get(item.entryId))
          .filter((item): item is ProjectedPoint => !!item && !item.culled)
      : [];
    const videoPathEntryIds = new Set(videoPathPoints.map((item) => item.point.entryId));
    const selectedVideoPathIndex = videoPathPoints.findIndex((item) => item.point.entryId === selectedEntryId);
    const emphasizedVideoPathIds = new Set(
      [selectedVideoPathIndex - 1, selectedVideoPathIndex, selectedVideoPathIndex + 1]
        .filter((index) => index >= 0 && index < videoPathPoints.length)
        .map((index) => videoPathPoints[index]!.point.entryId),
    );
    const videoPathFocusColor = selectedPoint?.color ?? videoPathPoints[0]?.point.color ?? selected;

    if (rankedSearchHits.length > 1) {
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

      if (selectedVideoPathIndex >= 0) {
        context.strokeStyle = hexToRgba(videoPathFocusColor, 0.72);
        context.lineWidth = 2;
        const highlightedSegments: Array<[number, number]> = [
          [selectedVideoPathIndex - 1, selectedVideoPathIndex],
          [selectedVideoPathIndex, selectedVideoPathIndex + 1],
        ];
        for (const [fromIndex, toIndex] of highlightedSegments) {
          const fromPoint = videoPathPoints[fromIndex];
          const toPoint = videoPathPoints[toIndex];
          if (fromPoint && toPoint) {
            context.beginPath();
            context.moveTo(fromPoint.x, fromPoint.y);
            context.lineTo(toPoint.x, toPoint.y);
            context.stroke();
          }
        }
      }
      context.restore();
    }

    for (const item of projectedPoints) {
      if (item.culled) {
        continue;
      }

      const isSelected = item.point.entryId === selectedEntryId;
      const isHovered = item.point.entryId === hoverState?.entryId;
      const isSearchHit = searchHitIds.has(item.point.entryId);
      const isVideoPathPoint = videoPathEntryIds.has(item.point.entryId);
      const isEmphasizedVideoPathPoint = emphasizedVideoPathIds.has(item.point.entryId);
      const hasSearch = searchHitIds.size > 0;
      const hasVideoPath = videoPathEntryIds.size > 0;
      const isOutsideSelectedIsland = selectedIslandId !== null && item.point.clusterId !== selectedIslandId;
      const radius = isSelected
        ? 5.2
        : isHovered
          ? 4.4
          : isSearchHit || isEmphasizedVideoPathPoint
            ? 3.4
            : isVideoPathPoint
              ? 2.7
              : 2.35;
      const alpha = isSelected
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
                : selectedIslandId !== null
                  ? 0.68
                  : 0.58;

      context.fillStyle = hexToRgba(isSearchHit && !isSelected ? selected : item.point.color, alpha);
      context.beginPath();
      context.arc(item.x, item.y, radius, 0, Math.PI * 2);
      context.fill();

      if (isEmphasizedVideoPathPoint && !isSelected && !isHovered) {
        context.strokeStyle = hexToRgba(videoPathFocusColor, 0.45);
        context.lineWidth = 1;
        context.beginPath();
        context.arc(item.x, item.y, radius + 2.5, 0, Math.PI * 2);
        context.stroke();
      }

      if (isSelected || isHovered) {
        context.strokeStyle = hexToRgba(text, isSelected ? 0.92 : 0.7);
        context.lineWidth = isSelected ? 1.8 : 1.2;
        context.beginPath();
        context.arc(item.x, item.y, radius + 3, 0, Math.PI * 2);
        context.stroke();
      }
    }

    if (rankedSearchHits.length > 0) {
      context.save();
      context.strokeStyle = hexToRgba(selected, 0.9);
      context.lineWidth = 1.5;
      for (let index = 0; index < rankedSearchHits.length; index += 1) {
        const item = rankedSearchHits[index]!;
        context.beginPath();
        context.arc(item.x, item.y, 8 + Math.min(index, 5) * 0.55, 0, Math.PI * 2);
        context.stroke();
      }
      context.restore();
    }
  }, [
    data,
    hoverState?.entryId,
    projectedPoints,
    searchHitIds,
    searchResults,
    selectedEntryId,
    selectedIslandId,
    selectedPoint?.color,
    size.height,
    size.width,
    theme,
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
    setSelectedIslandId(null);
    onSearch();
  }

  function selectEntry(entryId: string | null): void {
    setSelectedIslandId(null);
    onSelectEntry(entryId);
  }

  function selectIsland(cluster: SemanticLandscapeCluster): void {
    setSelectedIslandId(cluster.id);
    onSelectEntry(cluster.medoidEntryId);
  }

  function clearAtlas(): void {
    setSelectedIslandId(null);
    onClear();
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
      selectEntry(nearest?.point.entryId ?? null);
    }
  }

  function handlePointerLeave(): void {
    if (!dragRef.current?.active) {
      setHoverState(null);
    }
  }

  function handleWheel(event: React.WheelEvent<HTMLCanvasElement>): void {
    event.preventDefault();
    const zoomFactor = Math.exp(-event.deltaY * 0.001);

    setView3d((current) => ({
      ...current,
      zoom: clamp(current.zoom * zoomFactor, 0.42, 12),
    }));
  }

  function resetView(): void {
    setView3d(INITIAL_VIEW_3D);
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

      <aside className={classNames('atlas-sidebar', idleSidebar && 'is-idle', showTranscriptPanel && 'is-transcript')}>
        <form className="atlas-search" onSubmit={handleSubmit}>
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

        {showTranscriptPanel ? (
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

              <button className="atlas-panel-close" type="button" onClick={onCloseTranscript}>
                Close
              </button>
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
                  {transcriptItems.map((item) => {
                    const isSelected = item.entryId === selectedEntryId;

                    return (
                      <li
                        key={item.entryId}
                        ref={(element) => setTranscriptItemRef(item.entryId, element)}
                        className="transcript-row"
                      >
                        <button
                          aria-label={`Jump to ${item.videoId} cue ${item.segIndex}. ${formatCueRange(
                            item.startMs,
                            item.endMs,
                          )}.`}
                          className={classNames('transcript-time', isSelected && 'is-selected')}
                          type="button"
                          onClick={() => onSelectTranscriptEntry(item.entryId)}
                        >
                          <span>#{item.segIndex}</span>
                          <strong>{formatCueTimestamp(item.startMs)}</strong>
                          <em>{formatCueTimestamp(item.endMs)}</em>
                        </button>

                        <article
                          className={classNames('transcript-entry', isSelected && 'is-selected')}
                          role="button"
                          tabIndex={0}
                          onClick={() => onSelectTranscriptEntry(item.entryId)}
                          onKeyDown={(event) => handleSelectKey(event, () => onSelectTranscriptEntry(item.entryId))}
                        >
                          <div className="transcript-entry-meta">
                            <span>{item.videoId}#{item.segIndex}</span>
                            <span>{item.blockName || 'no block'}</span>
                          </div>
                          <div className="transcript-entry-copy-line">
                            <p className="transcript-entry-line transcript-entry-line--en">{item.en}</p>
                            <button
                              aria-label={`Search using cue ${item.segIndex} English text`}
                              className="transcript-entry-action"
                              type="button"
                              disabled={!item.en.trim()}
                              onClick={(event) => {
                                event.stopPropagation();
                                onSearchTranscriptLine(item.en);
                              }}
                            >
                              Search
                            </button>
                          </div>
                          <p className="transcript-entry-line transcript-entry-line--zh">{item.zh}</p>
                        </article>
                      </li>
                    );
                  })}
                </ol>
              </div>
            )}
          </section>
        ) : selectedIslandPanel ? (
          <section
            className="atlas-section atlas-island-focus"
            style={{ ['--cluster-color' as string]: selectedIslandPanel.cluster.color }}
          >
            <article className="atlas-island-card">
              <div className="atlas-island-heading">
                <strong>{selectedIslandPanel.cluster.label}</strong>
                <em>{selectedIslandPanel.cluster.labelMode}</em>
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
                  <dt>Theme</dt>
                  <dd>{Math.round(selectedIslandPanel.cluster.labelConfidence * 100)}%</dd>
                </div>
              </dl>

              <div className="atlas-phrase-list">
                {selectedIslandPanel.cluster.topPhrases.map((phrase) => (
                  <span key={phrase}>{phrase}</span>
                ))}
              </div>
            </article>

            <div className="atlas-section-header">
              <strong>Island Lines</strong>
              <span>{selectedIslandPanel.entries.length.toLocaleString()}</span>
            </div>
            <ol className="atlas-island-entry-list">
              {selectedIslandPanel.entries.map((entry) => (
                <li key={entry.entryId}>
                  <article
                    className={classNames('atlas-island-entry', entry.entryId === selectedEntryId && 'is-focus')}
                    role="button"
                    tabIndex={0}
                    onClick={() => selectEntry(entry.entryId)}
                    onKeyDown={(event) => {
                      if (event.key === 'Enter' || event.key === ' ') {
                        event.preventDefault();
                        selectEntry(entry.entryId);
                      }
                    }}
                  >
                    <div className="atlas-island-entry-meta">
                      <button
                        className="video-id-button"
                        type="button"
                        onClick={(event) => {
                          event.stopPropagation();
                          onOpenTranscript(entry.videoId, entry.entryId);
                        }}
                      >
                        {entry.videoId}
                      </button>
                      <span>#{entry.segIndex}</span>
                    </div>
                    <p className="atlas-island-entry-en">{entry.en}</p>
                    <p className="atlas-island-entry-zh">{entry.zh}</p>
                  </article>
                </li>
              ))}
            </ol>
          </section>
        ) : selectedEntry ? (
          <div className="atlas-detail">
            <article className="atlas-entry-detail">
              {selectedCluster ? (
                <>
                  <div className="atlas-region-line" style={{ ['--cluster-color' as string]: selectedCluster.color }}>
                    <span />
                    <strong>{selectedCluster.label}</strong>
                    <em>{selectedCluster.size.toLocaleString()} lines</em>
                  </div>
                  <p className="atlas-region-description">{selectedCluster.description}</p>
                </>
              ) : null}

              <div className="atlas-entry-meta">
                <button
                  className="video-id-button"
                  type="button"
                  onClick={() => onOpenTranscript(selectedEntry.videoId, selectedEntry.entryId)}
                >
                  {selectedEntry.videoId}
                </button>
                <span>#{selectedEntry.segIndex}</span>
              </div>

              <p className="atlas-entry-en">{selectedEntry.en}</p>
              <p className="atlas-entry-zh">{selectedEntry.zh}</p>

              {selectedCluster?.topPhrases.length ? (
                <div className="atlas-phrase-list">
                  {selectedCluster.topPhrases.map((phrase) => (
                    <span key={phrase}>{phrase}</span>
                  ))}
                </div>
              ) : null}
            </article>
          </div>
        ) : null}

        {searchResults.length ? (
          <section className="atlas-section">
            <div className="atlas-section-header">
              <strong>Semantic Matches</strong>
              <span>{searchResults.length}</span>
            </div>
            <ol className="atlas-result-list">
              {searchResults.slice(0, 12).map((result) => (
                <li key={result.entryId}>
                  <button
                    className={classNames('atlas-result-row', result.entryId === selectedEntryId && 'is-active')}
                    type="button"
                    onClick={() => selectEntry(result.entryId)}
                  >
                    <span>{result.videoId}#{result.segIndex}</span>
                    <strong>{result.score.toFixed(3)}</strong>
                    <em>{result.en}</em>
                  </button>
                </li>
              ))}
            </ol>
          </section>
        ) : showIslandBrowser ? (
          <section className="atlas-section atlas-island-section">
            <div className="atlas-section-header">
              <strong>Visual Islands</strong>
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
        ) : null}

        {showLocalContext ? (
          <section className="atlas-section atlas-context-section">
            <div className="atlas-section-header">
              <strong>Local Context</strong>
              <span>{contextItems.length}</span>
            </div>
            {contextItems.length ? (
              <ol className="atlas-context-list">
                {contextItems.map((item) => (
                  <li key={item.entryId}>
                    <article
                      className={classNames('atlas-context-item', item.isFocus && 'is-focus')}
                      role="button"
                      tabIndex={0}
                      onClick={() => selectEntry(item.entryId)}
                      onKeyDown={(event) => {
                        if (event.key === 'Enter' || event.key === ' ') {
                          event.preventDefault();
                          selectEntry(item.entryId);
                        }
                      }}
                    >
                      <div>
                        <span>{item.videoId}#{item.segIndex}</span>
                        <button
                          type="button"
                          onClick={(event) => {
                            event.stopPropagation();
                            onOpenTranscript(item.videoId, item.entryId);
                          }}
                        >
                          Open
                        </button>
                      </div>
                      <p>{item.en}</p>
                      <p>{item.zh}</p>
                    </article>
                  </li>
                ))}
              </ol>
            ) : (
              <p className="atlas-muted">{getEntryText(selectedEntry) ? 'Loading context.' : 'Select a dot or search result.'}</p>
            )}
          </section>
        ) : null}
      </aside>
    </section>
  );
}
