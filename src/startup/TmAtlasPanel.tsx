import { FormEvent, useEffect, useMemo, useRef, useState } from 'react';
import { classNames } from '../classes';
import type { ContextItem, SearchResult } from '../search/protocol';
import type {
  SemanticLandscapeCluster,
  SemanticLandscapeData,
  SemanticLandscapePoint,
} from './semantic-landscape';
import { hexToRgba } from './colors';

type ThemeMode = 'dark' | 'light';
type AtlasMode = '2d' | '3d';

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
  onQueryChange: (query: string) => void;
  onSearch: () => void;
  onSelectEntry: (entryId: string | null) => void;
  onOpenTranscript: (videoId: string, focusEntryId: string) => void;
  onClear: () => void;
  onToggleTheme: () => void;
}

interface View2d {
  scale: number;
  offsetX: number;
  offsetY: number;
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

interface DragState {
  active: boolean;
  mode: 'pan2d' | 'rotate3d' | 'pan3d';
  pointerId: number;
  lastX: number;
  lastY: number;
  startX: number;
  startY: number;
  moved: boolean;
}

const INITIAL_VIEW_2D: View2d = {
  scale: 0.92,
  offsetX: 0,
  offsetY: 0,
};

const INITIAL_VIEW_3D: View3d = {
  rotateX: 0.4,
  rotateY: -0.65,
  zoom: 1.18,
  offsetX: 0,
  offsetY: 0,
};

function clamp(value: number, minimum: number, maximum: number): number {
  return Math.min(maximum, Math.max(minimum, value));
}

function project2d(point: SemanticLandscapePoint, width: number, height: number, view: View2d) {
  const x = ((point.x / 1000) - 0.5) * width * view.scale + width / 2 + view.offsetX;
  const y = (0.5 - (point.y / 1000)) * height * view.scale + height / 2 + view.offsetY;
  return { x, y, depth: 0, culled: false };
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
  onQueryChange,
  onSearch,
  onSelectEntry,
  onOpenTranscript,
  onClear,
  onToggleTheme,
}: TmAtlasPanelProps) {
  const wrapRef = useRef<HTMLDivElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const dragRef = useRef<DragState | null>(null);
  const [mode, setMode] = useState<AtlasMode>('2d');
  const [size, setSize] = useState({ width: 1, height: 1 });
  const [view2d, setView2d] = useState<View2d>(INITIAL_VIEW_2D);
  const [view3d, setView3d] = useState<View3d>(INITIAL_VIEW_3D);
  const [hoveredEntryId, setHoveredEntryId] = useState<string | null>(null);

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

  const projectedPoints = useMemo<ProjectedPoint[]>(() => {
    if (!data) {
      return [];
    }

    const firstCluster = data.clusters[0]!;
    const points =
      mode === '3d'
        ? (() => {
            const rawItems = data.points.map((point): RawProjected3d => ({
              point,
              cluster: clusterById.get(point.clusterId) ?? firstCluster,
              ...project3dRaw(point, view3d, visualGeometry),
            }));
            const fitted = fitProjected3d(rawItems, size.width, size.height);

            return rawItems.map((item) => ({
              point: item.point,
              cluster: item.cluster,
              x: (item.rawX - fitted.centerX) * fitted.scale * view3d.zoom + size.width / 2 + view3d.offsetX,
              y: (item.rawY - fitted.centerY) * fitted.scale * view3d.zoom + size.height / 2 + view3d.offsetY,
              depth: item.depth,
              culled: item.culled,
            }));
          })()
        : data.points.map((point) => ({
            point,
            cluster: clusterById.get(point.clusterId) ?? firstCluster,
            ...project2d(point, size.width, size.height, view2d),
          }));

    if (mode === '3d') {
      points.sort((left, right) => left.depth - right.depth);
    }

    return points;
  }, [clusterById, data, mode, size.height, size.width, view2d, view3d, visualGeometry]);

  const selectedPoint = selectedEntryId ? pointById.get(selectedEntryId) ?? null : null;
  const selectedSearchResult = selectedEntryId ? searchResultById.get(selectedEntryId) ?? null : null;
  const selectedEntry = selectedPoint ?? selectedSearchResult ?? null;
  const selectedCluster = selectedPoint ? clusterById.get(selectedPoint.clusterId) ?? null : null;
  const hoveredPoint = hoveredEntryId ? pointById.get(hoveredEntryId) ?? null : null;
  const visiblePointCount = projectedPoints.filter((item) => !item.culled).length;
  const showIslandBrowser = !!data && searchResults.length === 0 && !query.trim();

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
    const text = theme === 'dark' ? '#edf8ef' : '#17251b';
    const computedStyle = window.getComputedStyle(document.documentElement);
    const selected = computedStyle.getPropertyValue('--accent').trim();

    context.fillStyle = background;
    context.fillRect(0, 0, size.width, size.height);

    context.strokeStyle = grid;
    context.lineWidth = 1;
    const gridStep = Math.max(80, Math.round(Math.min(size.width, size.height) / 7));
    for (let x = (view2d.offsetX % gridStep) - gridStep; x < size.width + gridStep; x += gridStep) {
      context.beginPath();
      context.moveTo(x, 0);
      context.lineTo(x, size.height);
      context.stroke();
    }
    for (let y = (view2d.offsetY % gridStep) - gridStep; y < size.height + gridStep; y += gridStep) {
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

    for (const item of projectedPoints) {
      if (item.culled) {
        continue;
      }

      const isSelected = item.point.entryId === selectedEntryId;
      const isHovered = item.point.entryId === hoveredEntryId;
      const isSearchHit = searchHitIds.has(item.point.entryId);
      const hasSearch = searchHitIds.size > 0;
      const radius = isSelected ? 5.2 : isHovered ? 4.4 : isSearchHit ? 3.4 : mode === '3d' ? 2.35 : 1.85;
      const alpha = isSelected ? 1 : isHovered ? 0.95 : isSearchHit ? 0.86 : hasSearch ? 0.16 : 0.58;

      context.fillStyle = isSelected || isSearchHit ? hexToRgba(selected, alpha) : hexToRgba(item.point.color, alpha);
      context.beginPath();
      context.arc(item.x, item.y, radius, 0, Math.PI * 2);
      context.fill();

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
    hoveredEntryId,
    mode,
    projectedPoints,
    searchHitIds,
    searchResults,
    selectedEntryId,
    size.height,
    size.width,
    theme,
    view2d,
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
    onSearch();
  }

  function handlePointerDown(event: React.PointerEvent<HTMLCanvasElement>): void {
    if (event.button !== 0) {
      return;
    }

    const canvas = event.currentTarget;
    canvas.setPointerCapture(event.pointerId);
    dragRef.current = {
      active: true,
      mode: mode === '2d' ? 'pan2d' : event.ctrlKey || event.metaKey ? 'pan3d' : 'rotate3d',
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

      if (drag.mode === 'pan2d') {
        setView2d((current) => ({
          ...current,
          offsetX: current.offsetX + dx,
          offsetY: current.offsetY + dy,
        }));
      } else if (drag.mode === 'pan3d') {
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
    setHoveredEntryId(nearest?.point.entryId ?? null);
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
      onSelectEntry(nearest?.point.entryId ?? null);
    }
  }

  function handlePointerLeave(): void {
    if (!dragRef.current?.active) {
      setHoveredEntryId(null);
    }
  }

  function handleWheel(event: React.WheelEvent<HTMLCanvasElement>): void {
    event.preventDefault();
    const zoomFactor = Math.exp(-event.deltaY * 0.001);

    if (mode === '3d') {
      setView3d((current) => ({
        ...current,
        zoom: clamp(current.zoom * zoomFactor, 0.42, 4.2),
      }));
      return;
    }

    const bounds = event.currentTarget.getBoundingClientRect();
    const cursorX = event.clientX - bounds.left;
    const cursorY = event.clientY - bounds.top;
    setView2d((current) => {
      const nextScale = clamp(current.scale * zoomFactor, 0.32, 8);
      const scaleChange = nextScale / current.scale;
      return {
        scale: nextScale,
        offsetX: cursorX - size.width / 2 - (cursorX - size.width / 2 - current.offsetX) * scaleChange,
        offsetY: cursorY - size.height / 2 - (cursorY - size.height / 2 - current.offsetY) * scaleChange,
      };
    });
  }

  function resetView(): void {
    setView2d(INITIAL_VIEW_2D);
    setView3d(INITIAL_VIEW_3D);
    setHoveredEntryId(null);
    onClear();
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
          <button className="atlas-title" type="button" onClick={onClear}>
            TM Atlas
          </button>

          <div className="atlas-mode-control" aria-label="Projection mode">
            <button
              className={classNames(mode === '2d' && 'is-active')}
              type="button"
              onClick={() => setMode('2d')}
            >
              2D
            </button>
            <button
              className={classNames(mode === '3d' && 'is-active')}
              type="button"
              onClick={() => setMode('3d')}
            >
              3D
            </button>
          </div>

          <button className="atlas-icon-button" type="button" onClick={resetView} title="Reset atlas">
            Reset
          </button>

          <button
            className="atlas-icon-button"
            type="button"
            onClick={onToggleTheme}
            title={`Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`}
          >
            {theme === 'dark' ? 'Light' : 'Dark'}
          </button>
        </div>

        <div className="atlas-status" aria-label="Load status">
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
              <span>UMAP</span>
              <strong>{visiblePointCount.toLocaleString()}</strong>
            </div>
          ) : null}
        </div>

        {hoveredPoint ? (
          <div className="atlas-tooltip">
            <strong>{hoveredPoint.videoId}#{hoveredPoint.segIndex}</strong>
            <span>{hoveredPoint.en}</span>
          </div>
        ) : null}
      </div>

      <aside className="atlas-sidebar">
        <form className="atlas-search" onSubmit={handleSubmit}>
          <input
            aria-label="Search English subtitle lines"
            type="text"
            value={query}
            placeholder="Search English lines"
            onChange={(event) => onQueryChange(event.target.value)}
          />
          <button type="submit" disabled={searching || !searchReady || !query.trim()}>
            {searching ? '...' : 'Search'}
          </button>
        </form>

        {errorText ? <p className="atlas-message atlas-message--error">{errorText}</p> : null}
        {searchNote ? <p className="atlas-message">{searchNote}</p> : null}

        <div className="atlas-detail">
          {selectedEntry ? (
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
          ) : (
            <article className="atlas-entry-detail atlas-entry-detail--empty">
              <strong>{data ? data.pointCount.toLocaleString() : '0'} English lines</strong>
              <span>{data ? `${data.clusterCount ?? data.clusters.length} UMAP regions` : 'UMAP data is loading'}</span>
            </article>
          )}
        </div>

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
                    onClick={() => onSelectEntry(result.entryId)}
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
                      selectedCluster?.id === cluster.id && 'is-active',
                    )}
                    style={{ ['--cluster-color' as string]: cluster.color }}
                    type="button"
                    onClick={() => onSelectEntry(cluster.medoidEntryId)}
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
                    onClick={() => onSelectEntry(item.entryId)}
                    onKeyDown={(event) => {
                      if (event.key === 'Enter' || event.key === ' ') {
                        event.preventDefault();
                        onSelectEntry(item.entryId);
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
      </aside>
    </section>
  );
}
