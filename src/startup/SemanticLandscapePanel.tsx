import { useEffect, useMemo, useRef, useState } from 'react';
import type {
  SemanticLandscapeCluster,
  SemanticLandscapeData,
  SemanticLandscapePoint,
} from './semantic-landscape';

const CANVAS_PADDING = 28;

type ThemeMode = 'dark' | 'light';

type HoverState = {
  entryId: string;
  x: number;
  y: number;
};

type ScreenPoint = {
  point: SemanticLandscapePoint;
  x: number;
  y: number;
};

interface SemanticLandscapePanelProps {
  data: SemanticLandscapeData;
  theme: ThemeMode;
  onOpenTranscript: (videoId: string, focusEntryId: string) => void;
}

function hexToRgba(hex: string, alpha: number): string {
  const value = hex.replace('#', '');
  const normalized = value.length === 3 ? value.split('').map((part) => `${part}${part}`).join('') : value;

  if (normalized.length !== 6) {
    return `rgba(127, 176, 105, ${alpha})`;
  }

  const red = Number.parseInt(normalized.slice(0, 2), 16);
  const green = Number.parseInt(normalized.slice(2, 4), 16);
  const blue = Number.parseInt(normalized.slice(4, 6), 16);
  return `rgba(${red}, ${green}, ${blue}, ${alpha})`;
}

function toCanvasPosition(value: number, extent: number): number {
  return CANVAS_PADDING + (value / 1000) * extent;
}

export default function SemanticLandscapePanel({
  data,
  theme,
  onOpenTranscript,
}: SemanticLandscapePanelProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [plotWidth, setPlotWidth] = useState(0);
  const [plotHeight, setPlotHeight] = useState(420);
  const [selectedClusterId, setSelectedClusterId] = useState<number | null>(null);
  const [selectedEntryId, setSelectedEntryId] = useState<string | null>(null);
  const [hoverState, setHoverState] = useState<HoverState | null>(null);

  const clusterById = useMemo(
    () => new Map<number, SemanticLandscapeCluster>(data.clusters.map((cluster) => [cluster.id, cluster])),
    [data.clusters],
  );

  const pointById = useMemo(
    () => new Map<string, SemanticLandscapePoint>(data.points.map((point) => [point.entryId, point])),
    [data.points],
  );

  useEffect(() => {
    const container = containerRef.current;
    if (!container) {
      return;
    }

    const observer = new ResizeObserver((entries) => {
      const entry = entries[0];
      if (!entry) {
        return;
      }

      const nextWidth = Math.max(320, Math.floor(entry.contentRect.width));
      setPlotWidth(nextWidth);
      setPlotHeight(Math.max(360, Math.min(540, Math.round(nextWidth * 0.58))));
    });

    observer.observe(container);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    if (!selectedEntryId || selectedClusterId === null) {
      return;
    }

    const selectedPoint = pointById.get(selectedEntryId);
    if (selectedPoint && selectedPoint.clusterId !== selectedClusterId) {
      setSelectedEntryId(null);
    }
  }, [pointById, selectedClusterId, selectedEntryId]);

  const effectivePlotWidth = Math.max(1, plotWidth - CANVAS_PADDING * 2);
  const effectivePlotHeight = Math.max(1, plotHeight - CANVAS_PADDING * 2);

  const screenPoints = useMemo<ScreenPoint[]>(
    () =>
      data.points.map((point) => ({
        point,
        x: toCanvasPosition(point.x, effectivePlotWidth),
        y: CANVAS_PADDING + (1 - point.y / 1000) * effectivePlotHeight,
      })),
    [data.points, effectivePlotHeight, effectivePlotWidth],
  );

  const interactivePoints =
    selectedClusterId === null
      ? screenPoints
      : screenPoints.filter(({ point }) => point.clusterId === selectedClusterId);

  const hoveredPoint = hoverState ? pointById.get(hoverState.entryId) ?? null : null;
  const selectedPoint = selectedEntryId ? pointById.get(selectedEntryId) ?? null : null;
  const detailPoint = selectedPoint ?? hoveredPoint ?? null;
  const detailCluster =
    selectedClusterId !== null
      ? clusterById.get(selectedClusterId) ?? null
      : detailPoint
        ? clusterById.get(detailPoint.clusterId) ?? null
        : null;

  const visiblePointCount =
    selectedClusterId === null ? data.points.length : interactivePoints.length;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || plotWidth === 0) {
      return;
    }

    const context = canvas.getContext('2d');
    if (!context) {
      return;
    }

    const devicePixelRatio = window.devicePixelRatio || 1;
    canvas.width = Math.round(plotWidth * devicePixelRatio);
    canvas.height = Math.round(plotHeight * devicePixelRatio);
    canvas.style.width = `${plotWidth}px`;
    canvas.style.height = `${plotHeight}px`;

    context.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0);
    context.clearRect(0, 0, plotWidth, plotHeight);

    const background = theme === 'dark' ? '#0d1410' : '#f3f7f2';
    const grid = theme === 'dark' ? 'rgba(197, 228, 203, 0.08)' : 'rgba(51, 104, 72, 0.12)';
    const frame = theme === 'dark' ? 'rgba(197, 228, 203, 0.18)' : 'rgba(51, 104, 72, 0.22)';
    const highlight = theme === 'dark' ? '#f4f8f2' : '#1f2430';

    context.fillStyle = background;
    context.fillRect(0, 0, plotWidth, plotHeight);

    context.strokeStyle = grid;
    context.lineWidth = 1;
    for (let step = 0; step <= 4; step += 1) {
      const x = CANVAS_PADDING + (effectivePlotWidth / 4) * step;
      const y = CANVAS_PADDING + (effectivePlotHeight / 4) * step;

      context.beginPath();
      context.moveTo(x, CANVAS_PADDING);
      context.lineTo(x, plotHeight - CANVAS_PADDING);
      context.stroke();

      context.beginPath();
      context.moveTo(CANVAS_PADDING, y);
      context.lineTo(plotWidth - CANVAS_PADDING, y);
      context.stroke();
    }

    context.strokeStyle = frame;
    context.strokeRect(
      CANVAS_PADDING,
      CANVAS_PADDING,
      effectivePlotWidth,
      effectivePlotHeight,
    );

    for (const cluster of data.clusters) {
      if (selectedClusterId !== null && cluster.id !== selectedClusterId) {
        continue;
      }

      const clusterX = toCanvasPosition(cluster.x, effectivePlotWidth);
      const clusterY = CANVAS_PADDING + (1 - cluster.y / 1000) * effectivePlotHeight;
      context.fillStyle = hexToRgba(cluster.color, 0.08);
      context.strokeStyle = hexToRgba(cluster.color, 0.38);
      context.lineWidth = 1.25;
      context.beginPath();
      context.arc(clusterX, clusterY, 12 + Math.sqrt(cluster.size) * 0.16, 0, Math.PI * 2);
      context.fill();
      context.stroke();
    }

    for (const { point, x, y } of screenPoints) {
      const cluster = clusterById.get(point.clusterId);
      if (!cluster) {
        continue;
      }

      const dimmed = selectedClusterId !== null && point.clusterId !== selectedClusterId;
      const isSelected = point.entryId === selectedEntryId;
      const isHovered = point.entryId === hoverState?.entryId;
      const radius = isSelected ? 4.2 : isHovered ? 3.8 : 1.7;
      const alpha = dimmed ? 0.08 : isSelected ? 0.98 : isHovered ? 0.82 : 0.34;

      context.fillStyle = hexToRgba(cluster.color, alpha);
      context.beginPath();
      context.arc(x, y, radius, 0, Math.PI * 2);
      context.fill();

      if (isSelected || isHovered) {
        context.strokeStyle = hexToRgba(highlight, isSelected ? 0.9 : 0.65);
        context.lineWidth = isSelected ? 1.4 : 1.1;
        context.beginPath();
        context.arc(x, y, radius + 2.4, 0, Math.PI * 2);
        context.stroke();
      }
    }
  }, [
    clusterById,
    data.clusters,
    effectivePlotHeight,
    effectivePlotWidth,
    hoverState,
    plotHeight,
    plotWidth,
    screenPoints,
    selectedClusterId,
    selectedEntryId,
    theme,
  ]);

  function findPoint(clientX: number, clientY: number): ScreenPoint | null {
    const canvas = canvasRef.current;
    if (!canvas) {
      return null;
    }

    const bounds = canvas.getBoundingClientRect();
    const x = clientX - bounds.left;
    const y = clientY - bounds.top;
    let nearest: ScreenPoint | null = null;
    let nearestDistance = 64;

    for (const candidate of interactivePoints) {
      const dx = candidate.x - x;
      const dy = candidate.y - y;
      const distance = dx * dx + dy * dy;
      if (distance < nearestDistance) {
        nearest = candidate;
        nearestDistance = distance;
      }
    }

    return nearest;
  }

  function handlePointerMove(event: React.PointerEvent<HTMLCanvasElement>): void {
    const nearest = findPoint(event.clientX, event.clientY);
    if (!nearest) {
      setHoverState(null);
      return;
    }

    setHoverState({
      entryId: nearest.point.entryId,
      x: nearest.x,
      y: nearest.y,
    });
  }

  function handlePointerLeave(): void {
    setHoverState(null);
  }

  function handleCanvasClick(event: React.MouseEvent<HTMLCanvasElement>): void {
    const nearest = findPoint(event.clientX, event.clientY);
    if (!nearest) {
      setSelectedEntryId(null);
      return;
    }

    setSelectedEntryId(nearest.point.entryId);
  }

  function handleClusterToggle(clusterId: number | null): void {
    setSelectedClusterId((current) => (current === clusterId ? null : clusterId));
    setHoverState(null);
  }

  return (
    <section className="panel semantic-panel" aria-label="Semantic landscape">
      <div className="panel-header semantic-panel-header">
        <div className="results-heading">
          <h2>Semantic Landscape</h2>
          <span>
            Precomputed {data.projection} distribution across all {data.pointCount.toLocaleString()}{' '}
            entries.
          </span>
        </div>
        <div className="semantic-summary">
          <span>{data.clusters.length} clusters</span>
          <span>{visiblePointCount.toLocaleString()} visible</span>
        </div>
      </div>

      <div className="semantic-cluster-strip" role="toolbar" aria-label="Semantic cluster filters">
        <button
          className={selectedClusterId === null ? 'semantic-cluster-chip is-active' : 'semantic-cluster-chip'}
          type="button"
          onClick={() => handleClusterToggle(null)}
        >
          <span className="semantic-cluster-dot semantic-cluster-dot--all" />
          <span>All</span>
          <strong>{data.pointCount.toLocaleString()}</strong>
        </button>

        {data.clusters.map((cluster) => (
          <button
            key={cluster.id}
            className={
              selectedClusterId === cluster.id
                ? 'semantic-cluster-chip is-active'
                : 'semantic-cluster-chip'
            }
            style={{ ['--cluster-color' as string]: cluster.color }}
            type="button"
            onClick={() => handleClusterToggle(cluster.id)}
          >
            <span className="semantic-cluster-dot" />
            <span>{cluster.label}</span>
            <strong>{cluster.size.toLocaleString()}</strong>
          </button>
        ))}
      </div>

      <div className="semantic-workspace">
        <div className="semantic-plot-shell">
          <div ref={containerRef} className="semantic-canvas-shell">
            <canvas
              ref={canvasRef}
              className="semantic-canvas"
              onClick={handleCanvasClick}
              onPointerLeave={handlePointerLeave}
              onPointerMove={handlePointerMove}
            />

            {hoverState && hoveredPoint ? (
              <div
                className="semantic-tooltip"
                style={{
                  left: `${Math.min(Math.max(hoverState.x + 12, 16), Math.max(16, plotWidth - 260))}px`,
                  top: `${Math.min(Math.max(hoverState.y + 12, 16), Math.max(16, plotHeight - 120))}px`,
                }}
              >
                <strong>{hoveredPoint.videoId}#{hoveredPoint.segIndex}</strong>
                <p>{hoveredPoint.en}</p>
              </div>
            ) : null}
          </div>

          <p className="semantic-footnote">
            Hover to inspect a point. Click to pin it. Filter by cluster to isolate one semantic
            region.
          </p>
        </div>

        <aside className="semantic-detail-panel">
          {detailPoint ? (
            <div className="semantic-detail-card semantic-detail-card--entry">
              <span
                className="semantic-detail-badge"
                style={{ ['--cluster-color' as string]: detailCluster?.color ?? '#84a98c' }}
              >
                {detailCluster?.label ?? 'Selected Entry'}
              </span>

              <div className="semantic-detail-result-header">
                <span className="result-metric">
                  <span className="result-metric-label">YouTube ID</span>
                  <strong>{detailPoint.videoId}</strong>
                </span>

                <span className="result-metric result-score">
                  <span className="result-metric-label">Entry</span>
                  <strong>#{detailPoint.segIndex}</strong>
                </span>
              </div>

              <div className="result-copy-group semantic-detail-copy-group">
                <div className="result-copy">
                  <div className="result-copy-line">
                    <p className="result-en semantic-entry-copy semantic-entry-copy--en">
                      {detailPoint.en}
                    </p>
                    <span className="result-char-count">{detailPoint.en.length} chars</span>
                  </div>
                </div>

                <div className="result-copy">
                  <p className="result-zh semantic-entry-copy semantic-entry-copy--zh">
                    {detailPoint.zh}
                  </p>
                </div>
              </div>

              {detailCluster?.keywords.length ? (
                <div className="semantic-keyword-list">
                  {detailCluster.keywords.map((keyword) => (
                    <span key={keyword} className="semantic-keyword">
                      {keyword}
                    </span>
                  ))}
                </div>
              ) : null}
              <button
                className="semantic-action-button"
                type="button"
                onClick={() => onOpenTranscript(detailPoint.videoId, detailPoint.entryId)}
              >
                Open Transcript
              </button>
            </div>
          ) : detailCluster ? (
            <div className="semantic-detail-card">
              <span
                className="semantic-detail-badge"
                style={{ ['--cluster-color' as string]: detailCluster.color }}
              >
                {detailCluster.label}
              </span>
              <div className="semantic-detail-stat">
                <strong>{detailCluster.size.toLocaleString()}</strong>
                <span>entries in this region</span>
              </div>
              {detailCluster.keywords.length ? (
                <div className="semantic-keyword-list">
                  {detailCluster.keywords.map((keyword) => (
                    <span key={keyword} className="semantic-keyword">
                      {keyword}
                    </span>
                  ))}
                </div>
              ) : null}
              <div className="semantic-sample-list">
                {detailCluster.samples.map((sample) => (
                  <button
                    key={sample.entryId}
                    className="semantic-sample-card"
                    type="button"
                    onClick={() => setSelectedEntryId(sample.entryId)}
                  >
                    <strong>
                      {sample.videoId}#{sample.segIndex}
                    </strong>
                    <span>{sample.en}</span>
                  </button>
                ))}
              </div>
            </div>
          ) : (
            <div className="semantic-detail-card is-empty">
              <strong>All-entry view</strong>
              <p>
                This panel shows a precomputed 2D projection of the stored MiniLM vectors. Each
                dot is one TM entry.
              </p>
              <p>
                Use the cluster chips to focus a region, then hover or click a point to inspect the
                underlying EN/ZH pair.
              </p>
              <div className="semantic-detail-stat">
                <strong>{data.pointCount.toLocaleString()}</strong>
                <span>semantic points loaded from startup data</span>
              </div>
            </div>
          )}
        </aside>
      </div>
    </section>
  );
}
