import { useMemo } from 'react';
import type {
  SemanticFlowTimeline,
  SemanticLandscapeCluster,
  SemanticFlowTimelineBin,
} from './semantic-landscape';

const CHART_WIDTH = 960;
const CHART_HEIGHT = 420;
const PADDING_TOP = 28;
const PADDING_RIGHT = 18;
const PADDING_BOTTOM = 40;
const PADDING_LEFT = 18;

type Point = {
  x: number;
  y: number;
};

interface SemanticFlowTimelinePanelProps {
  timeline: SemanticFlowTimeline;
  clusters: SemanticLandscapeCluster[];
}

function formatPercent(value: number): string {
  return `${Math.round(value * 100)}%`;
}

function formatClusterName(clusterId: number, clusterById: Map<number, SemanticLandscapeCluster>): string {
  if (clusterId < 0) {
    return 'No dominant cluster';
  }

  return clusterById.get(clusterId)?.label ?? 'Unknown cluster';
}

function buildAreaPath(top: Point[], bottom: Point[]): string {
  if (top.length === 0 || bottom.length === 0) {
    return '';
  }

  const topPath = top
    .map((point, index) => `${index === 0 ? 'M' : 'L'} ${point.x.toFixed(2)} ${point.y.toFixed(2)}`)
    .join(' ');
  const bottomPath = [...bottom]
    .reverse()
    .map((point) => `L ${point.x.toFixed(2)} ${point.y.toFixed(2)}`)
    .join(' ');
  return `${topPath} ${bottomPath} Z`;
}

function buildLinePath(points: Point[]): string {
  if (points.length === 0) {
    return '';
  }

  return points
    .map((point, index) => `${index === 0 ? 'M' : 'L'} ${point.x.toFixed(2)} ${point.y.toFixed(2)}`)
    .join(' ');
}

function sumClusterCounts(bin: SemanticFlowTimelineBin): number {
  let total = 0;
  for (const count of bin.clusterCounts) {
    total += count ?? 0;
  }
  return total;
}

export default function SemanticFlowTimelinePanel({
  timeline,
  clusters,
}: SemanticFlowTimelinePanelProps) {
  const clusterById = useMemo(
    () => new Map<number, SemanticLandscapeCluster>(clusters.map((cluster) => [cluster.id, cluster])),
    [clusters],
  );

  const clusterOrder = useMemo(
    () =>
      [...clusters].sort(
        (left, right) => right.size - left.size || left.label.localeCompare(right.label),
      ),
    [clusters],
  );

  const plotWidth = CHART_WIDTH - PADDING_LEFT - PADDING_RIGHT;
  const plotHeight = CHART_HEIGHT - PADDING_TOP - PADDING_BOTTOM;
  const centerY = PADDING_TOP + plotHeight / 2;
  const maxTotal = Math.max(1, ...timeline.bins.map((bin) => bin.total));
  const stepWidth = timeline.bins.length > 1 ? plotWidth / (timeline.bins.length - 1) : plotWidth;
  const peakBin = timeline.bins[timeline.peakBinIndex] ?? null;

  const clusterTotals = useMemo(() => {
    const totals = new Map<number, number>();
    for (const cluster of clusters) {
      totals.set(cluster.id, 0);
    }

    for (const bin of timeline.bins) {
      for (let index = 0; index < bin.clusterCounts.length; index += 1) {
        const value = bin.clusterCounts[index] ?? 0;
        totals.set(index, (totals.get(index) ?? 0) + value);
      }
    }

    return totals;
  }, [clusters, timeline.bins]);

  const dominantCluster = useMemo(() => {
    return [...clusterTotals.entries()]
      .sort((left, right) => right[1] - left[1] || left[0] - right[0])
      .map(([clusterId]) => clusterById.get(clusterId))
      .find(Boolean) ?? null;
  }, [clusterById, clusterTotals]);

  const paths = useMemo(() => {
    const topPoints = new Map<number, Point[]>();
    const bottomPoints = new Map<number, Point[]>();
    const medianLines = new Map<number, Point[]>();

    for (const cluster of clusterOrder) {
      topPoints.set(cluster.id, []);
      bottomPoints.set(cluster.id, []);
      medianLines.set(cluster.id, []);
    }

    timeline.bins.forEach((bin, index) => {
      const x = PADDING_LEFT + stepWidth * index;
      const total = Math.max(0, sumClusterCounts(bin));
      const envelopeHeight = total === 0 ? 0 : (total / maxTotal) * plotHeight * 0.88;
      let cursor = centerY - envelopeHeight / 2;

      for (const cluster of clusterOrder) {
        const count = bin.clusterCounts[cluster.id] ?? 0;
        const segmentHeight = total === 0 ? 0 : (count / total) * envelopeHeight;
        const top = cursor;
        const bottom = cursor + segmentHeight;

        topPoints.get(cluster.id)?.push({ x, y: top });
        bottomPoints.get(cluster.id)?.push({ x, y: bottom });
        medianLines.get(cluster.id)?.push({ x, y: top + segmentHeight / 2 });

        cursor = bottom;
      }
    });

    return clusterOrder.map((cluster) => ({
      cluster,
      areaPath: buildAreaPath(topPoints.get(cluster.id) ?? [], bottomPoints.get(cluster.id) ?? []),
      linePath: buildLinePath(medianLines.get(cluster.id) ?? []),
    }));
  }, [centerY, clusterOrder, maxTotal, plotHeight, stepWidth, timeline.bins]);

  const topLegendClusters = useMemo(() => {
    return [...clusters]
      .map((cluster) => ({
        cluster,
        total: clusterTotals.get(cluster.id) ?? 0,
        share: timeline.bins.length === 0 ? 0 : (clusterTotals.get(cluster.id) ?? 0) / Math.max(1, timeline.bins.reduce((sum, bin) => sum + sumClusterCounts(bin), 0)),
      }))
      .sort((left, right) => right.total - left.total || left.cluster.label.localeCompare(right.cluster.label))
      .slice(0, 6);
  }, [clusterTotals, clusters, timeline.bins]);

  const peakBandX =
    timeline.bins.length > 1 ? PADDING_LEFT + stepWidth * timeline.peakBinIndex - stepWidth / 2 : PADDING_LEFT;
  return (
    <section className="panel semantic-panel semantic-flow-panel" aria-label="Semantic flow timeline">
      <div className="panel-header semantic-panel-header">
        <div className="results-heading">
          <h2>Semantic Flow</h2>
          <span>
            Cluster composition across normalized video progress, centered by cue density instead of
            raw wall-clock time.
          </span>
        </div>

        <div className="semantic-summary">
          <span>{timeline.binCount} timeline slices</span>
          <span>{clusters.length} semantic bands</span>
        </div>
      </div>

      <div className="semantic-flow-chart-shell">
        <div className="semantic-flow-chart-header">
          <strong>Corpus-wide topic motion from intro to outro</strong>
          <span>Stream thickness tracks subtitle density at each progress slice.</span>
        </div>

        <svg
          className="semantic-flow-svg"
          viewBox={`0 0 ${CHART_WIDTH} ${CHART_HEIGHT}`}
          role="img"
          aria-label={`Peak cue density appears around ${peakBin?.label ?? 'the timeline midpoint'}. Intro is led by ${formatClusterName(timeline.leadingClusterId, clusterById)} and the ending is led by ${formatClusterName(timeline.trailingClusterId, clusterById)}.`}
        >
          {[0, 0.25, 0.5, 0.75, 1].map((step) => {
            const x = PADDING_LEFT + plotWidth * step;
            return (
              <line
                key={step}
                className="semantic-flow-grid-line"
                x1={x}
                x2={x}
                y1={PADDING_TOP}
                y2={CHART_HEIGHT - PADDING_BOTTOM}
              />
            );
          })}

          <line
            className="semantic-flow-grid-line semantic-flow-grid-line--center"
            x1={PADDING_LEFT}
            x2={CHART_WIDTH - PADDING_RIGHT}
            y1={centerY}
            y2={centerY}
          />

          {peakBin ? (
            <rect
              className="semantic-flow-peak-band"
              x={Math.max(PADDING_LEFT, peakBandX)}
              y={PADDING_TOP}
              width={Math.min(plotWidth, Math.max(stepWidth * 0.92, 10))}
              height={plotHeight}
              rx={12}
            />
          ) : null}

          {paths.map(({ cluster, areaPath, linePath }) => (
            <g key={cluster.id}>
              <path
                className="semantic-flow-area"
                d={areaPath}
                fill={cluster.color}
                fillOpacity={0.7}
              />
              <path
                className="semantic-flow-ridge"
                d={linePath}
                stroke={cluster.color}
                strokeWidth={1.2}
              />
            </g>
          ))}
        </svg>

        <div className="semantic-flow-axis" aria-hidden="true">
          <span>0%</span>
          <span>25%</span>
          <span>50%</span>
          <span>75%</span>
          <span>100%</span>
        </div>

        <p className="semantic-flow-subnote">
          Peak density lands in <strong>{peakBin?.label ?? '--'}</strong> with{' '}
          <strong>{timeline.peakTotal.toLocaleString()}</strong> cue midpoints.
        </p>
      </div>

      <div className="semantic-flow-stats">
        <div className="semantic-flow-stat">
          <span>Intro anchor</span>
          <strong>{formatClusterName(timeline.leadingClusterId, clusterById)}</strong>
        </div>

        <div className="semantic-flow-stat">
          <span>Outro anchor</span>
          <strong>{formatClusterName(timeline.trailingClusterId, clusterById)}</strong>
        </div>

        <div className="semantic-flow-stat">
          <span>Peak window</span>
          <strong>{peakBin?.label ?? '--'}</strong>
        </div>

        <div className="semantic-flow-stat">
          <span>Most present cluster</span>
          <strong>{dominantCluster?.label ?? '--'}</strong>
        </div>
      </div>

      <div className="semantic-flow-legend" aria-label="Most present flow clusters">
        {topLegendClusters.map(({ cluster, share }) => (
          <div
            key={cluster.id}
            className="semantic-flow-legend-item"
            style={{ ['--cluster-color' as string]: cluster.color }}
          >
            <span className="semantic-flow-legend-dot" aria-hidden="true" />
            <span className="semantic-flow-legend-name">{cluster.label}</span>
            <strong>{formatPercent(share)}</strong>
          </div>
        ))}
      </div>
    </section>
  );
}
