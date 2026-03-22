import { PointerEvent as ReactPointerEvent, useRef, useState } from 'react';
import type { CueTimeDistribution } from '../search/protocol';

const CHART_WIDTH = 920;
const CHART_HEIGHT = 240;
const CHART_PADDING_TOP = 20;
const CHART_PADDING_RIGHT = 12;
const CHART_PADDING_BOTTOM = 34;
const CHART_PADDING_LEFT = 16;

interface CueTimeDistributionPanelProps {
  data: CueTimeDistribution;
}

interface HoveredBinState {
  index: number;
  x: number;
  y: number;
}

function formatPercent(value: number): string {
  if (!Number.isFinite(value)) {
    return '--';
  }

  const percent = value * 100;
  if (percent >= 10) {
    return `${Math.round(percent)}%`;
  }

  return `${percent.toFixed(1)}%`;
}

function formatPercentRange(start: number, end: number): string {
  const startPercent = Math.round(start * 100);
  const endPercent = Math.round(end * 100);
  return startPercent === endPercent ? `${startPercent}%` : `${startPercent}% - ${endPercent}%`;
}

function formatDuration(ms: number): string {
  if (!Number.isFinite(ms) || ms <= 0) {
    return '--';
  }

  const totalSeconds = Math.round(ms / 1000);
  if (totalSeconds < 60) {
    return `${totalSeconds}s`;
  }

  const hours = Math.floor(totalSeconds / 3600);
  const minutes = Math.floor((totalSeconds % 3600) / 60);
  const seconds = totalSeconds % 60;

  if (hours > 0) {
    if (minutes === 0) {
      return `${hours}h`;
    }

    return `${hours}h ${minutes}m`;
  }

  if (seconds === 0) {
    return `${minutes}m`;
  }

  return `${minutes}m ${seconds}s`;
}

export default function CueTimeDistributionPanel({ data }: CueTimeDistributionPanelProps) {
  const chartShellRef = useRef<HTMLDivElement | null>(null);
  const [hoveredBin, setHoveredBin] = useState<HoveredBinState | null>(null);
  const plotWidth = CHART_WIDTH - CHART_PADDING_LEFT - CHART_PADDING_RIGHT;
  const plotHeight = CHART_HEIGHT - CHART_PADDING_TOP - CHART_PADDING_BOTTOM;
  const stepWidth = plotWidth / Math.max(1, data.bins.length);
  const peakBinIndex = data.bins.findIndex((value) => value === data.peakCoverage);
  const peakX = CHART_PADDING_LEFT + Math.max(0, peakBinIndex) * stepWidth;
  const excludedEntryCount = Math.max(0, data.totalEntryCount - data.timedEntryCount);
  const excludedVideoCount = Math.max(0, data.totalVideoCount - data.timedVideoCount);
  const averageCoverage = formatPercent(data.averageCoverage);
  const peakCoverage = formatPercent(data.peakCoverage);
  const peakWindow = formatPercentRange(data.peakRangeStart, data.peakRangeEnd);
  const meanCueLength = formatDuration(data.averageCueDurationMs);
  const medianSpan = formatDuration(data.medianVideoSpanMs);
  const hoveredBinIndex = hoveredBin?.index ?? -1;
  const hoveredCoverage =
    hoveredBinIndex >= 0 ? formatPercent(data.bins[hoveredBinIndex] ?? 0) : null;
  const hoveredWindow =
    hoveredBinIndex >= 0
      ? formatPercentRange(hoveredBinIndex / data.binCount, (hoveredBinIndex + 1) / data.binCount)
      : null;
  const hoveredTopLine = hoveredBinIndex >= 0 ? data.binTopLines[hoveredBinIndex] ?? null : null;
  const tooltipWidth = hoveredTopLine?.zh ? 332 : 300;
  const tooltipHeight = hoveredTopLine ? (hoveredTopLine.zh ? 164 : 138) : 100;
  const shellWidth = chartShellRef.current?.clientWidth ?? CHART_WIDTH;
  const shellHeight = chartShellRef.current?.clientHeight ?? CHART_HEIGHT;
  const tooltipLeft =
    hoveredBin === null
      ? 0
      : Math.min(Math.max(hoveredBin.x + 16, 12), Math.max(12, shellWidth - tooltipWidth - 12));
  const tooltipTop =
    hoveredBin === null
      ? 0
      : Math.min(Math.max(hoveredBin.y + 16, 12), Math.max(12, shellHeight - tooltipHeight - 12));

  const updateHoveredBin = (event: ReactPointerEvent<SVGRectElement>, index: number): void => {
    const bounds = chartShellRef.current?.getBoundingClientRect();
    if (!bounds) {
      return;
    }

    setHoveredBin({
      index,
      x: event.clientX - bounds.left,
      y: event.clientY - bounds.top,
    });
  };

  return (
    <section className="panel semantic-panel time-distribution-panel" aria-label="Cue time distribution">
      <div className="panel-header semantic-panel-header">
        <div className="results-heading">
          <h2>Time Distribution</h2>
          <span>
            Subtitle cue coverage after aligning every video from its first cue start to its
            last cue end.
          </span>
        </div>
      </div>

      <div ref={chartShellRef} className="time-distribution-chart-shell">
        <div className="time-distribution-chart-header">
          <strong>Share of aligned videos carrying cues</strong>
          <span>Normalized from 0% to 100% of each video span</span>
        </div>

        <svg
          className="time-distribution-svg"
          viewBox={`0 0 ${CHART_WIDTH} ${CHART_HEIGHT}`}
          role="img"
          aria-label={`Average cue coverage ${averageCoverage} across the aligned timeline with a peak of ${peakCoverage} around ${peakWindow}.`}
        >
          {[0, 0.25, 0.5, 0.75, 1].map((level) => {
            const y = CHART_PADDING_TOP + plotHeight - level * plotHeight;
            return (
              <g key={level}>
                <line
                  className={
                    level === 0
                      ? 'time-distribution-grid-line time-distribution-grid-line--baseline'
                      : 'time-distribution-grid-line'
                  }
                  x1={CHART_PADDING_LEFT}
                  x2={CHART_WIDTH - CHART_PADDING_RIGHT}
                  y1={y}
                  y2={y}
                />
                {level < 1 ? (
                  <text className="time-distribution-grid-label" x={CHART_WIDTH - CHART_PADDING_RIGHT} y={y - 6}>
                    {formatPercent(level)}
                  </text>
                ) : null}
              </g>
            );
          })}

          <rect
            className="time-distribution-peak-band"
            x={peakX}
            y={CHART_PADDING_TOP}
            width={stepWidth}
            height={plotHeight}
            rx={6}
          />

          {data.bins.map((value, index) => {
            const barHeight = Math.max(2, value * plotHeight);
            const x = CHART_PADDING_LEFT + index * stepWidth;
            const y = CHART_PADDING_TOP + plotHeight - barHeight;
            const width = Math.max(1.5, stepWidth - 1.25);
            const binTopLine = data.binTopLines[index] ?? null;
            const binWindow = formatPercentRange(index / data.binCount, (index + 1) / data.binCount);
            const barTitle = binTopLine
              ? `${binWindow} | ${formatPercent(value)} coverage | ${binTopLine.en}`
              : `${binWindow} | ${formatPercent(value)} coverage`;

            return (
              <rect
                key={index}
                className={
                  [
                    'time-distribution-bar',
                    index === peakBinIndex ? 'time-distribution-bar--peak' : '',
                    index === hoveredBinIndex ? 'time-distribution-bar--active' : '',
                  ]
                    .filter(Boolean)
                    .join(' ')
                }
                x={x}
                y={y}
                width={width}
                height={barHeight}
                rx={Math.min(3, width / 2)}
                onPointerEnter={(event) => updateHoveredBin(event, index)}
                onPointerMove={(event) => updateHoveredBin(event, index)}
                onPointerLeave={() => setHoveredBin((current) => (current?.index === index ? null : current))}
              >
                <title>{barTitle}</title>
              </rect>
            );
          })}
        </svg>

        {hoveredBin && hoveredWindow && hoveredCoverage ? (
          <div
            className="semantic-tooltip time-distribution-tooltip"
            style={{
              left: `${tooltipLeft}px`,
              top: `${tooltipTop}px`,
            }}
          >
            <strong>{hoveredWindow}</strong>
            <p className="time-distribution-tooltip-meta">{hoveredCoverage} coverage across aligned videos</p>
            {hoveredTopLine ? (
              <>
                <p>{hoveredTopLine.en}</p>
                {hoveredTopLine.zh ? (
                  <p className="time-distribution-tooltip-translation">{hoveredTopLine.zh}</p>
                ) : null}
                <p className="time-distribution-tooltip-meta">
                  Most frequent exact line pair in this slice, seen{' '}
                  {hoveredTopLine.count.toLocaleString()} time{hoveredTopLine.count === 1 ? '' : 's'}.
                </p>
              </>
            ) : (
              <p className="time-distribution-tooltip-meta">No timed cues intersect this slice.</p>
            )}
          </div>
        ) : null}

        <div className="time-distribution-axis" aria-hidden="true">
          <span>0%</span>
          <span>25%</span>
          <span>50%</span>
          <span>75%</span>
          <span>100%</span>
        </div>

        {excludedEntryCount > 0 || excludedVideoCount > 0 ? (
          <p className="time-distribution-subnote">
            Excluded {excludedEntryCount.toLocaleString()} cues from {excludedVideoCount.toLocaleString()}{' '}
            videos without complete timing spans.
          </p>
        ) : null}
      </div>

      <div className="time-distribution-stats">
        <div className="time-distribution-stat">
          <span>Average coverage</span>
          <strong>{averageCoverage}</strong>
        </div>

        <div className="time-distribution-stat">
          <span>Peak coverage</span>
          <strong>{peakCoverage}</strong>
        </div>

        <div className="time-distribution-stat">
          <span>Peak timing</span>
          <strong>{peakWindow}</strong>
        </div>

        <div className="time-distribution-stat">
          <span>Mean cue length</span>
          <strong>{meanCueLength}</strong>
        </div>

        <div className="time-distribution-stat">
          <span>Median span</span>
          <strong>{medianSpan}</strong>
        </div>
      </div>
    </section>
  );
}
