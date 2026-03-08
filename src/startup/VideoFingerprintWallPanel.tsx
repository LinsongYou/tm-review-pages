import { KeyboardEvent as ReactKeyboardEvent, useEffect, useMemo, useState } from 'react';
import type {
  SemanticLandscapeCluster,
  VideoFingerprintVideo,
  VideoFingerprintWall,
} from './semantic-landscape';

interface VideoFingerprintWallPanelProps {
  data: VideoFingerprintWall;
  clusters: SemanticLandscapeCluster[];
  onOpenTranscript: (videoId: string, focusEntryId: string) => void;
}

interface FingerprintStripProps {
  video: VideoFingerprintVideo;
  clusterById: Map<number, SemanticLandscapeCluster>;
  compact?: boolean;
}

function hexToRgba(hex: string, alpha: number): string {
  const value = hex.replace('#', '');
  const normalized = value.length === 3 ? value.split('').map((part) => `${part}${part}`).join('') : value;

  if (normalized.length !== 6) {
    return `rgba(106, 157, 120, ${alpha})`;
  }

  const red = Number.parseInt(normalized.slice(0, 2), 16);
  const green = Number.parseInt(normalized.slice(2, 4), 16);
  const blue = Number.parseInt(normalized.slice(4, 6), 16);
  return `rgba(${red}, ${green}, ${blue}, ${alpha})`;
}

function formatPercent(value: number): string {
  return `${Math.round(value * 100)}%`;
}

function FingerprintStrip({ video, clusterById, compact = false }: FingerprintStripProps) {
  return (
    <div
      className={compact ? 'video-fingerprint-strip video-fingerprint-strip--compact' : 'video-fingerprint-strip'}
      style={{ ['--fingerprint-columns' as string]: String(video.bins.length) }}
      aria-hidden="true"
    >
      {video.bins.map((clusterId, index) => {
        const cluster = clusterById.get(clusterId);
        const density = video.densities[index] ?? 0;
        const background = cluster ? cluster.color : 'var(--field-border)';
        const alpha = cluster ? Math.max(0.2, density * 0.88) : 0.12;

        return (
          <span
            key={`${video.videoId}-${index}`}
            className={cluster ? 'video-fingerprint-cell' : 'video-fingerprint-cell video-fingerprint-cell--empty'}
            style={{
              background,
              opacity: alpha,
            }}
          />
        );
      })}
    </div>
  );
}

function getTopClusters(
  video: VideoFingerprintVideo,
  clusterById: Map<number, SemanticLandscapeCluster>,
) {
  return video.clusterCounts
    .map((count, clusterId) => ({
      cluster: clusterById.get(clusterId) ?? null,
      count,
      share: video.entryCount === 0 ? 0 : count / video.entryCount,
    }))
    .filter((item) => item.cluster && item.count > 0)
    .sort((left, right) => right.count - left.count || (left.cluster?.label ?? '').localeCompare(right.cluster?.label ?? ''))
    .slice(0, 4);
}

export default function VideoFingerprintWallPanel({
  data,
  clusters,
  onOpenTranscript,
}: VideoFingerprintWallPanelProps) {
  const clusterById = useMemo(
    () => new Map<number, SemanticLandscapeCluster>(clusters.map((cluster) => [cluster.id, cluster])),
    [clusters],
  );

  const videoById = useMemo(
    () => new Map<string, VideoFingerprintVideo>(data.videos.map((video) => [video.videoId, video])),
    [data.videos],
  );

  const [selectedVideoId, setSelectedVideoId] = useState<string | null>(data.videos[0]?.videoId ?? null);

  useEffect(() => {
    if (!selectedVideoId || !videoById.has(selectedVideoId)) {
      setSelectedVideoId(data.videos[0]?.videoId ?? null);
    }
  }, [data.videos, selectedVideoId, videoById]);

  const selectedVideo =
    (selectedVideoId ? videoById.get(selectedVideoId) : null) ?? data.videos[0] ?? null;
  const selectedDominantCluster = selectedVideo
    ? clusterById.get(selectedVideo.dominantClusterId) ?? null
    : null;
  const selectedTopClusters = selectedVideo ? getTopClusters(selectedVideo, clusterById) : [];

  function handleRowKeyDown(
    event: ReactKeyboardEvent<HTMLElement>,
    videoId: string,
  ): void {
    if (event.target !== event.currentTarget) {
      return;
    }

    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      setSelectedVideoId(videoId);
    }
  }

  return (
    <section className="panel semantic-panel video-fingerprint-panel" aria-label="Video fingerprints wall">
      <div className="panel-header semantic-panel-header">
        <div className="results-heading">
          <h2>Video Fingerprints</h2>
          <span>
            Each row compresses a full video into a normalized semantic strip so recurring episode
            structure becomes visible at a glance.
          </span>
        </div>

        <div className="semantic-summary">
          <span>{data.videos.length} videos</span>
          <span>{data.binCount} bins per video</span>
        </div>
      </div>

      <div className="video-fingerprint-workspace">
        {selectedVideo ? (
          <aside className="video-fingerprint-detail-card">
            <span
              className="semantic-detail-badge"
              style={{ ['--cluster-color' as string]: selectedDominantCluster?.color ?? 'var(--accent)' }}
            >
              {selectedDominantCluster?.label ?? 'Selected Video'}
            </span>

            <div className="video-fingerprint-detail-header">
              <div className="video-fingerprint-detail-title">
                <button
                  className="video-id-button"
                  type="button"
                  onClick={() => onOpenTranscript(selectedVideo.videoId, selectedVideo.firstEntryId)}
                >
                  {selectedVideo.videoId}
                </button>
                <span>{selectedVideo.entryCount.toLocaleString()} vectorized cues</span>
              </div>

              <div className="video-fingerprint-detail-stats">
                <span>{formatPercent(selectedVideo.dominantShare)} dominant cluster share</span>
                <span>{selectedVideo.timedEntryCount.toLocaleString()} timed cues</span>
              </div>
            </div>

            <FingerprintStrip video={selectedVideo} clusterById={clusterById} />

            <div className="video-fingerprint-axis" aria-hidden="true">
              <span>0%</span>
              <span>25%</span>
              <span>50%</span>
              <span>75%</span>
              <span>100%</span>
            </div>

            <div className="video-fingerprint-top-clusters">
              {selectedTopClusters.map(({ cluster, count, share }) =>
                cluster ? (
                  <div key={`${selectedVideo.videoId}-${cluster.id}`} className="video-fingerprint-top-row">
                    <div className="video-fingerprint-top-meta">
                      <span
                        className="video-fingerprint-top-dot"
                        style={{ background: cluster.color }}
                        aria-hidden="true"
                      />
                      <strong>{cluster.label}</strong>
                      <span>{count.toLocaleString()} cues</span>
                    </div>

                    <div className="video-fingerprint-top-bar-track" aria-hidden="true">
                      <span
                        className="video-fingerprint-top-bar-fill"
                        style={{
                          width: `${Math.max(share * 100, 4)}%`,
                          background: cluster.color,
                        }}
                      />
                    </div>
                  </div>
                ) : null,
              )}
            </div>
          </aside>
        ) : null}

        <div className="video-fingerprint-wall-shell">
          <div className="video-fingerprint-wall-header">
            <strong>Sorted by semantic position in the global landscape</strong>
            <span>Rows are aligned from 0% to 100% of each video's subtitle span.</span>
          </div>

          <ol className="video-fingerprint-list">
            {data.videos.map((video) => {
              const dominantCluster = clusterById.get(video.dominantClusterId) ?? null;
              const topClusters = getTopClusters(video, clusterById).slice(0, 2);

              return (
                <li key={video.videoId}>
                  <article
                    className={
                      video.videoId === selectedVideo?.videoId
                        ? 'video-fingerprint-row is-active'
                        : 'video-fingerprint-row'
                    }
                    role="button"
                    tabIndex={0}
                    onClick={() => setSelectedVideoId(video.videoId)}
                    onKeyDown={(event) => handleRowKeyDown(event, video.videoId)}
                  >
                    <div className="video-fingerprint-row-meta">
                      <div className="video-fingerprint-row-title">
                        <button
                          className="video-id-button"
                          type="button"
                          onClick={(event) => {
                            event.stopPropagation();
                            onOpenTranscript(video.videoId, video.firstEntryId);
                          }}
                        >
                          {video.videoId}
                        </button>
                        <span>{video.entryCount.toLocaleString()} cues</span>
                      </div>

                      <div className="video-fingerprint-row-labels">
                        {dominantCluster ? (
                          <span
                            className="video-fingerprint-cluster-chip"
                            style={{
                              ['--cluster-color' as string]: dominantCluster.color,
                              background: hexToRgba(dominantCluster.color, 0.12),
                            }}
                          >
                            {dominantCluster.label}
                          </span>
                        ) : null}

                        {topClusters.map(({ cluster }) =>
                          cluster ? (
                            <span key={`${video.videoId}-${cluster.id}`} className="video-fingerprint-secondary-label">
                              {cluster.label}
                            </span>
                          ) : null,
                        )}
                      </div>
                    </div>

                    <div className="video-fingerprint-row-visual">
                      <FingerprintStrip video={video} clusterById={clusterById} compact />
                    </div>
                  </article>
                </li>
              );
            })}
          </ol>
        </div>
      </div>
    </section>
  );
}
