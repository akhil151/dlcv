import React from 'react';
import type { ProcessingMetrics } from '../../types';

export interface MetricsBarProps {
  metrics: ProcessingMetrics;
  detectedCount?: number;
}

const UNKNOWN = '—';

export const MetricsBar: React.FC<MetricsBarProps> = ({ metrics, detectedCount }) => {
  const formatTime = (secs: number): string => {
    if (!secs) return UNKNOWN;
    const mins = Math.floor(secs / 60);
    const remainder = Math.round(secs % 60);
    return `${String(mins).padStart(2, '0')}:${String(remainder).padStart(2, '0')}`;
  };

  return (
    <div className="grid grid-cols-1 md:grid-cols-4 gap-md">
      <div className="bg-surface-container-low border border-outline-variant rounded-xl p-md flex items-center gap-md">
        <div className="w-10 h-10 rounded-lg bg-primary/10 flex items-center justify-center text-primary">
          <span className="material-symbols-outlined">speed</span>
        </div>
        <div className="flex flex-col">
          <span className="font-label-sm text-on-surface-variant uppercase tracking-tighter">
            Throughput
          </span>
          <span className="font-headline-md text-headline-md text-on-surface">
            {metrics.fps > 0 ? `${metrics.fps} FPS` : UNKNOWN}
          </span>
        </div>
      </div>

      <div className="bg-surface-container-low border border-outline-variant rounded-xl p-md flex items-center gap-md">
        <div className="w-10 h-10 rounded-lg bg-primary/10 flex items-center justify-center text-primary">
          <span className="material-symbols-outlined">timer</span>
        </div>
        <div className="flex flex-col">
          <span className="font-label-sm text-on-surface-variant uppercase tracking-tighter">
            Elapsed Time
          </span>
          <span className="font-headline-md text-headline-md text-on-surface">
            {formatTime(metrics.elapsedSeconds)}
          </span>
        </div>
      </div>

      <div className="bg-surface-container-low border border-outline-variant rounded-xl p-md flex items-center gap-md">
        <div className="w-10 h-10 rounded-lg bg-primary/10 flex items-center justify-center text-primary">
          <span className="material-symbols-outlined">iframe</span>
        </div>
        <div className="flex flex-col">
          <span className="font-label-sm text-on-surface-variant uppercase tracking-tighter">
            Frames
          </span>
          <span className="font-headline-md text-headline-md text-on-surface">
            {metrics.totalFrames > 0
              ? `${metrics.processedFrames}/${metrics.totalFrames}`
              : UNKNOWN}
          </span>
        </div>
      </div>

      <div className="bg-surface-container-low border border-outline-variant rounded-xl p-md flex items-center gap-md">
        <div className="w-10 h-10 rounded-lg bg-primary/10 flex items-center justify-center text-primary">
          <span className="material-symbols-outlined">track_changes</span>
        </div>
        <div className="flex flex-col">
          <span className="font-label-sm text-on-surface-variant uppercase tracking-tighter">
            Face Regions Masked
          </span>
          <span className="font-headline-md text-headline-md text-on-surface">
            {detectedCount !== undefined ? detectedCount.toLocaleString() : UNKNOWN}
          </span>
        </div>
      </div>
    </div>
  );
};
