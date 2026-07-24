import React from 'react';
import type { ProcessingMetrics } from '../../types';

export interface MetricsBarProps {
  metrics?: ProcessingMetrics;
  detectedCount?: number;
}

export const MetricsBar: React.FC<MetricsBarProps> = ({
  metrics = {
    fps: 30,
    elapsedSeconds: 42.5,
    remainingSeconds: 15,
    throughputFrames: 1275,
    totalFrames: 1800,
    processedFrames: 1275,
  },
  detectedCount = 7,
}) => {
  const formatTime = (secs: number): string => {
    const mins = Math.floor(secs / 60);
    const remainder = (secs % 60).toFixed(1);
    const padMins = String(mins).padStart(2, '0');
    const padSecs = String(remainder).padStart(4, '0');
    return `${padMins}:${padSecs}`;
  };

  return (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-md">
      {/* Throughput */}
      <div className="bg-surface-container-low border border-outline-variant rounded-xl p-md flex items-center gap-md">
        <div className="w-10 h-10 rounded-lg bg-primary/10 flex items-center justify-center text-primary">
          <span className="material-symbols-outlined">speed</span>
        </div>
        <div className="flex flex-col">
          <span className="font-label-sm text-on-surface-variant uppercase tracking-tighter">
            Throughput
          </span>
          <span className="font-headline-md text-headline-md text-on-surface">
            {metrics.fps || 30} FPS
          </span>
        </div>
      </div>

      {/* Elapsed Time */}
      <div className="bg-surface-container-low border border-outline-variant rounded-xl p-md flex items-center gap-md">
        <div className="w-10 h-10 rounded-lg bg-primary/10 flex items-center justify-center text-primary">
          <span className="material-symbols-outlined">timer</span>
        </div>
        <div className="flex flex-col">
          <span className="font-label-sm text-on-surface-variant uppercase tracking-tighter">
            Elapsed Time
          </span>
          <span className="font-headline-md text-headline-md text-on-surface">
            {formatTime(metrics.elapsedSeconds || 42.5)}
          </span>
        </div>
      </div>

      {/* Total Entities */}
      <div className="bg-surface-container-low border border-outline-variant rounded-xl p-md flex items-center gap-md">
        <div className="w-10 h-10 rounded-lg bg-primary/10 flex items-center justify-center text-primary">
          <span className="material-symbols-outlined">track_changes</span>
        </div>
        <div className="flex flex-col">
          <span className="font-label-sm text-on-surface-variant uppercase tracking-tighter">
            Total Entities
          </span>
          <span className="font-headline-md text-headline-md text-on-surface">
            {detectedCount} Objects
          </span>
        </div>
      </div>
    </div>
  );
};
