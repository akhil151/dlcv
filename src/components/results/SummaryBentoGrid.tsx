import React from 'react';
import { ResultVideoPlayer } from './ResultVideoPlayer';
import type { RedactionSummary } from '../../types';

export interface SummaryBentoGridProps {
  summary?: RedactionSummary;
  videoTitle?: string;
  onPlayClick?: () => void;
}

export const SummaryBentoGrid: React.FC<SummaryBentoGridProps> = ({
  summary,
  videoTitle = 'output_redacted.mp4',
  onPlayClick,
}) => {
  const faces = summary?.facesBlurred ?? 0;
  const plates = summary?.licensePlatesRedacted ?? 0;
  const screens = summary?.screensHidden ?? 0;
  const idCards = summary?.idCardsFound ?? 0;
  const timeSeconds = summary?.processingTimeSeconds ?? 0;
  const avgFps = summary?.evaluationReport?.performanceMetrics?.avgFps ?? 0;

  const formatTotalTime = (secs: number): string => {
    if (!secs || secs === 0) return 'N/A';
    const mins = Math.floor(secs / 60);
    const remainder = Math.round(secs % 60);
    return mins > 0 ? `${mins}m ${remainder}s` : `${remainder}s`;
  };

  return (
    <section className="grid grid-cols-1 md:grid-cols-4 gap-md mb-xl select-none">
      {/* Video Preview Card */}
      <ResultVideoPlayer
        title={summary?.taskId ? `task_${summary.taskId.substring(0, 8)}.mp4` : videoTitle}
        videoUrl={summary?.protectedVideoUrl}
        onPlayClick={onPlayClick}
      />

      {/* Faces Blurred Card */}
      <div className="glass-card rounded-2xl p-lg flex flex-col justify-between border-l-4 border-l-primary hover:-translate-y-1 transition-all">
        <span className="material-symbols-outlined text-primary mb-md">face</span>
        <div>
          <p className="font-display text-[32px] leading-none mb-sm font-bold text-on-surface">{faces}</p>
          <p className="font-label-md text-on-surface-variant">Faces Blurred</p>
        </div>
      </div>

      {/* License Plates Redacted Card */}
      <div className="glass-card rounded-2xl p-lg flex flex-col justify-between border-l-4 border-l-tertiary hover:-translate-y-1 transition-all">
        <span className="material-symbols-outlined text-tertiary mb-md">directions_car</span>
        <div>
          <p className="font-display text-[32px] leading-none mb-sm font-bold text-on-surface">{plates}</p>
          <p className="font-label-md text-on-surface-variant">License Plates Redacted</p>
        </div>
      </div>

      {/* Screen Hidden Card */}
      <div className="glass-card rounded-2xl p-lg flex flex-col justify-between hover:-translate-y-1 transition-all">
        <span className="material-symbols-outlined text-on-surface-variant mb-md">monitor</span>
        <div>
          <p className="font-display text-[32px] leading-none mb-sm font-bold text-on-surface">{screens}</p>
          <p className="font-label-md text-on-surface-variant">Screens Hidden</p>
        </div>
      </div>

      {/* ID Cards Found Card */}
      <div
        className={`glass-card rounded-2xl p-lg flex flex-col justify-between hover:-translate-y-1 transition-all ${
          idCards === 0 ? 'opacity-50' : ''
        }`}
      >
        <span className="material-symbols-outlined text-on-surface-variant mb-md">badge</span>
        <div>
          <p className="font-display text-[32px] leading-none mb-sm font-bold text-on-surface">{idCards}</p>
          <p className="font-label-md text-on-surface-variant">ID Cards Found</p>
        </div>
      </div>

      {/* Performance Stats Card */}
      <div className="md:col-span-2 glass-card rounded-2xl p-lg flex items-center gap-xl hover:-translate-y-1 transition-all">
        <div className="flex-1 border-r border-outline-variant pr-lg">
          <p className="font-label-sm text-on-surface-variant uppercase mb-xs">Processing Time</p>
          <p className="font-display text-headline-md text-primary font-bold">
            {formatTotalTime(timeSeconds)}
          </p>
        </div>
        <div className="flex-1">
          <p className="font-label-sm text-on-surface-variant uppercase mb-xs font-bold">Inference Speed</p>
          <p className="font-display text-headline-md text-secondary font-bold">
            {avgFps > 0 ? `${avgFps} FPS` : 'N/A'}
          </p>
        </div>
      </div>
    </section>
  );
};
