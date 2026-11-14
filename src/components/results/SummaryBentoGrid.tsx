import React from 'react';
import type { RedactionSummary } from '../../types';

export interface SummaryBentoGridProps {
  summary?: RedactionSummary;
}

const formatTotalTime = (secs: number): string => {
  if (!secs) return '—';
  const mins = Math.floor(secs / 60);
  const remainder = Math.round(secs % 60);
  return mins > 0 ? `${mins}m ${remainder}s` : `${remainder}s`;
};

export const SummaryBentoGrid: React.FC<SummaryBentoGridProps> = ({ summary }) => {
  const faces = summary?.facesBlurred ?? 0;
  const frames = summary?.framesProcessed ?? 0;
  const framesWithFaces = summary?.framesWithDetections ?? 0;
  const timeSeconds = summary?.processingTimeSeconds ?? 0;
  const avgFps = summary?.avgProcessingFps ?? 0;

  const cards = [
    { icon: 'face', label: 'Face Regions Masked', value: faces.toLocaleString(), tone: 'border-l-primary' },
    { icon: 'iframe', label: 'Frames Processed', value: frames.toLocaleString(), tone: 'border-l-secondary' },
    {
      icon: 'visibility',
      label: 'Frames With Detected Faces',
      value: frames ? `${framesWithFaces} (${Math.round((framesWithFaces / frames) * 100)}%)` : '—',
      tone: '',
    },
    { icon: 'timer', label: 'Processing Time', value: formatTotalTime(timeSeconds), tone: '' },
  ];

  return (
    <section className="flex flex-col gap-md mb-xl select-none">
      <div className="grid grid-cols-2 md:grid-cols-4 gap-md">
        {cards.map((card, index) => (
          <div
            key={index}
            className={`glass-card rounded-2xl p-lg flex flex-col justify-between border-l-4 ${card.tone || 'border-l-outline-variant'}`}
          >
            <span className="material-symbols-outlined text-primary mb-md">{card.icon}</span>
            <div>
              <p className="font-display text-[28px] leading-none mb-sm font-bold text-on-surface">
                {card.value}
              </p>
              <p className="font-label-md text-on-surface-variant">{card.label}</p>
            </div>
          </div>
        ))}
      </div>

      <div className="glass-card rounded-2xl p-lg grid grid-cols-2 md:grid-cols-4 gap-md">
        <div>
          <p className="font-label-sm text-on-surface-variant uppercase mb-xs">Output</p>
          <p className="font-label-md text-on-surface">{summary?.outputResolution || '—'}</p>
        </div>
        <div>
          <p className="font-label-sm text-on-surface-variant uppercase mb-xs">Avg. pipeline FPS</p>
          <p className="font-label-md text-on-surface">{avgFps > 0 ? `${avgFps.toFixed(1)} FPS` : '—'}</p>
        </div>
        <div>
          <p className="font-label-sm text-on-surface-variant uppercase mb-xs">Blur settings</p>
          <p className="font-label-md text-on-surface capitalize">
            {summary?.blurStyle || '—'}
            {summary?.blurIntensity ? ` · ${summary.blurIntensity}%` : ''}
          </p>
        </div>
        <div>
          <p className="font-label-sm text-on-surface-variant uppercase mb-xs">Audio</p>
          <p className="font-label-md text-on-surface">
            {summary?.audioPreserved === true
              ? 'Preserved (AAC)'
              : summary?.audioPreserved === false
                ? 'No audio stream'
                : 'Unknown'}
          </p>
        </div>
      </div>

      <p className="font-label-sm text-on-surface-variant opacity-75">
        “Face regions masked” counts masked face rectangles across all frames (region-frames), not
        unique people. Detection coverage is the share of processed frames where at least one face was
        detected; it is not a precision/recall measurement.
      </p>
    </section>
  );
};
