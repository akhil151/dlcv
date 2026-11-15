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
  const plates = summary?.licensePlatesRedacted ?? 0;
  const documents = summary?.documentsRedacted ?? 0;
  const laptopScreens = summary?.laptopScreensRedacted ?? 0;
  const laptopQuadRegions = summary?.laptopScreenQuadRegions ?? 0;
  const laptopFallbackRegions = summary?.laptopScreenFallbackRegions ?? 0;
  const phoneScreens = summary?.phoneScreensRedacted ?? 0;
  const phoneQuadRegions = summary?.phoneScreenQuadRegions ?? 0;
  const phoneFallbackRegions = summary?.phoneScreenFallbackRegions ?? 0;
  const frames = summary?.framesProcessed ?? 0;
  const framesWithFaces = summary?.framesWithDetections ?? 0;
  const timeSeconds = summary?.processingTimeSeconds ?? 0;
  const avgFps = summary?.avgProcessingFps ?? 0;

  const cards = [
    { icon: 'face', label: 'Face Regions Masked', value: faces.toLocaleString(), tone: 'border-l-primary' },
    {
      icon: 'directions_car',
      label: 'Plate Regions Masked',
      value: plates.toLocaleString(),
      tone: 'border-l-tertiary-container',
    },
    {
      icon: 'badge',
      label: 'Document Regions Masked',
      value: documents.toLocaleString(),
      tone: 'border-l-error',
    },
    {
      icon: 'laptop_mac',
      label: 'Laptop Screen Regions Masked',
      value: laptopScreens.toLocaleString(),
      tone: 'border-l-primary',
      detail:
        laptopScreens > 0
          ? `${laptopQuadRegions.toLocaleString()} display quad · ${laptopFallbackRegions.toLocaleString()} device fallback`
          : undefined,
    },
    {
      icon: 'smartphone',
      label: 'Phone Screen Regions Masked',
      value: phoneScreens.toLocaleString(),
      tone: 'border-l-tertiary-container',
      detail:
        phoneScreens > 0
          ? `${phoneQuadRegions.toLocaleString()} display quad · ${phoneFallbackRegions.toLocaleString()} device fallback`
          : undefined,
    },
    { icon: 'iframe', label: 'Frames Processed', value: frames.toLocaleString(), tone: 'border-l-secondary' },
    {
      icon: 'visibility',
      label: 'Frames With Detected Targets',
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
              {'detail' in card && card.detail && (
                <p className="font-label-sm text-on-surface-variant opacity-70 mt-xs">{card.detail}</p>
              )}
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
        unique people. “Plate” and “document” regions are counted the same way, not unique vehicles
        or documents. The document detector locates identity documents but does not classify ID
        cards vs passports, so both are reported together. Laptop- and phone-screen regions are
        counted the same way. A “display quad” means the visible display boundary was refined; a
        “device fallback” means the display could not be separated reliably, so the whole device
        was masked instead. Detection coverage is the share of processed frames where at least
        one selected target was detected; it is not a precision/recall measurement.
      </p>
    </section>
  );
};
