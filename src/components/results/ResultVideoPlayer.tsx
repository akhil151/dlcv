import React from 'react';

export interface ResultVideoPlayerProps {
  title: string;
  videoUrl?: string;
  label?: string;
  /** Short line under the label (resolution, frame count, ...). */
  meta?: string;
  emptyMessage?: string;
  /** Marks the protected output as the verified result. */
  verified?: boolean;
}

export const ResultVideoPlayer: React.FC<ResultVideoPlayerProps> = ({
  title,
  videoUrl,
  label = 'Protected output',
  meta,
  emptyMessage = 'No verified video is available yet.',
  verified = false,
}) => {
  return (
    <div className="glass-card rounded-2xl overflow-hidden border border-outline-variant flex flex-col">
      <div className="flex items-start justify-between gap-sm px-md py-sm border-b border-outline-variant">
        <div className="min-w-0">
          <p className="font-label-sm uppercase tracking-widest text-primary">{label}</p>
          <p className="font-label-sm text-on-surface-variant truncate">
            {meta || title}
          </p>
        </div>
        {verified && (
          <span className="shrink-0 inline-flex items-center gap-xs px-sm py-xs rounded-full bg-primary/10 border border-primary/20 text-primary font-label-sm">
            <span className="material-symbols-outlined text-[14px] filled">verified</span>
            Verified
          </span>
        )}
      </div>
      {/* Fixed 16:9 stage for both videos, so the two panes line up exactly
          regardless of the source clip's aspect ratio. */}
      <div className="relative w-full aspect-video bg-black">
        {videoUrl ? (
          <video
            key={videoUrl}
            src={videoUrl}
            controls
            playsInline
            preload="metadata"
            className="absolute inset-0 w-full h-full object-contain bg-black"
          >
            Your browser does not support embedded video playback.
          </video>
        ) : (
          <div className="absolute inset-0 flex items-center justify-center px-xl">
            <p className="font-label-md text-white/60 text-center">{emptyMessage}</p>
          </div>
        )}
      </div>
    </div>
  );
};
