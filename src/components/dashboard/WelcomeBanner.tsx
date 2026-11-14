import React from 'react';
import { useNavigate } from 'react-router-dom';

export interface WelcomeBannerProps {
  tasksCompletedCount?: number;
  engineLabel?: string;
  engineReady?: boolean;
  onUploadClick?: () => void;
}

export const WelcomeBanner: React.FC<WelcomeBannerProps> = ({
  tasksCompletedCount = 0,
  engineLabel = 'YuNet face redaction (CPU)',
  engineReady = false,
  onUploadClick,
}) => {
  const navigate = useNavigate();

  const handleClick = () => {
    if (onUploadClick) {
      onUploadClick();
    } else {
      navigate('/upload');
    }
  };

  return (
    <section className="mb-xl flex flex-col lg:flex-row justify-between items-start lg:items-center gap-xl glass-panel p-xl md:p-2xl rounded-3xl border border-outline-variant/60 shadow-2xl">
      <div className="flex flex-col gap-sm max-w-3xl">
        <div
          className={`inline-flex items-center gap-xs border px-md py-xs rounded-full text-xs font-label-sm w-fit ${
            engineReady
              ? 'bg-primary/10 text-primary border-primary/20'
              : 'bg-error/10 text-error border-error/30'
          }`}
        >
          <span
            className={`w-2 h-2 rounded-full ${engineReady ? 'bg-emerald-400' : 'bg-error'}`}
          ></span>
          <span>{engineReady ? engineLabel : 'Engine unavailable'}</span>
        </div>
        <h2 className="font-display text-display text-on-surface font-bold leading-tight">
          Neural Vision Control Center
        </h2>
        <p className="font-body-lg text-body-lg text-on-surface-variant max-w-2xl leading-relaxed">
          Automated face redaction for uploaded video files.{' '}
          {tasksCompletedCount > 0
            ? `${tasksCompletedCount} verified processing task${tasksCompletedCount === 1 ? '' : 's'} completed.`
            : 'Upload a video to start a verified face-redaction task.'}
        </p>
      </div>

      <button
        type="button"
        onClick={handleClick}
        className="group flex items-center gap-md bg-primary text-on-primary px-xl py-md rounded-xl font-headline-md font-bold hover:shadow-glow hover:scale-[1.02] transition-all active:scale-[0.98] cursor-pointer shrink-0"
      >
        <span className="material-symbols-outlined text-[28px] group-hover:rotate-12 transition-transform">upload_file</span>
        <span>Upload Video</span>
      </button>
    </section>
  );
};
