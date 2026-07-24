import React from 'react';
import { useNavigate } from 'react-router-dom';

export interface WelcomeBannerProps {
  tasksCompletedCount?: number;
  onUploadClick?: () => void;
}

export const WelcomeBanner: React.FC<WelcomeBannerProps> = ({
  tasksCompletedCount = 0,
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
        <div className="inline-flex items-center gap-xs bg-primary/10 text-primary border border-primary/20 px-md py-xs rounded-full text-xs font-label-sm w-fit">
          <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
          <span>YOLOv11 TensorRT Active Engine</span>
        </div>
        <h2 className="font-display text-display text-on-surface font-bold leading-tight">
          Neural Vision Control Center
        </h2>
        <p className="font-body-lg text-body-lg text-on-surface-variant max-w-2xl leading-relaxed">
          Automated privacy redaction engine for video streams. {tasksCompletedCount > 0 ? `${tasksCompletedCount} processing tasks completed.` : 'Ready for automated face, screen, and license plate redaction.'}
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
