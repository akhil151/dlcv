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
    <section className="mb-xl flex flex-col md:flex-row justify-between items-start md:items-center gap-lg glass-panel p-xl rounded-2xl border border-outline-variant/60 shadow-xl">
      <div className="flex flex-col gap-xs">
        <div className="inline-flex items-center gap-xs bg-primary/10 text-primary border border-primary/20 px-sm py-[2px] rounded-full text-xs font-label-sm w-fit mb-xs">
          <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
          <span>YOLOv11 TensorRT Active</span>
        </div>
        <h2 className="font-display text-headline-lg text-on-surface font-bold">
          Neural Vision Control Center
        </h2>
        <p className="font-body-md text-on-surface-variant max-w-xl">
          Automated privacy redaction engine for video streams. {tasksCompletedCount > 0 ? `${tasksCompletedCount} processing tasks completed.` : 'Ready for video ingest.'}
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
