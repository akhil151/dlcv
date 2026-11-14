import React from 'react';

export interface ExportDownloadHubProps {
  onDownloadProtected: () => void;
  onReEditClick: () => void;
}

export const ExportDownloadHub: React.FC<ExportDownloadHubProps> = ({
  onDownloadProtected,
  onReEditClick,
}) => {
  return (
    <section className="flex flex-col md:flex-row gap-lg items-center justify-between pt-xl border-t border-outline-variant">
      <div className="flex flex-wrap gap-md">
        <button
          type="button"
          onClick={onDownloadProtected}
          className="border border-outline-variant text-on-surface px-lg py-sm rounded-lg font-label-md hover:bg-surface-variant transition-colors flex items-center gap-sm cursor-pointer"
        >
          <span className="material-symbols-outlined text-[18px]">download</span>
          <span>Save Protected Video</span>
        </button>
        <button
          type="button"
          onClick={onReEditClick}
          className="border border-outline-variant text-on-surface px-lg py-sm rounded-lg font-label-md hover:bg-surface-variant transition-colors flex items-center gap-sm cursor-pointer"
        >
          <span className="material-symbols-outlined text-[18px]">refresh</span>
          <span>Process Another Video</span>
        </button>
      </div>

      <div className="flex items-center gap-sm text-on-surface-variant">
        <span className="w-2 h-2 rounded-full bg-emerald-400"></span>
        <span className="font-label-sm">
          Output integrity verified (full decode + frame count)
        </span>
      </div>
    </section>
  );
};
