import React from 'react';

export interface ExportDownloadHubProps {
  onDownloadAuditLog?: () => void;
  onReEditClick?: () => void;
}

export const ExportDownloadHub: React.FC<ExportDownloadHubProps> = ({
  onDownloadAuditLog,
  onReEditClick,
}) => {
  return (
    <section className="flex flex-col md:flex-row gap-lg items-center justify-between pt-xl border-t border-outline-variant">
      <div className="flex flex-wrap gap-md">
        <button
          type="button"
          onClick={onDownloadAuditLog}
          className="border border-outline-variant text-on-surface px-lg py-sm rounded-lg font-label-md hover:bg-surface-variant transition-colors flex items-center gap-sm cursor-pointer"
        >
          <span className="material-symbols-outlined text-[18px]">assessment</span>
          <span>View Processing Report</span>
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
        <span className="w-2 h-2 rounded-full bg-green-500 animate-pulse"></span>
        <span className="font-label-sm">System Status: Optimal</span>
      </div>
    </section>
  );
};
