import React from 'react';

export interface EngineStatusSummary {
  label: string;
  ready: boolean;
}

export interface TopAppBarProps {
  title?: string;
  activeTaskName?: string;
  isProcessing?: boolean;
  onCancelProcess?: () => void;
  engine?: EngineStatusSummary | null;
}

export const TopAppBar: React.FC<TopAppBarProps> = ({
  title = 'Dashboard',
  activeTaskName,
  isProcessing = false,
  onCancelProcess,
  engine = null,
}) => {
  return (
    <header className="sticky top-0 w-full z-40 flex justify-between items-center px-xl h-16 bg-surface/80 backdrop-blur-md border-b border-outline-variant select-none">
      <div className="flex items-center gap-md">
        {activeTaskName ? (
          <div className="flex items-center gap-md">
            <span className="text-on-surface-variant font-label-md">Active Task:</span>
            <span className="font-bold text-on-surface font-label-md truncate max-w-sm md:max-w-lg">
              {activeTaskName}
            </span>
            {isProcessing && (
              <span className="bg-primary/10 text-primary px-sm py-[2px] rounded text-[10px] font-bold uppercase border border-primary/20 animate-pulse">
                PROCESSING
              </span>
            )}
          </div>
        ) : (
          <h2 className="font-headline-md text-headline-md text-on-surface font-bold">
            {title}
          </h2>
        )}
      </div>

      <div className="flex items-center gap-md">
        {isProcessing && onCancelProcess ? (
          <button
            type="button"
            onClick={onCancelProcess}
            className="bg-error/10 text-error border border-error/20 px-md py-sm rounded-lg font-label-md hover:bg-error/20 transition-all cursor-pointer flex items-center gap-xs"
          >
            <span className="material-symbols-outlined text-[16px]">cancel</span>
            <span>Cancel Process</span>
          </button>
        ) : (
          <div className="flex items-center gap-sm">
            <div className="hidden sm:flex items-center gap-xs bg-surface-container-high border border-outline-variant px-sm py-xs rounded-full text-xs font-mono text-on-surface-variant">
              <span
                className={`w-2 h-2 rounded-full ${engine?.ready ? 'bg-emerald-400' : 'bg-tertiary'}`}
              ></span>
              <span>{engine?.label || 'Engine status unknown'}</span>
            </div>
          </div>
        )}
      </div>
    </header>
  );
};
