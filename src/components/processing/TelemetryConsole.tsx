import React, { useEffect, useRef } from 'react';

export interface TelemetryConsoleProps {
  logs?: string[];
  autoScroll?: boolean;
}

export const TelemetryConsole: React.FC<TelemetryConsoleProps> = ({
  logs = [],
  autoScroll = true,
}) => {
  const logContainerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (autoScroll && logContainerRef.current) {
      logContainerRef.current.scrollTop = logContainerRef.current.scrollHeight;
    }
  }, [logs, autoScroll]);

  return (
    <div className="bg-surface-container-lowest border border-outline-variant rounded-xl p-md flex flex-col gap-xs font-mono text-xs">
      <div className="flex items-center justify-between border-b border-outline-variant/50 pb-xs mb-xs text-on-surface-variant font-label-sm">
        <span className="flex items-center gap-xs uppercase tracking-widest text-[10px]">
          <span className="material-symbols-outlined text-xs">terminal</span>
          Telemetry Log Stream
        </span>
        <span className="text-[10px] opacity-60">LIVE LOGS</span>
      </div>

      <div
        ref={logContainerRef}
        className="max-h-44 overflow-y-auto flex flex-col gap-1 pr-xs custom-scrollbar text-on-surface-variant"
      >
        {logs.length > 0 ? (
          logs.map((log, idx) => (
            <div key={idx} className="leading-relaxed hover:text-on-surface">
              {log}
            </div>
          ))
        ) : (
          <div className="opacity-50 italic">Waiting for engine telemetry stream output...</div>
        )}
      </div>
    </div>
  );
};
