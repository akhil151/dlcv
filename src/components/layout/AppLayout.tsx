import React, { useEffect, useState } from 'react';
import { SideNavBar } from './SideNavBar';
import { TopAppBar, type EngineStatusSummary } from './TopAppBar';
import { api } from '../../services/api';

export interface AppLayoutProps {
  children: React.ReactNode;
  title?: string;
  activeTaskName?: string;
  isProcessing?: boolean;
  onCancelProcess?: () => void;
  hideTopNav?: boolean;
  noPadding?: boolean;
}

export const AppLayout: React.FC<AppLayoutProps> = ({
  children,
  title,
  activeTaskName,
  isProcessing,
  onCancelProcess,
  hideTopNav = false,
  noPadding = false,
}) => {
  const [engine, setEngine] = useState<EngineStatusSummary | null>(null);

  useEffect(() => {
    let cancelled = false;
    api
      .getReadiness()
      .then((snapshot) => {
        if (cancelled) return;
        const parts = [
          snapshot.detector.available ? snapshot.detector.name : `${snapshot.detector.name} unavailable`,
          snapshot.detector.device.toUpperCase(),
          snapshot.ffmpeg.available ? 'FFmpeg ready' : 'FFmpeg missing',
        ];
        setEngine({ label: parts.join(' • '), ready: snapshot.processingReady });
      })
      .catch(() => {
        if (!cancelled) setEngine({ label: 'Backend unreachable', ready: false });
      });
    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <div className="flex min-h-screen bg-surface selection:bg-primary/30 selection:text-primary">
      <SideNavBar />

      <main className="ml-[240px] flex-1 flex flex-col min-h-screen overflow-y-auto relative">
        {!hideTopNav && (
          <TopAppBar
            title={title}
            activeTaskName={activeTaskName}
            isProcessing={isProcessing}
            onCancelProcess={onCancelProcess}
            engine={engine}
          />
        )}
        <div className={`flex-1 ${noPadding ? '' : 'p-xl'}`}>{children}</div>
      </main>
    </div>
  );
};
