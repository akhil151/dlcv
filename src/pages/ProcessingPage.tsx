import React, { useEffect, useState } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import { AppLayout } from '../components/layout/AppLayout';
import { DualStreamViewer } from '../components/processing/DualStreamViewer';
import { MetricsBar } from '../components/processing/MetricsBar';
import { TelemetryConsole } from '../components/processing/TelemetryConsole';
import { useAppStore } from '../store/useAppStore';
import { wsManager, type WebSocketStatus } from '../services/websocket';
import type { ProcessingMetrics } from '../types';

export const ProcessingPage: React.FC = () => {
  const navigate = useNavigate();
  const { taskId = 'demo-task' } = useParams();

  const {
    latestTelemetry,
    wsStatus,
    setLatestTelemetry,
    setWsStatus,
  } = useAppStore();

  const [logs, setLogs] = useState<string[]>([
    `[${new Date().toLocaleTimeString()}] SafeFrame AI Neural Worker initialized for Task #${taskId}`,
    `[${new Date().toLocaleTimeString()}] Connecting to live WebSocket telemetry stream...`,
  ]);

  // Connect to live WebSocket telemetry endpoint
  useEffect(() => {
    if (!taskId) return;

    wsManager.connect(
      taskId,
      (packet) => {
        setLatestTelemetry(packet);
        if (packet.logLine) {
          setLogs((prev) => [...prev, packet.logLine as string]);
        }
        if (packet.status === 'COMPLETED' || (packet.progressPct && packet.progressPct >= 100)) {
          setTimeout(() => {
            navigate(`/results/${taskId}`);
          }, 1200);
        }
      },
      (status: WebSocketStatus) => {
        setWsStatus(status);
      }
    );

    return () => {
      wsManager.disconnect();
    };
  }, [taskId, navigate, setLatestTelemetry, setWsStatus]);

  const progressPct = Math.round(latestTelemetry?.progressPct ?? (wsStatus === 'CONNECTED' ? 5 : 0));
  const currentStage = latestTelemetry?.currentStage || (wsStatus === 'CONNECTED' ? 'Neural Redaction' : 'Connecting Telemetry...');
  const isCompleted = latestTelemetry?.status === 'COMPLETED' || progressPct >= 100;

  const metrics: ProcessingMetrics = {
    fps: latestTelemetry?.fps ?? 0,
    elapsedSeconds: latestTelemetry?.elapsedTime ?? 0,
    remainingSeconds: latestTelemetry?.estimatedRemainingTime ?? 0,
    throughputFrames: latestTelemetry?.frameIndex ?? 0,
    totalFrames: latestTelemetry?.totalFrames ?? 0,
    processedFrames: latestTelemetry?.frameIndex ?? 0,
  };

  const activeTaskFilename = `Task_${taskId}.mp4`;
  const detectedObjects = latestTelemetry?.detectedObjects || [];

  return (
    <AppLayout
      title="Processing Stream"
      activeTaskName={activeTaskFilename}
      isProcessing={!isCompleted}
      onCancelProcess={() => {
        wsManager.disconnect();
        navigate('/upload');
      }}
    >
      <div className="max-w-container-max mx-auto w-full flex flex-col gap-xl">
        {/* WebSocket Status Banner if reconnecting */}
        {wsStatus !== 'CONNECTED' && (
          <div className="p-sm bg-tertiary/10 border border-tertiary/30 rounded-lg text-tertiary text-xs flex items-center gap-sm">
            <span className="material-symbols-outlined text-[16px] animate-spin">sync</span>
            <span>WebSocket Telemetry: {wsStatus}. Connecting to CUDA processing node...</span>
          </div>
        )}

        {/* 12-Column Grid Stage */}
        <div className="grid grid-cols-12 gap-xl flex-grow">
          {/* Left Column: Dual Stream & Metrics */}
          <div className="col-span-12 lg:col-span-8 flex flex-col gap-lg">
            <DualStreamViewer
              detectedEntities={detectedObjects}
              cameraName={`Task Stream #${taskId.substring(0, 8)}`}
              isRedacting={!isCompleted}
            />

            <MetricsBar
              metrics={metrics}
              detectedCount={detectedObjects.length || (latestTelemetry?.activeDetections ?? 0)}
            />

            {/* Log Telemetry Console */}
            <TelemetryConsole logs={logs} autoScroll />
          </div>

          {/* Right Column: Status & Progress */}
          <div className="col-span-12 lg:col-span-4 flex flex-col gap-lg">
            {/* Main Progress Card */}
            <div className="glass-panel border border-outline-variant rounded-2xl p-lg flex flex-col gap-lg shadow-xl">
              <div className="flex flex-col gap-sm">
                <div className="flex justify-between items-end">
                  <h2 className="font-label-md font-bold text-on-surface">Overall Progress</h2>
                  <span className="text-headline-lg font-display font-bold text-primary leading-none">
                    {progressPct}%
                  </span>
                </div>
                <div className="h-3 w-full bg-surface-container-highest rounded-full overflow-hidden border border-outline-variant relative">
                  <div
                    className="h-full bg-primary progress-shimmer rounded-full shadow-glow transition-all duration-300"
                    style={{ width: `${progressPct}%` }}
                  ></div>
                </div>
              </div>

              <div className="flex flex-col gap-md">
                <h3 className="font-label-sm text-on-surface-variant uppercase tracking-widest border-b border-outline-variant pb-xs">
                  Pipeline Status
                </h3>

                {/* Pipeline List Items */}
                <div className="flex flex-col gap-sm">
                  <div className="flex items-center gap-md p-sm bg-primary/5 rounded-lg border border-primary/20">
                    <span className="material-symbols-outlined text-primary text-body-md">
                      check_circle
                    </span>
                    <span className="font-label-md text-on-surface">Target Detection</span>
                    <span className="ml-auto font-label-sm text-primary">Active</span>
                  </div>

                  <div className="flex items-center gap-md p-sm bg-surface-container-highest rounded-lg border border-primary/50 animate-pulse">
                    <div className="w-4 h-4 rounded-full border-2 border-primary border-t-transparent animate-spin"></div>
                    <span className="font-label-md text-on-surface font-bold">
                      {currentStage}
                    </span>
                    <span className="ml-auto font-label-sm text-primary">Live</span>
                  </div>

                  <div className="flex items-center gap-md p-sm opacity-50">
                    <span className="material-symbols-outlined text-on-surface-variant text-body-md">
                      tune
                    </span>
                    <span className="font-label-md text-on-surface">Gaussian Blur Masking</span>
                    <span className="ml-auto font-label-sm">Active</span>
                  </div>
                </div>
              </div>

              {/* Detections Stream Summary */}
              <div className="bg-surface-container-lowest border border-outline-variant rounded-xl p-md">
                <h4 className="font-label-sm text-on-surface-variant mb-sm flex items-center gap-sm">
                  <span className="material-symbols-outlined text-xs">list</span>
                  LIVE DETECTED ENTITIES
                </h4>
                {detectedObjects.length > 0 ? (
                  <div className="flex flex-wrap gap-xs">
                    {detectedObjects.map((obj, i) => (
                      <span
                        key={i}
                        className="bg-primary/10 text-primary border border-primary/20 rounded px-sm py-[2px] font-label-sm"
                      >
                        {obj.label || `Entity #${i + 1}`}
                      </span>
                    ))}
                  </div>
                ) : (
                  <p className="font-label-sm text-on-surface-variant opacity-70 italic">
                    Scanning frames for face, license plate, and screen targets...
                  </p>
                )}
              </div>
            </div>
          </div>
        </div>

        {/* Footer */}
        <footer className="mt-auto border-t border-outline-variant py-md flex flex-col md:flex-row justify-between items-center text-label-sm">
          <span className="font-label-sm text-on-surface-variant opacity-70">
            SafeFrame AI Vision Stream • Task #{taskId}
          </span>
          <p className="font-label-sm text-on-surface-variant opacity-60 mt-xs md:mt-0">
            © 2024 SafeFrame AI. Technical Authority in Video Privacy.
          </p>
        </footer>
      </div>
    </AppLayout>
  );
};
