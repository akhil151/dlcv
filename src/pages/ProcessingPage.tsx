import React, { useEffect, useState } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import { AppLayout } from '../components/layout/AppLayout';
import { DualStreamViewer } from '../components/processing/DualStreamViewer';
import { MetricsBar } from '../components/processing/MetricsBar';
import { TelemetryConsole } from '../components/processing/TelemetryConsole';
import { useAppStore } from '../store/useAppStore';
import { api } from '../services/api';
import { wsManager, type WebSocketStatus } from '../services/websocket';
import type { ProcessingMetrics } from '../types';

const POLL_INTERVAL_MS = 2000;

const TERMINAL_STATUSES = ['COMPLETED', 'FAILED', 'CANCELLED'];

const EMPTY_METRICS: ProcessingMetrics = {
  fps: 0,
  elapsedSeconds: 0,
  remainingSeconds: 0,
  throughputFrames: 0,
  totalFrames: 0,
  processedFrames: 0,
  framesWithDetections: 0,
  masksTotal: 0,
  newTracksTotal: 0,
  detectionLatencyMs: 0,
  blurLatencyMs: 0,
  avgProcessingFps: 0,
  debugFramesSaved: 0,
};

export const ProcessingPage: React.FC = () => {
  const navigate = useNavigate();
  const { taskId = '' } = useParams();

  const {
    activeTask,
    latestTelemetry,
    wsStatus,
    setActiveTask,
    setLatestTelemetry,
    setWsStatus,
  } = useAppStore();

  const [logs, setLogs] = useState<string[]>([]);
  const [statusError, setStatusError] = useState<string | null>(null);
  const [isCancelling, setIsCancelling] = useState(false);

  const status = activeTask?.status;
  const isVerifiedComplete = status === 'COMPLETED' && activeTask?.outputVerified === true;
  const isFailed = status === 'FAILED' || status === 'CANCELLED';
  const isProcessing = !isVerifiedComplete && !isFailed;

  // Authoritative REST status: fetched immediately and polled until terminal.
  // WebSocket telemetry is used for live detail only, never for terminal state.
  useEffect(() => {
    if (!taskId) return;
    let cancelled = false;
    let timer: number | undefined;

    const poll = async () => {
      try {
        const task = await api.getTaskStatus(taskId);
        if (cancelled) return;
        setActiveTask(task);
        setStatusError(null);
        if (task.status === 'COMPLETED') {
          if (task.outputVerified) {
            navigate(`/results/${taskId}`);
          } else {
            setStatusError('The task reported completion without a verified protected output.');
          }
          return;
        }
        if (task.status === 'FAILED' || task.status === 'CANCELLED') {
          return;
        }
      } catch (err: any) {
        if (cancelled) return;
        // Keep polling: a transient failure must not be reported as completion.
        setStatusError(err?.message || 'Unable to read task status from the server.');
      }
      if (!cancelled) timer = window.setTimeout(poll, POLL_INTERVAL_MS);
    };

    poll();
    return () => {
      cancelled = true;
      if (timer) window.clearTimeout(timer);
    };
  }, [taskId, navigate, setActiveTask]);

  // Live WebSocket telemetry (detail only).
  useEffect(() => {
    if (!taskId) return;
    wsManager.connect(
      taskId,
      (packet) => {
        setLatestTelemetry(packet);
        if (packet.logLine) {
          setLogs((prev) => [...prev.slice(-199), packet.logLine as string]);
        }
      },
      (nextStatus: WebSocketStatus) => {
        setWsStatus(nextStatus);
      }
    );
    return () => {
      wsManager.disconnect();
    };
  }, [taskId, setLatestTelemetry, setWsStatus]);

  if (!taskId) {
    return (
      <AppLayout title="Processing Stream">
        <div className="max-w-container-max mx-auto w-full p-xl glass-panel border border-error/30 rounded-2xl text-error">
          No task identifier was provided. Return to the upload page to start a task.
        </div>
      </AppLayout>
    );
  }

  const progressPct = activeTask?.progressPct ?? null;
  const progressLabel = progressPct === null ? '—' : `${Math.round(progressPct)}%`;
  const progressWidth = progressPct === null ? 0 : Math.min(100, Math.max(0, progressPct));

  const currentStage =
    latestTelemetry?.currentStage ||
    (status === 'PROCESSING'
      ? 'Detection + Redaction'
      : TERMINAL_STATUSES.includes(status || '')
        ? status || 'Stopped'
        : 'Awaiting status');

  const metrics: ProcessingMetrics = activeTask?.metrics || EMPTY_METRICS;
  const activeTaskFilename = activeTask?.filename || `Task_${taskId}.mp4`;
  const detectedObjects = latestTelemetry?.detectedObjects || [];
  const errorText = activeTask?.failureReason || statusError;

  const stageStatus = (stage: string) => {
    if (isFailed) return { icon: 'cancel', label: 'Halted', tone: 'text-error' };
    if (isVerifiedComplete) return { icon: 'check_circle', label: 'Done', tone: 'text-primary' };
    if (stage === currentStage) return { icon: 'progress_activity', label: 'Active', tone: 'text-primary' };
    return { icon: 'radio_button_unchecked', label: 'Queued', tone: 'text-on-surface-variant' };
  };

  const handleCancel = async () => {
    if (isCancelling) return;
    setIsCancelling(true);
    try {
      const task = await api.cancelTask(taskId);
      setActiveTask(task);
      wsManager.disconnect();
    } catch (err: any) {
      setStatusError(err?.message || 'Cancellation request failed.');
    } finally {
      setIsCancelling(false);
    }
  };

  return (
    <AppLayout
      title="Processing Stream"
      activeTaskName={activeTaskFilename}
      isProcessing={isProcessing}
      onCancelProcess={handleCancel}
    >
      <div className="max-w-container-max mx-auto w-full flex flex-col gap-xl">
        {errorText && (
          <div className="p-md bg-error/10 border border-error/30 rounded-xl text-error flex items-start gap-sm text-sm">
            <span className="material-symbols-outlined text-[18px]">error</span>
            <div className="flex flex-col">
              <span className="font-label-md">
                {isFailed ? 'Processing did not complete' : 'Status unavailable'}
              </span>
              <span className="font-body-md">{errorText}</span>
            </div>
          </div>
        )}

        {isCancelling && (
          <div className="p-sm bg-tertiary/10 border border-tertiary/30 rounded-lg text-tertiary text-xs">
            Cancellation requested. The worker stops between frames and deletes any partial output.
          </div>
        )}

        {wsStatus !== 'CONNECTED' && isProcessing && (
          <div className="p-sm bg-tertiary/10 border border-tertiary/30 rounded-lg text-tertiary text-xs flex items-center gap-sm">
            <span className="material-symbols-outlined text-[16px] animate-spin">sync</span>
            <span>
              WebSocket telemetry: {wsStatus}. Task status is still polled from the backend.
            </span>
          </div>
        )}

        <div className="grid grid-cols-12 gap-xl flex-grow">
          <div className="col-span-12 lg:col-span-8 flex flex-col gap-lg">
            <DualStreamViewer
              detectedEntities={detectedObjects}
              cameraName={`Task #${taskId.substring(0, 10)}`}
              isRedacting={isProcessing}
              statusLabel={isVerifiedComplete ? 'Verified' : status || 'Processing'}
            />

            <MetricsBar metrics={metrics} detectedCount={metrics.masksTotal} />

            <TelemetryConsole logs={logs} autoScroll />
          </div>

          <div className="col-span-12 lg:col-span-4 flex flex-col gap-lg">
            <div className="glass-panel border border-outline-variant rounded-2xl p-lg flex flex-col gap-lg shadow-xl">
              <div className="flex flex-col gap-sm">
                <div className="flex justify-between items-end">
                  <h2 className="font-label-md font-bold text-on-surface">Overall Progress</h2>
                  <span className="text-headline-lg font-display font-bold text-primary leading-none">
                    {progressLabel}
                  </span>
                </div>
                <div className="h-3 w-full bg-surface-container-highest rounded-full overflow-hidden border border-outline-variant relative">
                  <div
                    className="h-full bg-primary progress-shimmer rounded-full shadow-glow transition-all duration-300"
                    style={{ width: `${progressWidth}%` }}
                  ></div>
                </div>
                <span className="font-label-sm text-on-surface-variant">
                  {metrics.processedFrames > 0 && metrics.totalFrames > 0
                    ? `${metrics.processedFrames} of ${metrics.totalFrames} frames processed`
                    : 'Exact progress has not been reported yet.'}
                  {metrics.elapsedSeconds > 0 ? ` • ${metrics.elapsedSeconds}s elapsed` : ''}
                </span>
              </div>

              <div className="flex flex-col gap-md">
                <h3 className="font-label-sm text-on-surface-variant uppercase tracking-widest border-b border-outline-variant pb-xs">
                  Pipeline Status
                </h3>

                <div className="flex flex-col gap-sm">
                  {['Detection', 'Tracking', currentStage].map((stage, index) => {
                    const state = stageStatus(stage);
                    const isCurrent = !isFailed && stage === currentStage;
                    return (
                      <div
                        key={`${stage}-${index}`}
                        className={`flex items-center gap-md p-sm rounded-lg border ${
                          isFailed
                            ? 'bg-error/5 border-error/30'
                            : isCurrent
                              ? 'bg-surface-container-highest border-primary/50'
                              : 'bg-surface-container-low border-outline-variant'
                        }`}
                      >
                        <span className={`material-symbols-outlined text-body-md ${state.tone}`}>
                          {state.icon}
                        </span>
                        <span className={`font-label-md text-on-surface ${isCurrent ? 'font-bold' : ''}`}>
                          {stage}
                        </span>
                        <span className={`ml-auto font-label-sm ${state.tone}`}>{state.label}</span>
                      </div>
                    );
                  })}
                </div>
              </div>

              <div className="bg-surface-container-lowest border border-outline-variant rounded-xl p-md flex flex-col gap-xs">
                <h4 className="font-label-sm text-on-surface-variant mb-sm flex items-center gap-sm">
                  <span className="material-symbols-outlined text-xs">list</span>
                  TASK DETAILS
                </h4>
                <div className="flex justify-between text-xs font-label-sm">
                  <span className="text-on-surface-variant">Blur style</span>
                  <span className="text-on-surface capitalize">{activeTask?.blurStyle || '—'}</span>
                </div>
                <div className="flex justify-between text-xs font-label-sm">
                  <span className="text-on-surface-variant">Intensity</span>
                  <span className="text-on-surface">
                    {activeTask?.blurIntensity ? `${activeTask.blurIntensity}%` : '—'}
                  </span>
                </div>
                <div className="flex justify-between text-xs font-label-sm">
                  <span className="text-on-surface-variant">Face regions masked</span>
                  <span className="text-on-surface">{metrics.masksTotal.toLocaleString()}</span>
                </div>
                <div className="flex justify-between text-xs font-label-sm">
                  <span className="text-on-surface-variant">Debug frames</span>
                  <span className="text-on-surface">
                    {activeTask?.debugMode ? metrics.debugFramesSaved : 'off'}
                  </span>
                </div>
              </div>

              {isFailed && (
                <button
                  type="button"
                  onClick={() => navigate('/upload')}
                  className="w-full py-sm rounded-lg border border-outline-variant font-label-md hover:bg-surface-variant transition-colors cursor-pointer"
                >
                  Back to Upload
                </button>
              )}
            </div>
          </div>
        </div>

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
