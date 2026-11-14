import React, { useEffect, useState } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import { AppLayout } from '../components/layout/AppLayout';
import { SummaryBentoGrid } from '../components/results/SummaryBentoGrid';
import { ExportDownloadHub } from '../components/results/ExportDownloadHub';
import { ResultVideoPlayer } from '../components/results/ResultVideoPlayer';
import { useAppStore } from '../store/useAppStore';
import { api } from '../services/api';

type ResultView = 'loading' | 'verified' | 'processing' | 'failed' | 'cancelled' | 'error';

export const ResultsPage: React.FC = () => {
  const navigate = useNavigate();
  const { taskId = '' } = useParams();

  const { activeSummary, setActiveSummary } = useAppStore();

  const [view, setView] = useState<ResultView>('loading');
  const [failureReason, setFailureReason] = useState<string | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  useEffect(() => {
    if (!taskId) {
      setView('error');
      setErrorMessage('No task identifier was provided.');
      return;
    }

    let cancelled = false;

    const load = async () => {
      setView('loading');
      setErrorMessage(null);
      setFailureReason(null);
      // Never let stale state from a previous task look like a success.
      setActiveSummary(null);
      try {
        const task = await api.getTaskStatus(taskId);
        if (cancelled) return;

        if (task.status === 'FAILED') {
          setFailureReason(task.failureReason || 'Processing failed.');
          setView('failed');
          return;
        }
        if (task.status === 'CANCELLED') {
          setFailureReason(task.failureReason || 'Processing was cancelled.');
          setView('cancelled');
          return;
        }
        if (task.status !== 'COMPLETED') {
          setView('processing');
          return;
        }
        if (task.outputVerified !== true) {
          setFailureReason('The task completed without a verified protected output.');
          setView('failed');
          return;
        }
        setView('verified');
        try {
          const summary = await api.getTaskSummary(taskId);
          if (!cancelled) setActiveSummary(summary);
        } catch (summaryErr: any) {
          if (!cancelled) {
            setErrorMessage(summaryErr?.message || 'Unable to load the result summary.');
          }
        }
      } catch (err: any) {
        if (!cancelled) {
          setErrorMessage(err?.message || 'Unable to read task status from the server.');
          setView('error');
        }
      }
    };

    load();
    return () => {
      cancelled = true;
    };
  }, [taskId, setActiveSummary]);

  const isVerified = view === 'verified';
  const protectedUrl = isVerified ? api.getDownloadUrl(taskId) : undefined;
  const originalUrl = isVerified ? api.getOriginalUrl(taskId) : undefined;

  const methodLabels: Record<string, string> = {
    gaussian: 'Gaussian Blur',
    pixelate: 'Pixelation',
    blackout: 'Solid Blackout',
  };
  const methodLabel = activeSummary?.blurStyle
    ? methodLabels[activeSummary.blurStyle] || activeSummary.blurStyle
    : null;
  const methodBadge = methodLabel
    ? `${methodLabel}${activeSummary?.blurIntensity ? ` · ${activeSummary.blurIntensity}%` : ''}`
    : null;
  const resolution = activeSummary?.outputResolution;
  const verifiedFrames = activeSummary?.verifiedFrameCount || activeSummary?.framesProcessed;
  const originalMeta = resolution ? `Original · ${resolution}` : 'Original source';
  const protectedMeta =
    [resolution, verifiedFrames ? `${verifiedFrames.toLocaleString()} frames verified` : null]
      .filter(Boolean)
      .join(' · ') || 'Protected output';

  const handleDownloadMaster = () => {
    if (!isVerified) return; // downloads only for verified outputs
    window.open(api.getDownloadUrl(taskId), '_blank');
  };

  const stateBadge = (() => {
    switch (view) {
      case 'verified':
        return { icon: 'check_circle', label: 'Verified Output', tone: 'primary' };
      case 'processing':
        return { icon: 'hourglass_top', label: 'Still Processing', tone: 'tertiary' };
      case 'cancelled':
        return { icon: 'cancel', label: 'Cancelled', tone: 'error' };
      case 'failed':
        return { icon: 'report', label: 'Processing Failed', tone: 'error' };
      case 'error':
        return { icon: 'error', label: 'Result Unavailable', tone: 'error' };
      default:
        return { icon: 'hourglass_empty', label: 'Loading', tone: 'tertiary' };
    }
  })();

  const toneClasses: Record<string, string> = {
    primary: 'bg-primary/10 border-primary/20 text-primary',
    tertiary: 'bg-tertiary/10 border-tertiary/20 text-tertiary',
    error: 'bg-error/10 border-error/20 text-error',
  };

  return (
    <AppLayout title="Redaction Results">
      <div className="max-w-5xl mx-auto w-full pb-20">
        <header className="mb-xl text-center md:text-left flex flex-col md:flex-row md:items-end justify-between gap-md">
          <div>
            <div className="flex flex-wrap items-center gap-sm mb-md">
              <div className={`inline-flex items-center gap-sm px-md py-xs rounded-full border ${toneClasses[stateBadge.tone]}`}>
                <span className="material-symbols-outlined text-[16px] filled">{stateBadge.icon}</span>
                <span className="font-label-sm uppercase tracking-widest">{stateBadge.label}</span>
              </div>
              {isVerified && methodBadge && (
                <div className="inline-flex items-center gap-sm px-md py-xs rounded-full border border-outline-variant bg-surface-container text-on-surface-variant">
                  <span className="material-symbols-outlined text-[16px]">blur_on</span>
                  <span className="font-label-sm">{methodBadge}</span>
                </div>
              )}
            </div>
            <h2 className="font-display text-display mb-sm font-bold text-on-surface">
              {isVerified ? 'Protected Video Ready' : view === 'processing' ? 'Processing In Progress' : 'Result Not Available'}
            </h2>
            <p className="font-body-lg text-on-surface-variant max-w-2xl">
              {isVerified
                ? 'Every input frame was re-decoded from the encoded output with the expected resolution and frame count before this download was enabled.'
                : 'This page only presents results for tasks the backend has confirmed as completed and verified.'}
            </p>
          </div>

          <div className="flex flex-wrap gap-md shrink-0">
            <button
              type="button"
              onClick={handleDownloadMaster}
              disabled={!isVerified}
              className={`px-xl h-14 rounded-xl font-label-md flex items-center gap-md transition-all font-bold ${
                isVerified
                  ? 'bg-primary text-on-primary hover:scale-[1.02] active:scale-95 shadow-lg shadow-primary/10 cursor-pointer'
                  : 'bg-surface-container-highest text-on-surface-variant cursor-not-allowed opacity-60'
              }`}
            >
              <span className="material-symbols-outlined">download</span>
              <span>{isVerified ? 'Download Protected Video' : 'Download Unavailable'}</span>
            </button>
          </div>
        </header>

        {view === 'loading' && (
          <div className="glass-panel rounded-2xl p-xl border border-outline-variant flex items-center gap-md">
            <div className="w-5 h-5 rounded-full border-2 border-primary border-t-transparent animate-spin"></div>
            <span className="font-label-md text-on-surface-variant">Reading verified task status…</span>
          </div>
        )}

        {(view === 'failed' || view === 'cancelled' || view === 'error') && (
          <div className="glass-panel rounded-2xl p-xl border border-error/30 bg-error/5 flex flex-col gap-sm">
            <div className="flex items-center gap-sm text-error">
              <span className="material-symbols-outlined">error</span>
              <span className="font-label-md font-bold">
                {view === 'cancelled' ? 'Processing was cancelled' : view === 'failed' ? 'Processing did not complete' : 'Result could not be loaded'}
              </span>
            </div>
            <p className="font-body-md text-on-surface-variant">
              {failureReason || errorMessage || 'No verified protected output is available for this task.'}
            </p>
            <button
              type="button"
              onClick={() => navigate('/upload')}
              className="self-start mt-sm border border-outline-variant text-on-surface px-lg py-sm rounded-lg font-label-md hover:bg-surface-variant transition-colors cursor-pointer"
            >
              Back to Upload
            </button>
          </div>
        )}

        {view === 'processing' && (
          <div className="glass-panel rounded-2xl p-xl border border-outline-variant flex flex-col gap-md">
            <div className="flex items-center gap-sm text-tertiary">
              <span className="material-symbols-outlined">hourglass_top</span>
              <span className="font-label-md font-bold">This task is still processing</span>
            </div>
            <p className="font-body-md text-on-surface-variant">
              No protected output exists yet. Follow the real progress on the live processing view.
            </p>
            <button
              type="button"
              onClick={() => navigate(`/processing/${taskId}`)}
              className="self-start border border-outline-variant text-on-surface px-lg py-sm rounded-lg font-label-md hover:bg-surface-variant transition-colors cursor-pointer"
            >
              View Live Processing
            </button>
          </div>
        )}

        {isVerified && (
          <>
            <section className="mb-xl">
              <div className="flex flex-wrap items-baseline justify-between gap-sm mb-md">
                <h3 className="font-display text-lg font-semibold text-on-surface">
                  Original vs. Protected
                </h3>
                <p className="font-label-sm text-on-surface-variant">
                  Same clip, same playback position — only the detected faces differ.
                </p>
              </div>
              <div className="grid gap-lg lg:grid-cols-2">
                <ResultVideoPlayer
                  title={activeSummary?.taskId ? `safeframe_${activeSummary.taskId}_original.mp4` : `safeframe_${taskId}_original.mp4`}
                  label="Original source"
                  meta={originalMeta}
                  videoUrl={originalUrl}
                />
                <ResultVideoPlayer
                  title={activeSummary?.taskId ? `safeframe_${activeSummary.taskId}_protected.mp4` : `safeframe_${taskId}_protected.mp4`}
                  label="Protected output"
                  meta={protectedMeta}
                  videoUrl={protectedUrl}
                  verified
                />
              </div>
            </section>

            {errorMessage && (
              <div className="mb-md p-md bg-tertiary/10 border border-tertiary/30 rounded-xl text-tertiary text-sm">
                Summary unavailable: {errorMessage}
              </div>
            )}

            <SummaryBentoGrid summary={activeSummary || undefined} />

            <ExportDownloadHub
              onDownloadProtected={handleDownloadMaster}
              onReEditClick={() => navigate('/upload')}
            />
          </>
        )}

        <footer className="fixed bottom-0 right-0 left-0 md:left-[240px] bg-surface-container-lowest border-t border-outline-variant py-md px-xl flex justify-between items-center z-40 select-none">
          <div className="flex items-center gap-xs text-on-surface-variant text-label-sm">
            <span className={`w-2 h-2 rounded-full ${isVerified ? 'bg-emerald-400' : 'bg-tertiary'}`}></span>
            <span>
              Task #{taskId} • {isVerified ? 'Verified — ready for export' : stateBadge.label}
            </span>
          </div>
          <p className="font-label-sm text-on-surface-variant opacity-70">
            © 2024 SafeFrame AI. Technical Authority in Video Privacy.
          </p>
        </footer>
      </div>
    </AppLayout>
  );
};
