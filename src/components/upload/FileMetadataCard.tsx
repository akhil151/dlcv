import React from 'react';
import type { VideoMetadata } from '../../types';

export interface FileMetadataCardProps {
  metadata?: VideoMetadata | null;
  selectedFileName?: string;
  selectedFileSize?: number;
  uploadProgressPct?: number;
  isUploading?: boolean;
  canStart?: boolean;
  disabledReason?: string | null;
  onStartProcess?: () => void;
}

export const FileMetadataCard: React.FC<FileMetadataCardProps> = ({
  metadata,
  selectedFileName,
  selectedFileSize,
  uploadProgressPct = 0,
  isUploading = false,
  canStart = false,
  disabledReason,
  onStartProcess,
}) => {
  const formatBytes = (bytes: number): string => {
    if (!bytes) return '—';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return `${parseFloat((bytes / Math.pow(k, i)).toFixed(1))} ${sizes[i]}`;
  };

  const formatDuration = (seconds: number): string => {
    if (!seconds) return '—';
    const mins = Math.floor(seconds / 60);
    const rest = seconds % 60;
    return mins > 0 ? `${mins}m ${rest}s` : `${rest}s`;
  };

  const hasFile = Boolean(metadata || selectedFileName);

  return (
    <aside className="w-full flex flex-col gap-lg shrink-0">
      <section className="bg-surface-container-low border border-outline-variant rounded-xl p-lg flex flex-col gap-md">
        <h4 className="font-label-md text-label-md text-primary uppercase tracking-widest">
          File Information
        </h4>

        {!hasFile ? (
          <div className="py-xl flex flex-col items-center justify-center text-center gap-sm" id="file-meta-empty">
            <span className="material-symbols-outlined text-outline-variant text-[32px]">draft</span>
            <p className="font-body-md text-body-md text-on-surface-variant italic">
              Select a file to inspect its real metadata before processing.
            </p>
          </div>
        ) : (
          <div className="flex flex-col gap-md" id="file-meta-active">
            <div className="p-md bg-surface-container rounded-lg border border-outline-variant">
              <p className="font-label-sm text-label-sm text-on-surface-variant mb-xs">Filename</p>
              <p className="font-body-md text-body-md text-on-surface font-semibold break-all" id="meta-name">
                {metadata?.filename || selectedFileName}
              </p>
            </div>

            <div className="grid grid-cols-2 gap-md">
              <div className="p-md bg-surface-container rounded-lg border border-outline-variant">
                <p className="font-label-sm text-label-sm text-on-surface-variant mb-xs">Size</p>
                <p className="font-body-md text-body-md text-on-surface" id="meta-size">
                  {metadata ? formatBytes(metadata.fileSizeBytes) : formatBytes(selectedFileSize || 0)}
                </p>
              </div>

              <div className="p-md bg-surface-container rounded-lg border border-outline-variant">
                <p className="font-label-sm text-label-sm text-on-surface-variant mb-xs">Container</p>
                <p className="font-body-md text-body-md text-on-surface" id="meta-format">
                  {metadata?.format || 'Verifying…'}
                </p>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-md">
              <div className="p-md bg-surface-container rounded-lg border border-outline-variant">
                <p className="font-label-sm text-label-sm text-on-surface-variant mb-xs">Resolution</p>
                <p className="font-body-md text-body-md text-on-surface" id="meta-res">
                  {metadata?.resolution || 'Verifying…'}
                </p>
              </div>
              <div className="p-md bg-surface-container rounded-lg border border-outline-variant">
                <p className="font-label-sm text-label-sm text-on-surface-variant mb-xs">Frame rate</p>
                <p className="font-body-md text-body-md text-on-surface">
                  {metadata?.fps ? `${metadata.fps.toFixed(2)} fps` : 'Verifying…'}
                </p>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-md">
              <div className="p-md bg-surface-container rounded-lg border border-outline-variant">
                <p className="font-label-sm text-label-sm text-on-surface-variant mb-xs">Duration</p>
                <p className="font-body-md text-body-md text-on-surface">
                  {metadata ? formatDuration(metadata.durationSeconds) : 'Verifying…'}
                </p>
              </div>
              <div className="p-md bg-surface-container rounded-lg border border-outline-variant">
                <p className="font-label-sm text-label-sm text-on-surface-variant mb-xs">Frames</p>
                <p className="font-body-md text-body-md text-on-surface">
                  {metadata?.frameCount ? metadata.frameCount.toLocaleString() : 'Verifying…'}
                </p>
              </div>
            </div>

            <div className="p-md bg-surface-container rounded-lg border border-outline-variant">
              <p className="font-label-sm text-label-sm text-on-surface-variant mb-xs">
                Estimated processing time (CPU, local)
              </p>
              <p className="font-body-md text-body-md text-primary font-bold">
                {metadata?.estimatedProcessingSeconds
                  ? `~ ${formatDuration(metadata.estimatedProcessingSeconds)}`
                  : '—'}
              </p>
            </div>

            {isUploading && (
              <div className="flex flex-col gap-xs">
                <div className="flex justify-between font-label-sm text-on-surface-variant">
                  <span>{uploadProgressPct >= 100 ? 'Verifying on server…' : 'Uploading…'}</span>
                  <span>{uploadProgressPct}%</span>
                </div>
                <div className="h-2 w-full bg-surface-container-highest rounded-full overflow-hidden">
                  <div
                    className="h-full bg-primary transition-all duration-200"
                    style={{ width: `${Math.min(100, Math.max(0, uploadProgressPct))}%` }}
                  />
                </div>
              </div>
            )}
          </div>
        )}
      </section>

      <section className="flex flex-col gap-md">
        <button
          id="btn-process"
          type="button"
          disabled={!canStart || isUploading}
          onClick={onStartProcess}
          className={`w-full py-md rounded-lg font-headline-md font-bold text-headline-md transition-all active:scale-95 flex items-center justify-center gap-sm ${
            !canStart || isUploading
              ? 'bg-outline-variant text-on-surface-variant cursor-not-allowed'
              : 'bg-primary text-on-primary hover:shadow-lg hover:shadow-primary/20 cursor-pointer'
          }`}
        >
          {isUploading ? (
            <>
              <span className="material-symbols-outlined animate-spin">sync</span>
              <span>Uploading…</span>
            </>
          ) : (
            'Start Face Redaction'
          )}
        </button>

        <p className="font-label-sm text-label-sm text-center text-on-surface-variant opacity-70">
          {disabledReason ||
            'Processing begins only after the backend confirms the detector and encoder are ready.'}
        </p>
      </section>
    </aside>
  );
};
