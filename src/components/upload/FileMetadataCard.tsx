import React from 'react';
import type { VideoMetadata } from '../../types';

export interface FileMetadataCardProps {
  metadata?: VideoMetadata | null;
  selectedFileName?: string;
  selectedFileSize?: number;
  isUploading?: boolean;
  autoRedactFaces?: boolean;
  redactLicensePlates?: boolean;
  onToggleFaces?: () => void;
  onTogglePlates?: () => void;
  onStartProcess?: () => void;
}

export const FileMetadataCard: React.FC<FileMetadataCardProps> = ({
  metadata,
  selectedFileName,
  selectedFileSize,
  isUploading = false,
  autoRedactFaces = true,
  redactLicensePlates = true,
  onToggleFaces,
  onTogglePlates,
  onStartProcess,
}) => {
  const formatBytes = (bytes: number): string => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return `${parseFloat((bytes / Math.pow(k, i)).toFixed(1))} ${sizes[i]}`;
  };

  const hasFile = Boolean(metadata || selectedFileName);

  return (
    <aside className="w-full lg:w-[340px] flex flex-col gap-lg shrink-0">
      {/* File Information Section */}
      <section className="bg-surface-container-low border border-outline-variant rounded-xl p-lg flex flex-col gap-md">
        <h4 className="font-label-md text-label-md text-primary uppercase tracking-widest">
          File Information
        </h4>

        {!hasFile ? (
          /* Placeholder State (Before Selection) */
          <div className="py-xl flex flex-col items-center justify-center text-center gap-sm" id="file-meta-empty">
            <span className="material-symbols-outlined text-outline-variant text-[32px]">draft</span>
            <p className="font-body-md text-body-md text-on-surface-variant italic">
              Select a file to view detailed metadata and estimated processing time.
            </p>
          </div>
        ) : (
          /* File Details (Active State) */
          <div className="flex flex-col gap-md" id="file-meta-active">
            <div className="p-md bg-surface-container rounded-lg border border-outline-variant">
              <p className="font-label-sm text-label-sm text-on-surface-variant mb-xs">Filename</p>
              <p className="font-body-md text-body-md text-on-surface font-semibold truncate" id="meta-name">
                {metadata?.filename || selectedFileName}
              </p>
            </div>

            <div className="grid grid-cols-2 gap-md">
              <div className="p-md bg-surface-container rounded-lg border border-outline-variant">
                <p className="font-label-sm text-label-sm text-on-surface-variant mb-xs">Size</p>
                <p className="font-body-md text-body-md text-on-surface" id="meta-size">
                  {metadata
                    ? formatBytes(metadata.fileSizeBytes)
                    : selectedFileSize
                    ? formatBytes(selectedFileSize)
                    : 'Calculating...'}
                </p>
              </div>

              <div className="p-md bg-surface-container rounded-lg border border-outline-variant">
                <p className="font-label-sm text-label-sm text-on-surface-variant mb-xs">Format</p>
                <p className="font-body-md text-body-md text-on-surface" id="meta-format">
                  {metadata?.format || 'H.264 / MP4'}
                </p>
              </div>
            </div>

            <div className="p-md bg-surface-container rounded-lg border border-outline-variant">
              <p className="font-label-sm text-label-sm text-on-surface-variant mb-xs">Resolution</p>
              <p className="font-body-md text-body-md text-on-surface" id="meta-res">
                {metadata?.resolution || '3840 x 2160 (4K)'}
              </p>
            </div>

            <div className="p-md bg-surface-container rounded-lg border border-outline-variant">
              <p className="font-label-sm text-label-sm text-on-surface-variant mb-xs">Estimated Time</p>
              <p className="font-body-md text-body-md text-primary font-bold">
                {metadata?.estimatedProcessingSeconds
                  ? `~ ${metadata.estimatedProcessingSeconds}s`
                  : '~ 4 min 12 sec'}
              </p>
            </div>
          </div>
        )}
      </section>

      {/* Processing Options & Action Section */}
      <section className="flex flex-col gap-md">
        <div className="flex items-center gap-sm">
          <input
            id="auto-redact"
            type="checkbox"
            checked={autoRedactFaces}
            onChange={onToggleFaces}
            className="w-4 h-4 rounded border-outline-variant bg-surface text-primary focus:ring-primary cursor-pointer"
          />
          <label className="font-label-md text-label-md text-on-surface cursor-pointer" htmlFor="auto-redact">
            Auto-detect all faces
          </label>
        </div>

        <div className="flex items-center gap-sm">
          <input
            id="license-plates"
            type="checkbox"
            checked={redactLicensePlates}
            onChange={onTogglePlates}
            className="w-4 h-4 rounded border-outline-variant bg-surface text-primary focus:ring-primary cursor-pointer"
          />
          <label className="font-label-md text-label-md text-on-surface cursor-pointer" htmlFor="license-plates">
            Redact license plates
          </label>
        </div>

        <button
          id="btn-process"
          type="button"
          disabled={!hasFile || isUploading}
          onClick={onStartProcess}
          className={`mt-md w-full py-md rounded-lg font-headline-md font-bold text-headline-md transition-all active:scale-95 flex items-center justify-center gap-sm ${
            !hasFile || isUploading
              ? 'bg-outline-variant text-on-surface-variant cursor-not-allowed'
              : 'bg-primary text-on-primary hover:shadow-lg hover:shadow-primary/20 cursor-pointer'
          }`}
        >
          {isUploading ? (
            <>
              <span className="material-symbols-outlined animate-spin">sync</span>
              <span>Initializing...</span>
            </>
          ) : (
            'Start Processing'
          )}
        </button>

        <p className="font-label-sm text-label-sm text-center text-on-surface-variant opacity-50">
          AI identification engine will initiate on start
        </p>
      </section>
    </aside>
  );
};
