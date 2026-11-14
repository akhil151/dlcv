import React from 'react';
import type { DetectedEntity } from '../../types';

export interface DualStreamViewerProps {
  detectedEntities?: DetectedEntity[];
  cameraName?: string;
  isRedacting?: boolean;
  statusLabel?: string;
}

/**
 * The backend does not stream decoded frames over the WebSocket, so this panel
 * deliberately does not pretend to show a live video feed. It shows the real
 * detection telemetry that is received, and states plainly that frame preview
 * is unavailable until the protected output is verified.
 *
 * Debug frame images (annotated / original / protected) are written to the
 * backend's storage/debug/<task_id>/ directory when debug mode is enabled.
 */
export const DualStreamViewer: React.FC<DualStreamViewerProps> = ({
  detectedEntities = [],
  cameraName = 'Input stream',
  isRedacting = false,
  statusLabel = 'Processing',
}) => {
  return (
    <div className="flex flex-col gap-sm">
      <div className="flex justify-between items-center px-sm">
        <span className="font-label-md text-primary font-bold">Processing Pipeline</span>
        <span className="bg-primary/10 text-primary border border-primary/20 text-[10px] px-sm py-[2px] rounded font-bold uppercase">
          {statusLabel}
        </span>
      </div>

      <div className="video-container relative rounded-xl overflow-hidden border border-primary/30 flex flex-col items-center justify-center gap-md p-xl min-h-[240px] bg-surface-container-lowest">
        {isRedacting && <div className="scanning-line"></div>}
        <span className="material-symbols-outlined text-primary text-[40px]">movie</span>
        <p className="font-body-md text-on-surface text-center">
          No live frame preview is streamed by the backend.
        </p>
        <p className="font-label-sm text-on-surface-variant text-center max-w-md">
          {isRedacting
            ? 'Frames are being decoded, scanned and re-encoded. The protected video becomes available once every frame has been verified.'
            : 'The protected output is available on the results page after verification.'}
        </p>
        <div className="flex items-center gap-sm font-label-sm text-on-surface-variant">
          <span className="material-symbols-outlined text-[16px]">videocam</span>
          <span>{cameraName}</span>
        </div>
      </div>

      <div className="flex flex-wrap items-center gap-xs px-sm">
        <span className="font-label-sm text-on-surface-variant">
          Detected now ({detectedEntities.length}):
        </span>
        {detectedEntities.length === 0 ? (
          <span className="font-label-sm text-on-surface-variant opacity-60">none</span>
        ) : (
          detectedEntities.slice(0, 12).map((entity, index) => (
            <span
              key={entity.id || index}
              className="bg-primary/10 text-primary border border-primary/20 rounded px-sm py-[2px] font-label-sm"
            >
              {entity.label || entity.category} · {Math.round(entity.confidence * 100)}%
            </span>
          ))
        )}
      </div>
    </div>
  );
};
