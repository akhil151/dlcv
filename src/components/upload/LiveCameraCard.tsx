import React from 'react';

export interface LiveCameraCardProps {
  onRecordedFileReady?: (file: File) => void;
}

export const LiveCameraCard: React.FC<LiveCameraCardProps> = () => {
  return (
    <div className="glass-panel p-xl rounded-2xl border border-outline-variant flex flex-col gap-md w-full select-none">
      <div className="flex justify-between items-center border-b border-outline-variant pb-md">
        <div className="flex items-center gap-sm">
          <span className="material-symbols-outlined text-primary text-[24px]">videocam</span>
          <div>
            <h3 className="font-headline-md text-on-surface font-bold">Live Camera Stream</h3>
            <p className="font-label-sm text-on-surface-variant">
              Real-time WebRTC / RTSP camera feed ingest for instant neural redaction
            </p>
          </div>
        </div>
        <div className="flex items-center gap-xs">
          <span className="bg-primary/10 text-primary border border-primary/20 px-sm py-[2px] rounded-full text-xs font-bold uppercase flex items-center gap-xs">
            <span className="w-2 h-2 rounded-full bg-primary animate-pulse"></span>
            STANDBY
          </span>
        </div>
      </div>

      {/* Video Stream Placeholder Box */}
      <div className="relative aspect-video bg-surface-container-lowest rounded-xl overflow-hidden border border-outline-variant/60 flex items-center justify-center group">
        <div className="flex flex-col items-center gap-md text-center p-xl">
          <div className="w-16 h-16 rounded-full bg-surface-container-high border border-outline-variant flex items-center justify-center text-primary shadow-lg">
            <span className="material-symbols-outlined text-[36px]">videocam</span>
          </div>
          <div className="flex flex-col gap-xs max-w-md">
            <h4 className="font-headline-md text-on-surface font-bold">Live Camera Ingest Prepared</h4>
            <p className="font-body-md text-on-surface-variant text-label-md">
              Connect a WebRTC stream or USB camera feed for live frame-by-frame AI privacy blurring.
            </p>
          </div>
          <button
            type="button"
            className="mt-xs bg-primary/20 text-primary border border-primary/30 px-lg py-sm rounded-lg font-label-md font-bold hover:bg-primary/30 transition-all cursor-pointer flex items-center gap-xs"
          >
            <span className="material-symbols-outlined text-[18px]">sensors</span>
            <span>Connect Live Stream Feed</span>
          </button>
        </div>
      </div>
    </div>
  );
};
