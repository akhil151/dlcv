import React from 'react';
import type { DetectedEntity } from '../../types';

export interface DualStreamViewerProps {
  originalStreamUrl?: string;
  protectedStreamUrl?: string;
  detectedEntities?: DetectedEntity[];
  cameraName?: string;
  isRedacting?: boolean;
}

export const DualStreamViewer: React.FC<DualStreamViewerProps> = ({
  originalStreamUrl = 'https://lh3.googleusercontent.com/aida-public/AB6AXuCMP_QAK-JmNjdlEsjviCiWD39H16I6VnmlWDY-OFiVw7fRmqODy6G2BflKglMNEl8ymf4FVR-uo5QCx3wlGF3hgd4lbXflBfOhuP1cNGkQprx9BgLbmVLFJoTUw6gstT5NdgZtpragM4ahvo8PDnkWe0v_ln6-XcJNhijjgF5AXUZVa-5rP6pMyD7IYp8SP5TUOhEdi-NiiWip_0bcdBIcFhbIqxBT_5h5ulxY5DmcXERrXAbUWYe4XVL7x9io2w6LwNUrLiK_0eQ',
  protectedStreamUrl = 'https://lh3.googleusercontent.com/aida-public/AB6AXuBkuYkuj7RgyL3ft3S6M0rHrDkvgcEWbx2lb4vdJm1NF-h2vE1aY4i2ZqsDKvPkl5psD6gWV3hc18maaTa4vtXFYpjtSl0T-WMs_2OOpKby-F1TCjk2Hv50oqIegJzz2Gwzl6Ldup63cxM2LbDKNvNHVpQ6DpdkQQ93xqplQasvW7M_jObIGpRU7vaDAsYksRj3kcrWdAQAIYx2vRR84CLWsmHMq78RBaTPapbz5qm-hA41nYdzkNirZZyi7-y-P1KS76GitWeZ3rI',
  detectedEntities = [],
  cameraName = 'Cam_04_Sec_Beta',
  isRedacting = true,
}) => {
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-md">
      {/* Original Preview Stream */}
      <div className="flex flex-col gap-sm">
        <div className="flex justify-between items-center px-sm">
          <span className="font-label-md text-on-surface-variant">Original Stream</span>
          <span className="font-label-sm text-on-surface-variant opacity-50">RAW INPUT</span>
        </div>
        <div className="video-container relative rounded-xl overflow-hidden border border-outline-variant">
          <img
            src={originalStreamUrl}
            alt="Original Video Feed"
            className="w-full h-full object-cover grayscale opacity-60"
          />
          <div className="absolute inset-0 bg-gradient-to-t from-black/60 to-transparent"></div>
          <div className="absolute bottom-4 left-4 font-label-sm text-white/70 flex items-center gap-sm">
            <span className="material-symbols-outlined scale-75">videocam</span>
            <span>{cameraName}</span>
          </div>
        </div>
      </div>

      {/* Protected Preview Stream */}
      <div className="flex flex-col gap-sm">
        <div className="flex justify-between items-center px-sm">
          <span className="font-label-md text-primary font-bold">Protected Output</span>
          <span className="bg-primary text-on-primary text-[10px] px-sm py-[2px] rounded font-bold uppercase">
            LIVE REDACTION
          </span>
        </div>
        <div className="video-container relative rounded-xl overflow-hidden border border-primary/40 shadow-[0_0_20px_rgba(173,198,255,0.1)]">
          <img
            src={protectedStreamUrl}
            alt="Protected Redacted Feed"
            className="w-full h-full object-cover"
          />
          {isRedacting && <div className="scanning-line"></div>}

          {/* Dynamic Bounding Boxes */}
          {detectedEntities.length > 0 ? (
            detectedEntities.map((obj, idx) => (
              <div
                key={obj.id || idx}
                style={{
                  top: `${obj.bbox.y}%`,
                  left: `${obj.bbox.x}%`,
                  width: `${obj.bbox.width}%`,
                  height: `${obj.bbox.height}%`,
                }}
                className="absolute border border-primary bg-primary/10 rounded-sm flex items-center justify-center"
              >
                <div className="absolute -top-6 left-0 text-[10px] px-1 font-bold bg-primary text-on-primary">
                  {obj.label || obj.category.toUpperCase()} [{Math.round(obj.confidence * 100)}%]
                </div>
              </div>
            ))
          ) : (
            <>
              {/* Default Bounding Boxes per processing.html */}
              <div className="absolute top-1/4 left-1/3 w-16 h-40 border border-primary bg-primary/10 rounded-sm">
                <div className="absolute -top-6 left-0 bg-primary text-on-primary text-[10px] px-1 font-bold">
                  FACE #12 [98%]
                </div>
              </div>
              <div className="absolute top-1/2 left-2/3 w-12 h-24 border border-primary bg-primary/10 rounded-sm">
                <div className="absolute -top-6 left-0 bg-primary text-on-primary text-[10px] px-1 font-bold">
                  FACE #13 [94%]
                </div>
              </div>
              <div className="absolute bottom-1/4 right-1/4 w-32 h-16 border border-tertiary-container bg-tertiary-container/10 rounded-sm">
                <div className="absolute -top-6 left-0 bg-tertiary-container text-white text-[10px] px-1 font-bold">
                  PLATE [88%]
                </div>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
};
