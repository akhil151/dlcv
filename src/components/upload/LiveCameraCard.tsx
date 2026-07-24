import React, { useRef, useState } from 'react';

export interface LiveCameraCardProps {
  onRecordedFileReady?: (file: File) => void;
}

export const LiveCameraCard: React.FC<LiveCameraCardProps> = ({ onRecordedFileReady }) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const recordedChunksRef = useRef<Blob[]>([]);

  const [isCameraActive, setIsCameraActive] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [cameraError, setCameraError] = useState<string | null>(null);

  const startCamera = async () => {
    setCameraError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1920 }, height: { ideal: 1080 } },
        audio: false,
      });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.play();
      }
      setIsCameraActive(true);
    } catch (err: any) {
      console.error('[LiveCameraCard] Failed to access webcam:', err);
      setCameraError('Camera access denied or unavailable on this device.');
    }
  };

  const stopCamera = () => {
    if (videoRef.current && videoRef.current.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream;
      stream.getTracks().forEach((track) => track.stop());
      videoRef.current.srcObject = null;
    }
    setIsCameraActive(false);
    setIsRecording(false);
  };

  const startRecording = () => {
    if (!videoRef.current || !videoRef.current.srcObject) return;
    const stream = videoRef.current.srcObject as MediaStream;
    recordedChunksRef.current = [];

    const recorder = new MediaRecorder(stream, { mimeType: 'video/webm' });
    recorder.ondataavailable = (e) => {
      if (e.data.size > 0) {
        recordedChunksRef.current.push(e.data);
      }
    };
    recorder.onstop = () => {
      const blob = new Blob(recordedChunksRef.current, { type: 'video/webm' });
      const file = new File([blob], `live_camera_capture_${Date.now()}.webm`, {
        type: 'video/webm',
      });
      if (onRecordedFileReady) {
        onRecordedFileReady(file);
      }
    };

    recorder.start(1000);
    mediaRecorderRef.current = recorder;
    setIsRecording(true);
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
      mediaRecorderRef.current.stop();
    }
    setIsRecording(false);
  };

  return (
    <div className="glass-panel p-xl rounded-2xl border border-outline-variant flex flex-col gap-md w-full">
      <div className="flex justify-between items-center border-b border-outline-variant pb-md">
        <div className="flex items-center gap-sm">
          <span className="material-symbols-outlined text-primary text-[24px]">videocam</span>
          <div>
            <h3 className="font-headline-md text-on-surface font-bold">Live Camera Feed</h3>
            <p className="font-label-sm text-on-surface-variant">
              Capture live webcam feed for real-time AI privacy redaction
            </p>
          </div>
        </div>
        <div className="flex items-center gap-xs">
          {isRecording ? (
            <span className="bg-error/10 text-error border border-error/20 px-sm py-[2px] rounded-full text-xs font-bold uppercase animate-pulse flex items-center gap-xs">
              <span className="w-2 h-2 rounded-full bg-error"></span>
              REC LIVE
            </span>
          ) : isCameraActive ? (
            <span className="bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 px-sm py-[2px] rounded-full text-xs font-bold flex items-center gap-xs">
              <span className="w-2 h-2 rounded-full bg-emerald-400"></span>
              CAM READY
            </span>
          ) : (
            <span className="bg-surface-container-high text-on-surface-variant px-sm py-[2px] rounded-full text-xs">
              INACTIVE
            </span>
          )}
        </div>
      </div>

      {cameraError && (
        <div className="p-sm bg-error/10 border border-error/20 rounded-lg text-error text-xs">
          {cameraError}
        </div>
      )}

      {/* Video Stream Canvas */}
      <div className="relative aspect-video bg-black rounded-xl overflow-hidden border border-outline-variant flex items-center justify-center group">
        <video
          ref={videoRef}
          muted
          playsInline
          className={`w-full h-full object-cover ${isCameraActive ? 'block' : 'hidden'}`}
        />

        {!isCameraActive && (
          <div className="flex flex-col items-center gap-md text-center p-xl">
            <div className="w-16 h-16 rounded-full bg-surface-container-high flex items-center justify-center text-on-surface-variant">
              <span className="material-symbols-outlined text-[36px]">videocam_off</span>
            </div>
            <p className="font-label-md text-on-surface-variant">
              Camera is currently powered off
            </p>
            <button
              type="button"
              onClick={startCamera}
              className="bg-primary text-on-primary px-lg py-sm rounded-lg font-label-md font-bold hover:shadow-glow hover:scale-[1.02] active:scale-95 transition-all cursor-pointer flex items-center gap-xs"
            >
              <span className="material-symbols-outlined text-[18px]">power_settings_new</span>
              <span>Enable Camera</span>
            </button>
          </div>
        )}
      </div>

      {/* Camera Controls */}
      {isCameraActive && (
        <div className="flex flex-wrap items-center justify-between gap-md pt-sm">
          <button
            type="button"
            onClick={stopCamera}
            className="border border-outline-variant text-on-surface px-md py-sm rounded-lg font-label-md hover:bg-surface-variant transition-all cursor-pointer"
          >
            Turn Off Camera
          </button>

          {!isRecording ? (
            <button
              type="button"
              onClick={startRecording}
              className="bg-error text-on-error px-lg py-sm rounded-lg font-label-md font-bold hover:opacity-90 active:scale-95 transition-all cursor-pointer flex items-center gap-xs"
            >
              <span className="material-symbols-outlined text-[18px]">circle</span>
              <span>Start Capture</span>
            </button>
          ) : (
            <button
              type="button"
              onClick={stopRecording}
              className="bg-primary text-on-primary px-lg py-sm rounded-lg font-label-md font-bold hover:opacity-90 active:scale-95 transition-all cursor-pointer flex items-center gap-xs"
            >
              <span className="material-symbols-outlined text-[18px]">stop</span>
              <span>Stop & Send to AI Pipeline</span>
            </button>
          )}
        </div>
      )}
    </div>
  );
};
