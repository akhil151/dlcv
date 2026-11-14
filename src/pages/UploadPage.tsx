import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { AppLayout } from '../components/layout/AppLayout';
import { FileDropzone } from '../components/upload/FileDropzone';
import { FileMetadataCard } from '../components/upload/FileMetadataCard';
import { RedactionTargetPicker } from '../components/upload/RedactionTargetPicker';
import { BlurConfigPanel } from '../components/upload/BlurConfigPanel';
import { useAppStore } from '../store/useAppStore';
import { api } from '../services/api';
import {
  ACCEPTED_VIDEO_EXTENSIONS,
  SUPPORTED_CATEGORIES,
  type ReadinessSnapshot,
} from '../types';

export const UploadPage: React.FC = () => {
  const navigate = useNavigate();
  const [isBusy, setIsBusy] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [readiness, setReadiness] = useState<ReadinessSnapshot | null>(null);
  const [readinessError, setReadinessError] = useState<string | null>(null);

  const {
    selectedFile,
    videoMetadata,
    uploadProgressPct,
    selectedCategories,
    blurStyle,
    blurIntensity,
    setSelectedFile,
    setVideoMetadata,
    setUploadProgressPct,
    toggleCategory,
    setBlurStyle,
    setBlurIntensity,
    setActiveTaskId,
  } = useAppStore();

  useEffect(() => {
    let cancelled = false;
    api
      .getReadiness()
      .then((snapshot) => {
        if (!cancelled) {
          setReadiness(snapshot);
          setReadinessError(null);
        }
      })
      .catch((err: Error) => {
        if (!cancelled) {
          setReadiness(null);
          setReadinessError(err.message || 'Backend readiness check failed.');
        }
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const supportedSelection = selectedCategories.filter((category) =>
    SUPPORTED_CATEGORIES.includes(category)
  );
  const canStart =
    Boolean(videoMetadata) &&
    supportedSelection.length > 0 &&
    Boolean(readiness?.processingReady) &&
    !isBusy;

  const disabledReason = (() => {
    if (readinessError) return `Backend unavailable: ${readinessError}`;
    if (readiness && !readiness.processingReady) {
      return `Engine not ready: ${readiness.reasons.join('; ') || 'unknown dependency'}`;
    }
    if (!videoMetadata) return 'Upload a video to enable processing.';
    if (supportedSelection.length === 0) return 'Select at least one supported target.';
    return null;
  })();

  const handleFileSelect = async (file: File) => {
    // A new selection invalidates every piece of previous-file state.
    setErrorMessage(null);
    setVideoMetadata(null);
    setUploadProgressPct(0);
    setSelectedFile(file);

    const extension = file.name.slice(file.name.lastIndexOf('.')).toLowerCase();
    if (!ACCEPTED_VIDEO_EXTENSIONS.includes(extension)) {
      setErrorMessage(
        `Unsupported file type "${extension || 'unknown'}". Use ${ACCEPTED_VIDEO_EXTENSIONS.join(', ')}.`
      );
      return;
    }

    setIsBusy(true);
    try {
      const meta = await api.uploadVideo(file, (pct) => setUploadProgressPct(pct));
      setVideoMetadata(meta);
      setUploadProgressPct(100);
    } catch (err: any) {
      setVideoMetadata(null);
      setErrorMessage(err.message || 'Failed to upload video to the backend.');
    } finally {
      setIsBusy(false);
    }
  };

  const handleStartProcess = async () => {
    if (!readiness?.processingReady) {
      setErrorMessage(
        `Engine not ready: ${readiness?.reasons.join('; ') || 'backend readiness unknown'}`
      );
      return;
    }
    if (supportedSelection.length === 0) {
      setErrorMessage('Select at least one supported redaction target (faces).');
      return;
    }

    setErrorMessage(null);
    setIsBusy(true);
    try {
      let videoId = videoMetadata?.videoId;

      if (!videoId && selectedFile) {
        const meta = await api.uploadVideo(selectedFile, (pct) => setUploadProgressPct(pct));
        setVideoMetadata(meta);
        videoId = meta.videoId;
      }
      if (!videoId) throw new Error('Video upload failed or video ID is unavailable.');

      const res = await api.startRedaction({
        videoId,
        // Only validated categories are ever sent to the backend.
        targetCategories: supportedSelection,
        blurStyle,
        blurIntensity,
      });

      setActiveTaskId(res.taskId);
      navigate(`/processing/${res.taskId}`);
    } catch (err: any) {
      setErrorMessage(err.message || 'Failed to initiate the redaction pipeline.');
    } finally {
      setIsBusy(false);
    }
  };

  const readinessTone = readiness?.processingReady
    ? 'text-emerald-400'
    : readinessError || readiness
      ? 'text-error'
      : 'text-on-surface-variant';
  const readinessLabel = readinessError
    ? 'Backend unreachable'
    : readiness
      ? readiness.processingReady
        ? `Engine ready — ${readiness.detector.name} (${readiness.detector.device.toUpperCase()}) + FFmpeg ${
            readiness.ffmpeg.version || ''
          }`
        : `Engine unavailable — ${readiness.reasons.join('; ')}`
      : 'Checking backend readiness…';

  return (
    <AppLayout title="New Ingest Task">
      <div className="max-w-container-max mx-auto w-full flex flex-col gap-lg select-none">
        <div className="flex border-b border-outline-variant gap-md">
          <button
            type="button"
            className="pb-md font-label-md flex items-center gap-xs border-b-2 border-primary text-primary font-bold"
          >
            <span className="material-symbols-outlined text-[20px]">upload_file</span>
            <span>Upload Video File</span>
          </button>
          <button
            type="button"
            disabled
            title="Live camera ingest is not implemented in this build."
            className="pb-md font-label-md flex items-center gap-xs border-b-2 border-transparent text-on-surface-variant opacity-50 cursor-not-allowed"
          >
            <span className="material-symbols-outlined text-[20px]">videocam</span>
            <span>Live Camera Ingest</span>
            <span className="text-[10px] uppercase border border-outline-variant rounded px-xs">
              Not available
            </span>
          </button>
        </div>

        {/* Real engine readiness */}
        <div className="flex items-center gap-sm text-xs">
          <span className={`w-2 h-2 rounded-full bg-current ${readinessTone}`} />
          <span className={readinessTone}>{readinessLabel}</span>
          {readiness && !readiness.processingReady && (
            <span className="text-on-surface-variant">
              (start {readiness.detector.name} / install FFmpeg, then reload)
            </span>
          )}
        </div>

        {errorMessage && (
          <div className="p-md bg-error/10 border border-error/30 rounded-xl text-error flex items-center justify-between text-sm">
            <div className="flex items-center gap-sm">
              <span className="material-symbols-outlined">error</span>
              <span>{errorMessage}</span>
            </div>
            <button
              type="button"
              onClick={() => setErrorMessage(null)}
              className="text-xs underline hover:opacity-80 cursor-pointer"
            >
              Dismiss
            </button>
          </div>
        )}

        <div className="flex flex-col lg:flex-row gap-xl max-w-container-max mx-auto w-full items-start">
          <div className="flex-1 w-full flex flex-col gap-lg">
            <FileDropzone onFileSelect={handleFileSelect} />

            <RedactionTargetPicker
              selectedCategories={selectedCategories}
              onToggleCategory={toggleCategory}
            />

            <BlurConfigPanel
              blurStyle={blurStyle}
              blurIntensity={blurIntensity}
              onStyleChange={setBlurStyle}
              onIntensityChange={setBlurIntensity}
            />
          </div>

          <div className="w-full lg:w-[420px] shrink-0">
            <FileMetadataCard
              metadata={videoMetadata}
              selectedFileName={selectedFile?.name}
              selectedFileSize={selectedFile?.size}
              uploadProgressPct={uploadProgressPct}
              isUploading={isBusy}
              canStart={canStart}
              disabledReason={disabledReason}
              onStartProcess={handleStartProcess}
            />
          </div>
        </div>

        <footer className="mt-auto w-full py-md flex flex-col md:flex-row justify-between items-center border-t border-outline-variant text-label-sm">
          <div className="flex items-center gap-xs text-on-surface-variant opacity-70">
            <span className="material-symbols-outlined text-[16px]">verified_user</span>
            <span>SafeFrame AI — YuNet face redaction • H.264 output</span>
          </div>
          <p className="font-label-sm text-on-surface-variant opacity-60 mt-xs md:mt-0">
            © 2024 SafeFrame AI. Technical Authority in Video Privacy.
          </p>
        </footer>
      </div>
    </AppLayout>
  );
};
