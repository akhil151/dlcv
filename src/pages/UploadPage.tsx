import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { AppLayout } from '../components/layout/AppLayout';
import { FileDropzone } from '../components/upload/FileDropzone';
import { LiveCameraCard } from '../components/upload/LiveCameraCard';
import { FileMetadataCard } from '../components/upload/FileMetadataCard';
import { useAppStore } from '../store/useAppStore';
import { api } from '../services/api';

export const UploadPage: React.FC = () => {
  const navigate = useNavigate();
  const [activeTab, setActiveTab] = useState<'file' | 'camera'>('file');
  const [isUploading, setIsUploading] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const {
    selectedFile,
    videoMetadata,
    selectedCategories,
    setSelectedFile,
    setVideoMetadata,
    setUploadProgressPct,
    toggleCategory,
    setActiveTaskId,
  } = useAppStore();

  const autoRedactFaces = selectedCategories.includes('faces');
  const redactLicensePlates = selectedCategories.includes('license_plates');

  const handleToggleFaces = () => {
    toggleCategory('faces');
  };

  const handleTogglePlates = () => {
    toggleCategory('license_plates');
  };

  const handleFileSelect = async (file: File) => {
    setSelectedFile(file);
    setErrorMessage(null);
    setIsUploading(true);
    setUploadProgressPct(0);

    try {
      const meta = await api.uploadVideo(file, (pct) => {
        setUploadProgressPct(pct);
      });
      setVideoMetadata(meta);
    } catch (err: any) {
      console.error('[UploadPage] Upload failed:', err);
      setErrorMessage(err.message || 'Failed to upload video to server. Please try again.');
    } finally {
      setIsUploading(false);
    }
  };

  const handleStartProcess = async () => {
    if (!videoMetadata && !selectedFile) {
      setErrorMessage('Please select a video file or record a live camera feed before starting processing.');
      return;
    }

    setErrorMessage(null);
    setIsUploading(true);

    try {
      let videoId = videoMetadata?.videoId;

      if (!videoId && selectedFile) {
        const meta = await api.uploadVideo(selectedFile);
        setVideoMetadata(meta);
        videoId = meta.videoId;
      }

      if (!videoId) {
        throw new Error('Video upload failed or video ID unavailable.');
      }

      const res = await api.startRedaction({
        videoId: videoId,
        targetCategories: selectedCategories,
        blurStyle: 'gaussian',
        blurIntensity: 50,
      });

      setActiveTaskId(res.taskId);
      navigate(`/processing/${res.taskId}`);
    } catch (err: any) {
      console.error('[UploadPage] Failed to start redaction task:', err);
      setErrorMessage(err.message || 'Failed to initiate redaction pipeline.');
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <AppLayout title="New Ingest Task">
      <div className="max-w-container-max mx-auto w-full flex flex-col gap-lg select-none">
        {/* Ingest Mode Mode Selector Tabs */}
        <div className="flex border-b border-outline-variant gap-md">
          <button
            type="button"
            onClick={() => setActiveTab('file')}
            className={`pb-md font-label-md flex items-center gap-xs cursor-pointer border-b-2 transition-all ${
              activeTab === 'file'
                ? 'border-primary text-primary font-bold'
                : 'border-transparent text-on-surface-variant hover:text-on-surface'
            }`}
          >
            <span className="material-symbols-outlined text-[20px]">upload_file</span>
            <span>Upload Video File</span>
          </button>
          <button
            type="button"
            onClick={() => setActiveTab('camera')}
            className={`pb-md font-label-md flex items-center gap-xs cursor-pointer border-b-2 transition-all ${
              activeTab === 'camera'
                ? 'border-primary text-primary font-bold'
                : 'border-transparent text-on-surface-variant hover:text-on-surface'
            }`}
          >
            <span className="material-symbols-outlined text-[20px]">videocam</span>
            <span>Live Camera Ingest</span>
          </button>
        </div>

        {/* Error Banner if any */}
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

        {/* 2-Column Stage */}
        <div className="flex flex-col lg:flex-row gap-xl max-w-container-max mx-auto w-full items-start">
          {/* Left Column: File Dropzone OR Live Camera */}
          <div className="flex-1 w-full">
            {activeTab === 'file' ? (
              <FileDropzone onFileSelect={handleFileSelect} />
            ) : (
              <LiveCameraCard onRecordedFileReady={handleFileSelect} />
            )}
          </div>

          {/* Right Column: File Information & Config Panel */}
          <div className="w-full lg:w-[420px] shrink-0">
            <FileMetadataCard
              metadata={videoMetadata}
              selectedFileName={selectedFile?.name}
              selectedFileSize={selectedFile?.size}
              isUploading={isUploading}
              autoRedactFaces={autoRedactFaces}
              redactLicensePlates={redactLicensePlates}
              onToggleFaces={handleToggleFaces}
              onTogglePlates={handleTogglePlates}
              onStartProcess={handleStartProcess}
            />
          </div>
        </div>

        {/* Technical Footer */}
        <footer className="mt-auto w-full py-md flex flex-col md:flex-row justify-between items-center border-t border-outline-variant text-label-sm">
          <div className="flex items-center gap-xs text-on-surface-variant opacity-70">
            <span className="material-symbols-outlined text-[16px]">verified_user</span>
            <span>SafeFrame AI Vision Engine • H.264 / WebM / MP4 supported</span>
          </div>
          <p className="font-label-sm text-on-surface-variant opacity-60 mt-xs md:mt-0">
            © 2024 SafeFrame AI. Technical Authority in Video Privacy.
          </p>
        </footer>
      </div>
    </AppLayout>
  );
};
