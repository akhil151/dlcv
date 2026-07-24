import React, { useRef, useState } from 'react';

export interface FileDropzoneProps {
  onFileSelect: (file: File) => void;
  acceptedFormats?: string[];
}

export const FileDropzone: React.FC<FileDropzoneProps> = ({
  onFileSelect,
  acceptedFormats = ['MP4', 'AVI', 'MOV'],
}) => {
  const [isDragOver, setIsDragOver] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(true);
  };

  const handleDragLeave = () => {
    setIsDragOver(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      onFileSelect(e.dataTransfer.files[0]);
    }
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      onFileSelect(e.target.files[0]);
    }
  };

  return (
    <div className="flex-[2] flex flex-col gap-lg w-full">
      {/* Drag and Drop Canvas */}
      <div
        id="drop-zone"
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={() => fileInputRef.current?.click()}
        className={`drag-area-dashed rounded-xl flex flex-col items-center justify-center p-xl group cursor-pointer transition-all ${
          isDragOver ? 'bg-surface-container-high border-primary' : 'hover:bg-surface-container-low'
        } min-h-[380px]`}
      >
        <div className="w-20 h-20 rounded-full bg-surface-container flex items-center justify-center mb-lg group-hover:scale-110 transition-transform">
          <span className="material-symbols-outlined text-primary text-[48px]">upload_file</span>
        </div>

        <h3 className="font-headline-md text-headline-md text-on-surface text-center mb-sm">
          Drag and drop your video file here
        </h3>

        <p className="font-body-md text-body-md text-on-surface-variant text-center opacity-70">
          Supported formats:{' '}
          {acceptedFormats.map((fmt, idx) => (
            <React.Fragment key={fmt}>
              <span className="text-primary-fixed font-semibold">{fmt}</span>
              {idx < acceptedFormats.length - 1 ? ', ' : ''}
            </React.Fragment>
          ))}
        </p>

        <input
          ref={fileInputRef}
          id="file-input"
          type="file"
          accept=".mp4,.avi,.mov"
          onChange={handleInputChange}
          className="hidden"
        />

        <button
          type="button"
          onClick={(e) => {
            e.stopPropagation();
            fileInputRef.current?.click();
          }}
          className="mt-xl px-xl py-md border border-outline-variant rounded-lg font-label-md text-on-surface hover:bg-surface-variant transition-colors cursor-pointer"
        >
          Browse Files
        </button>
      </div>

      {/* Local Security Guarantee */}
      <div className="flex items-center gap-md p-md bg-surface-container-lowest rounded-lg border border-outline-variant/30">
        <span className="material-symbols-outlined text-primary">security</span>
        <p className="font-label-sm text-label-sm text-on-surface-variant">
          All files are processed locally or within encrypted secure instances. Your raw data is
          never stored without your explicit consent.
        </p>
      </div>
    </div>
  );
};
