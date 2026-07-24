import React from 'react';
import { useNavigate } from 'react-router-dom';
import type { RecentVideoItem } from '../../types';

export interface RecentVideoTableProps {
  videos?: RecentVideoItem[];
  isLoading?: boolean;
  onVideoClick?: (video: RecentVideoItem) => void;
  onViewAllClick?: () => void;
}

export const RecentVideoTable: React.FC<RecentVideoTableProps> = ({
  videos = [],
  isLoading = false,
  onVideoClick,
  onViewAllClick,
}) => {
  const navigate = useNavigate();

  return (
    <section className="glass-panel rounded-2xl overflow-hidden border border-outline-variant/60 shadow-xl">
      {/* Header */}
      <div className="px-xl py-lg border-b border-outline-variant flex justify-between items-center bg-surface-container/40">
        <div className="flex items-center gap-sm">
          <span className="material-symbols-outlined text-primary text-[20px]">video_library</span>
          <h3 className="font-headline-md text-on-surface font-bold">Recent Processing Jobs</h3>
        </div>
        {videos.length > 0 && (
          <button
            type="button"
            onClick={onViewAllClick}
            className="text-primary font-label-md hover:underline cursor-pointer transition-colors"
          >
            View All
          </button>
        )}
      </div>

      {/* Loading Skeleton */}
      {isLoading && (
        <div className="p-xl flex flex-col gap-md">
          {[1, 2].map((i) => (
            <div key={i} className="h-16 bg-surface-container-high/40 rounded-xl animate-pulse"></div>
          ))}
        </div>
      )}

      {/* Empty State */}
      {!isLoading && videos.length === 0 && (
        <div className="p-2xl flex flex-col items-center justify-center text-center gap-md my-lg">
          <div className="w-16 h-16 rounded-full bg-surface-container-high border border-outline-variant flex items-center justify-center text-on-surface-variant">
            <span className="material-symbols-outlined text-[32px]">movie_edit</span>
          </div>
          <div className="flex flex-col gap-xs max-w-lg">
            <h4 className="font-headline-md text-on-surface font-bold">No Videos Processed Yet</h4>
            <p className="font-body-md text-on-surface-variant text-label-md leading-relaxed">
              Upload your first video or connect a live camera feed to begin automated face, screen, and license plate redaction.
            </p>
          </div>
          <button
            type="button"
            onClick={() => navigate('/upload')}
            className="mt-sm bg-primary text-on-primary px-lg py-sm rounded-lg font-label-md font-bold hover:shadow-glow hover:scale-[1.02] active:scale-95 transition-all flex items-center gap-xs cursor-pointer"
          >
            <span className="material-symbols-outlined text-[18px]">upload_file</span>
            <span>Upload First Video</span>
          </button>
        </div>
      )}

      {/* Video Row Items */}
      {!isLoading && videos.length > 0 && (
        <div className="divide-y divide-outline-variant">
          {videos.map((video) => (
            <div
              key={video.id}
              onClick={() => onVideoClick && onVideoClick(video)}
              className="px-xl py-md flex items-center gap-xl hover:bg-surface-variant/40 transition-all cursor-pointer group"
            >
              <div className="w-32 aspect-video bg-surface-container-high rounded-lg border border-outline-variant overflow-hidden relative shrink-0">
                {video.thumbnailUrl ? (
                  <img
                    src={video.thumbnailUrl}
                    alt={video.filename}
                    className="w-full h-full object-cover"
                  />
                ) : (
                  <div className="w-full h-full flex items-center justify-center bg-surface-container">
                    <span className="material-symbols-outlined text-on-surface-variant">videocam</span>
                  </div>
                )}
                <div className="absolute inset-0 bg-black/40 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity">
                  <span className="material-symbols-outlined text-white">play_arrow</span>
                </div>
              </div>

              <div className="flex-1 min-w-0">
                <h4 className="font-body-lg font-medium text-on-surface truncate group-hover:text-primary transition-colors">
                  {video.filename}
                </h4>
                <p className="font-label-sm text-on-surface-variant">
                  {video.fileSizeMb >= 1000
                    ? `${(video.fileSizeMb / 1024).toFixed(1)} GB`
                    : `${video.fileSizeMb} MB`}{' '}
                  • Redaction: {Array.isArray(video.targets) ? video.targets.join(', ') : video.targets}
                </p>
              </div>

              <div className="hidden md:block text-right shrink-0">
                <p className="font-label-md text-on-surface">{video.createdAt}</p>
                <p className="font-label-sm text-on-surface-variant">
                  {video.duration || '4K MP4'}
                </p>
              </div>

              <div className="bg-primary/10 text-primary border border-primary/20 px-md py-xs rounded-full flex items-center gap-xs shrink-0">
                <span className="material-symbols-outlined text-[14px]">check_circle</span>
                <span className="font-label-sm">{video.status || 'SUCCESS'}</span>
              </div>
            </div>
          ))}
        </div>
      )}
    </section>
  );
};
