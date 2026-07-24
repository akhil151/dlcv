import React from 'react';

export interface ResultVideoPlayerProps {
  title?: string;
  thumbnailUrl?: string;
  videoUrl?: string;
  onPlayClick?: () => void;
}

export const ResultVideoPlayer: React.FC<ResultVideoPlayerProps> = ({
  title = 'surveillance_0912.mp4',
  thumbnailUrl = 'https://lh3.googleusercontent.com/aida-public/AB6AXuBRea9vFpYYyxFxkaBMI31utLtboA-wALEEgQH5_DXFu-pdzzxkPD--JoK1Uca7nMqpLxOku-A2D0sVPSJRo8kAhug3wp5kdH1LqkOpLFuVXC6a68pGaiZLvIOd2M8n4mOxvWaxWnTTSii0TpOujMM7vm5i4vLxwPJ2wkkD8uDnWlLhjUdL5YLeVlqgLCGwn4YANFpEaIqV1NFVKGj8Ge9S_mfLaHjOKvpeUYYY-r0RG3kvhxAfrqMylBJEUR8vvg5LPStyQXnc3r4',
  videoUrl,
  onPlayClick,
}) => {
  const handlePlay = () => {
    if (onPlayClick) {
      onPlayClick();
    } else if (videoUrl) {
      window.open(videoUrl, '_blank');
    }
  };

  return (
    <div className="md:col-span-4 lg:col-span-2 glass-card rounded-xl overflow-hidden relative group h-[320px]">
      <div className="absolute inset-0 bg-gradient-to-t from-surface-container-lowest to-transparent z-10"></div>
      <div className="absolute inset-0 z-0 overflow-hidden">
        <img
          src={thumbnailUrl}
          alt={title}
          className="w-full h-full object-cover opacity-60 group-hover:scale-105 transition-transform duration-700"
        />
      </div>
      <div className="absolute bottom-0 left-0 p-lg z-20 w-full flex justify-between items-end">
        <div>
          <span className="font-label-sm text-primary-fixed uppercase mb-xs block">Preview</span>
          <h3 className="font-headline-md text-on-surface">{title}</h3>
        </div>
        <button
          type="button"
          onClick={handlePlay}
          className="w-12 h-12 rounded-full bg-white/10 backdrop-blur-md flex items-center justify-center hover:bg-white/20 transition-colors cursor-pointer"
        >
          <span className="material-symbols-outlined text-white">play_arrow</span>
        </button>
      </div>
    </div>
  );
};
