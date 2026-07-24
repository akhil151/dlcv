import React from 'react';

export interface ProgressBarProps {
  progress: number; // 0 to 100
  showLabel?: boolean;
  color?: 'primary' | 'secondary' | 'tertiary';
  height?: string;
  className?: string;
}

export const ProgressBar: React.FC<ProgressBarProps> = ({
  progress,
  showLabel = false,
  color = 'primary',
  height = 'h-2',
  className = '',
}) => {
  const clampedProgress = Math.min(100, Math.max(0, progress));

  const colorClasses = {
    primary: 'bg-primary-container',
    secondary: 'bg-secondary-container',
    tertiary: 'bg-tertiary-container',
  };

  return (
    <div className={`w-full flex flex-col gap-xs ${className}`}>
      {showLabel && (
        <div className="flex justify-between items-center text-xs font-label text-on-surface-variant">
          <span>Processing Progress</span>
          <span className="font-bold text-primary">{Math.round(clampedProgress)}%</span>
        </div>
      )}
      <div className={`w-full bg-surface-container-high rounded-full overflow-hidden relative ${height}`}>
        <div
          className={`h-full ${colorClasses[color]} transition-all duration-300 rounded-full relative overflow-hidden`}
          style={{ width: `${clampedProgress}%` }}
        >
          <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/20 to-transparent animate-[shimmer_2s_infinite]" />
        </div>
      </div>
    </div>
  );
};
