import React from 'react';
import type { BlurStyle } from '../../types';

export interface BlurConfigPanelProps {
  blurStyle: BlurStyle;
  blurIntensity: number;
  onStyleChange: (style: BlurStyle) => void;
  onIntensityChange: (intensity: number) => void;
}

export const BlurConfigPanel: React.FC<BlurConfigPanelProps> = ({
  blurStyle,
  blurIntensity,
  onStyleChange,
  onIntensityChange,
}) => {
  const styles: { id: BlurStyle; label: string; icon: string }[] = [
    { id: 'gaussian', label: 'Gaussian Blur', icon: 'blur_on' },
    { id: 'pixelate', label: 'Pixelate', icon: 'grid_on' },
    { id: 'blackout', label: 'Solid Blackout', icon: 'contrast' },
  ];

  return (
    <div className="flex flex-col gap-md bg-surface-container-low p-lg rounded-xl border border-outline-variant/50">
      <h4 className="font-display text-sm font-semibold text-primary uppercase tracking-wider">
        Redaction Style Settings
      </h4>

      {/* Blur Style Buttons */}
      <div className="flex flex-col gap-xs">
        <label className="font-label text-xs text-on-surface-variant font-medium">
          Anonymization Mode
        </label>
        <div className="grid grid-cols-3 gap-sm">
          {styles.map((s) => {
            const isSelected = blurStyle === s.id;
            return (
              <button
                key={s.id}
                type="button"
                onClick={() => onStyleChange(s.id)}
                className={`flex flex-col items-center justify-center p-sm rounded-lg border text-center transition-all cursor-pointer ${
                  isSelected
                    ? 'bg-primary-container/20 border-primary text-primary font-bold shadow-sm'
                    : 'bg-surface-container border-outline-variant/40 text-on-surface-variant hover:bg-surface-container-high'
                }`}
              >
                <span className="material-symbols-outlined text-[22px] mb-xs">
                  {s.icon}
                </span>
                <span className="font-label text-xs">{s.label}</span>
              </button>
            );
          })}
        </div>
      </div>

      {/* Intensity Slider */}
      <div className="flex flex-col gap-xs pt-xs">
        <div className="flex justify-between items-center text-xs font-label">
          <span className="text-on-surface-variant font-medium">Mask Intensity</span>
          <span className="text-primary font-bold">{blurIntensity}%</span>
        </div>
        <input
          type="range"
          min="10"
          max="100"
          value={blurIntensity}
          onChange={(e) => onIntensityChange(Number(e.target.value))}
          className="w-full accent-primary bg-surface-container-high h-2 rounded-lg cursor-pointer"
        />
      </div>
    </div>
  );
};
