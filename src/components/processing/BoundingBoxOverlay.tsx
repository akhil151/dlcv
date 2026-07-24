import React from 'react';
import type { DetectedEntity } from '../../types';

export interface BoundingBoxOverlayProps {
  entities: DetectedEntity[];
}

export const BoundingBoxOverlay: React.FC<BoundingBoxOverlayProps> = ({ entities }) => {
  return (
    <div className="absolute inset-0 pointer-events-none overflow-hidden select-none">
      {entities.map((entity) => {
        const { x, y, width, height } = entity.bbox;
        const isPlate = entity.category === 'license_plates';

        return (
          <div
            key={entity.id}
            style={{
              left: `${x}%`,
              top: `${y}%`,
              width: `${width}%`,
              height: `${height}%`,
            }}
            className={`absolute border-2 active-redaction flex items-center justify-center transition-all duration-300 ${
              isPlate ? 'border-tertiary-container bg-tertiary-container/10' : 'border-primary bg-primary/10'
            }`}
          >
            {/* Backdrop Blur Mask */}
            <div className="w-full h-full backdrop-blur-md opacity-90" />

            {/* Label Chip */}
            <span
              className={`absolute top-0 left-0 -translate-y-full text-[10px] px-xs py-[2px] font-bold uppercase tracking-wider ${
                isPlate ? 'bg-tertiary-container text-white' : 'bg-primary text-on-primary'
              }`}
            >
              {entity.label} [{Math.round(entity.confidence * 100)}%]
            </span>
          </div>
        );
      })}
    </div>
  );
};
