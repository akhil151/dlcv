import React from 'react';
import { ToggleSwitch } from '../ui/ToggleSwitch';
import type { RedactionCategory } from '../../types';
import { SUPPORTED_CATEGORIES } from '../../types';

export interface TargetCategoryOption {
  id: RedactionCategory;
  label: string;
  description: string;
  icon: string;
}

export interface RedactionTargetPickerProps {
  selectedCategories: RedactionCategory[];
  onToggleCategory: (category: RedactionCategory) => void;
}

const CATEGORY_OPTIONS: TargetCategoryOption[] = [
  {
    id: 'faces',
    label: 'Human Faces',
    description: 'Detects faces with YuNet and masks every detected face region.',
    icon: 'face',
  },
  {
    id: 'license_plates',
    label: 'License Plates',
    description: 'No validated license-plate detector is configured yet.',
    icon: 'directions_car',
  },
  {
    id: 'id_cards',
    label: 'ID Cards & Passports',
    description: 'No validated document detector is configured yet.',
    icon: 'badge',
  },
  {
    id: 'laptop_screens',
    label: 'Laptop Screens',
    description: 'No validated screen detector is configured yet.',
    icon: 'laptop_mac',
  },
  {
    id: 'phone_screens',
    label: 'Phone Screens',
    description: 'No validated phone-screen detector is configured yet.',
    icon: 'smartphone',
  },
];

export const RedactionTargetPicker: React.FC<RedactionTargetPickerProps> = ({
  selectedCategories,
  onToggleCategory,
}) => {
  return (
    <div className="flex flex-col gap-md">
      <div className="flex flex-col">
        <h4 className="font-display text-headline-sm font-semibold text-on-surface">
          Redaction Targets
        </h4>
        <p className="font-body text-xs text-on-surface-variant">
          Face redaction is the validated capability in this build. Other categories are
          visible but disabled and are never sent to the backend.
        </p>
      </div>

      <div className="grid grid-cols-1 gap-sm">
        {CATEGORY_OPTIONS.map((opt) => {
          const isSupported = SUPPORTED_CATEGORIES.includes(opt.id);
          const isChecked = isSupported && selectedCategories.includes(opt.id);
          return (
            <div key={opt.id} className={isSupported ? '' : 'opacity-55'}>
              <ToggleSwitch
                checked={isChecked}
                onChange={() => {
                  if (isSupported) onToggleCategory(opt.id);
                }}
                label={opt.label}
                description={opt.description}
                icon={opt.icon}
                disabled={!isSupported}
                badge={isSupported ? undefined : 'Coming Later'}
              />
            </div>
          );
        })}
      </div>
    </div>
  );
};
