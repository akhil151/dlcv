import React from 'react';
import { ToggleSwitch } from '../ui/ToggleSwitch';
import type { RedactionCategory } from '../../types';

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
    description: 'Detects and redacts facial features across all video frames.',
    icon: 'face',
  },
  {
    id: 'license_plates',
    label: 'License Plates',
    description: 'Anonymizes vehicle license plates and registration numbers.',
    icon: 'directions_car',
  },
  {
    id: 'id_cards',
    label: 'ID Cards & Passports',
    description: 'Blurs visible identity cards, driver licenses, and badges.',
    icon: 'badge',
  },
  {
    id: 'laptop_screens',
    label: 'Laptop Screens',
    description: 'Obscures open computer screens and desktop displays.',
    icon: 'laptop_mac',
  },
  {
    id: 'phone_screens',
    label: 'Phone Screens',
    description: 'Hides mobile phone displays and smartphone UI text.',
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
          Select entity categories to automatically detect and obscure during processing.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-sm">
        {CATEGORY_OPTIONS.map((opt) => {
          const isChecked = selectedCategories.includes(opt.id);
          return (
            <ToggleSwitch
              key={opt.id}
              checked={isChecked}
              onChange={() => onToggleCategory(opt.id)}
              label={opt.label}
              description={opt.description}
              icon={opt.icon}
            />
          );
        })}
      </div>
    </div>
  );
};
