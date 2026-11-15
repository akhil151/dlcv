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
  /** False when the backend reports the plate detector weights as missing. */
  licensePlatesAvailable?: boolean;
  /** False when the backend reports the identity-document weights as missing. */
  documentsAvailable?: boolean;
  /** False when the backend reports the laptop-screen detector as missing. */
  laptopScreensAvailable?: boolean;
  /** False when the backend reports the phone-screen detector as missing. */
  phoneScreensAvailable?: boolean;
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
    description:
      'Detects vehicle license plates with an RT-DETR model and masks each ' +
      'plate with a rectangular cover in the selected redaction style.',
    icon: 'directions_car',
  },
  {
    id: 'id_cards',
    label: 'ID Cards',
    description:
      'Detects identity documents with a DocAligner corner model and masks the ' +
      'whole card, including the portrait, text and any machine-readable zone.',
    icon: 'badge',
  },
  {
    id: 'passports',
    label: 'Passports',
    description:
      'Uses the same validated document detector; the full passport data page ' +
      'and MRZ are masked. The model locates documents but does not classify ' +
      'card vs passport, so either document option masks every detected document.',
    icon: 'menu_book',
  },
  {
    id: 'laptop_screens',
    label: 'Laptop Screens',
    description:
      'Detects laptops with an RT-DETR model and masks the visible display. ' +
      'The display boundary is refined from image edges; when it cannot be ' +
      'separated reliably the whole laptop is masked and the summary says so.',
    icon: 'laptop_mac',
  },
  {
    id: 'phone_screens',
    label: 'Phone Screens',
    description:
      'Detects mobile phones with an RT-DETR model and masks the visible ' +
      'display in portrait or landscape. The display boundary is refined from ' +
      'image edges; when it cannot be separated reliably the whole phone is ' +
      'masked and the summary says so.',
    icon: 'smartphone',
  },
];

export const RedactionTargetPicker: React.FC<RedactionTargetPickerProps> = ({
  selectedCategories,
  onToggleCategory,
  licensePlatesAvailable = true,
  documentsAvailable = true,
  laptopScreensAvailable = true,
  phoneScreensAvailable = true,
}) => {
  return (
    <div className="flex flex-col gap-md">
      <div className="flex flex-col">
        <h4 className="font-display text-headline-sm font-semibold text-on-surface">
          Redaction Targets
        </h4>
        <p className="font-body text-xs text-on-surface-variant">
          Face, license-plate, identity-document, laptop-screen and phone-screen redaction
          are validated in this build. Every target is independent and is sent to the
          backend only when its detector is ready.
        </p>
      </div>

      <div className="grid grid-cols-1 gap-sm">
        {CATEGORY_OPTIONS.map((opt) => {
          const isSupported = SUPPORTED_CATEGORIES.includes(opt.id);
          const isChecked = isSupported && selectedCategories.includes(opt.id);
          const isDocument = opt.id === 'id_cards' || opt.id === 'passports';
          const plateWeightsMissing = opt.id === 'license_plates' && !licensePlatesAvailable;
          const documentWeightsMissing = isDocument && !documentsAvailable;
          const laptopWeightsMissing =
            opt.id === 'laptop_screens' && !laptopScreensAvailable;
          const phoneWeightsMissing =
            opt.id === 'phone_screens' && !phoneScreensAvailable;
          const weightsMissing =
            plateWeightsMissing || documentWeightsMissing || laptopWeightsMissing ||
            phoneWeightsMissing;
          return (
            <div key={opt.id} className={isSupported ? '' : 'opacity-55'}>
              <ToggleSwitch
                checked={isChecked}
                onChange={() => {
                  if (isSupported) onToggleCategory(opt.id);
                }}
                label={opt.label}
                description={
                  plateWeightsMissing
                    ? 'Plate detector weights are not installed on the server ' +
                      '(run scripts/download_plate_model.py).'
                    : documentWeightsMissing
                      ? 'Identity-document detector weights are not installed on the ' +
                        'server (run scripts/download_document_model.py).'
                      : laptopWeightsMissing
                        ? 'Laptop-screen detector weights are not installed on the ' +
                          'server (run scripts/download_laptop_screen_model.py).'
                        : phoneWeightsMissing
                          ? 'Phone-screen detector weights are not installed on the ' +
                            'server (run scripts/download_phone_screen_model.py).'
                          : opt.description
                }
                icon={opt.icon}
                disabled={!isSupported || weightsMissing}
                badge={
                  !isSupported
                    ? 'Coming Later'
                    : weightsMissing
                      ? 'Weights Missing'
                      : undefined
                }
              />
            </div>
          );
        })}
      </div>
    </div>
  );
};
