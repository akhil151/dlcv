import React from 'react';

export interface ToggleSwitchProps {
  checked: boolean;
  onChange: (checked: boolean) => void;
  label?: string;
  description?: string;
  disabled?: boolean;
  icon?: string;
}

export const ToggleSwitch: React.FC<ToggleSwitchProps> = ({
  checked,
  onChange,
  label,
  description,
  disabled = false,
  icon,
}) => {
  return (
    <label
      className={`flex items-center justify-between p-md rounded-xl border transition-all cursor-pointer select-none ${
        checked
          ? 'bg-primary/10 border-primary/40 text-on-surface'
          : 'bg-surface-container border-outline-variant/50 text-on-surface-variant hover:bg-surface-container-high'
      } ${disabled ? 'opacity-50 cursor-not-allowed' : ''}`}
    >
      <div className="flex items-center gap-md">
        {icon && (
          <span
            className={`material-symbols-outlined text-[22px] ${
              checked ? 'text-primary' : 'text-on-surface-variant'
            }`}
          >
            {icon}
          </span>
        )}
        <div className="flex flex-col">
          {label && <span className="font-label text-sm font-semibold text-on-surface">{label}</span>}
          {description && (
            <span className="font-body text-xs text-on-surface-variant">{description}</span>
          )}
        </div>
      </div>
      <button
        type="button"
        role="switch"
        aria-checked={checked}
        disabled={disabled}
        onClick={() => !disabled && onChange(!checked)}
        className={`relative inline-flex h-6 w-11 shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors duration-200 ease-in-out focus:outline-none ${
          checked ? 'bg-primary-container' : 'bg-surface-container-high'
        }`}
      >
        <span
          className={`pointer-events-none inline-block h-5 w-5 transform rounded-full bg-white shadow-lg ring-0 transition duration-200 ease-in-out ${
            checked ? 'translate-x-5' : 'translate-x-0'
          }`}
        />
      </button>
    </label>
  );
};
