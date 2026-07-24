import React from 'react';

export interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'primary' | 'secondary' | 'ghost' | 'danger' | 'outline';
  size?: 'sm' | 'md' | 'lg';
  isLoading?: boolean;
  leftIcon?: React.ReactNode;
  rightIcon?: React.ReactNode;
}

export const Button: React.FC<ButtonProps> = ({
  children,
  variant = 'primary',
  size = 'md',
  isLoading = false,
  leftIcon,
  rightIcon,
  className = '',
  disabled,
  ...props
}) => {
  const baseClasses =
    'inline-flex items-center justify-center font-label font-bold rounded-lg transition-all duration-200 focus:outline-none disabled:opacity-50 disabled:cursor-not-allowed active:scale-95';

  const sizeClasses = {
    sm: 'px-md py-xs text-xs gap-xs h-8',
    md: 'px-lg py-sm text-sm gap-sm h-10',
    lg: 'px-xl py-md text-base gap-md h-12',
  };

  const variantClasses = {
    primary:
      'bg-primary-container text-on-primary-container hover:opacity-90 shadow-sm hover:shadow-glow',
    secondary:
      'bg-secondary-container text-on-secondary-container hover:opacity-90',
    ghost:
      'bg-transparent text-on-surface-variant hover:text-primary hover:bg-surface-variant/40',
    danger:
      'bg-error/10 text-error border border-error/20 hover:bg-error/20',
    outline:
      'border border-outline-variant text-on-surface hover:bg-surface-variant',
  };

  return (
    <button
      className={`${baseClasses} ${sizeClasses[size]} ${variantClasses[variant]} ${className}`}
      disabled={disabled || isLoading}
      {...props}
    >
      {isLoading ? (
        <span className="material-symbols-outlined animate-spin text-[18px]">
          progress_activity
        </span>
      ) : (
        leftIcon
      )}
      <span>{children}</span>
      {!isLoading && rightIcon}
    </button>
  );
};
