import React from 'react';

export interface GlassCardProps extends React.HTMLAttributes<HTMLDivElement> {
  children: React.ReactNode;
  glow?: boolean;
  hoverEffect?: boolean;
  className?: string;
}

export const GlassCard: React.FC<GlassCardProps> = ({
  children,
  glow = false,
  hoverEffect = false,
  className = '',
  ...props
}) => {
  return (
    <div
      className={`glass-card rounded-xl overflow-hidden transition-all duration-300 ${
        glow ? 'border-primary/40 shadow-glow' : 'border-outline-variant/50'
      } ${
        hoverEffect ? 'hover:border-primary/60 hover:bg-surface-container-high/40 hover:-translate-y-[2px]' : ''
      } ${className}`}
      {...props}
    >
      {children}
    </div>
  );
};
