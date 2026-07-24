import React from 'react';

export interface StatItem {
  id: string;
  label: string;
  value: string | number;
  subtext: string;
  colorVariant?: 'primary' | 'secondary' | 'tertiary';
}

export interface StatOverviewGridProps {
  stats?: StatItem[];
}

const DEFAULT_STATS: StatItem[] = [
  {
    id: 'videos-processed',
    label: 'Videos Processed',
    value: '12',
    subtext: '+2 today',
    colorVariant: 'primary',
  },
  {
    id: 'objects-detected',
    label: 'Objects Detected',
    value: '1,402',
    subtext: 'Across 180m',
    colorVariant: 'secondary',
  },
  {
    id: 'avg-processing-time',
    label: 'Avg Processing Time',
    value: '45s',
    subtext: '-5% efficiency',
    colorVariant: 'tertiary',
  },
];

export const StatOverviewGrid: React.FC<StatOverviewGridProps> = ({
  stats = DEFAULT_STATS,
}) => {
  const valueColorClasses = {
    primary: 'text-primary',
    secondary: 'text-secondary',
    tertiary: 'text-tertiary',
  };

  const subtextColorClasses = {
    primary: 'text-primary/60',
    secondary: 'text-secondary/60',
    tertiary: 'text-tertiary/60',
  };

  return (
    <section className="grid grid-cols-1 md:grid-cols-3 gap-lg mb-xl">
      {stats.map((stat) => (
        <div
          key={stat.id}
          className="glass-panel stats-card-gradient p-xl rounded-xl border border-outline-variant flex flex-col gap-sm transition-all duration-300 hover:-translate-y-1 hover:border-primary"
        >
          <span className="font-label-md text-on-surface-variant uppercase tracking-widest">
            {stat.label}
          </span>
          <div className="flex items-baseline gap-sm">
            <span
              className={`font-display text-headline-lg ${
                valueColorClasses[stat.colorVariant || 'primary']
              }`}
            >
              {stat.value}
            </span>
            <span
              className={`font-label-sm ${
                subtextColorClasses[stat.colorVariant || 'primary']
              }`}
            >
              {stat.subtext}
            </span>
          </div>
        </div>
      ))}
    </section>
  );
};
