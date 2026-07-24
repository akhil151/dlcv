import React from 'react';
import { Link } from 'react-router-dom';

export interface LandingHeaderProps {
  onGetStartedClick?: () => void;
}

export const LandingHeader: React.FC<LandingHeaderProps> = ({ onGetStartedClick }) => {
  return (
    <header className="fixed top-0 left-0 w-full z-50 flex justify-between items-center px-gutter h-16 bg-surface/80 backdrop-blur-md border-b border-outline-variant select-none">
      <div className="flex items-center gap-md">
        <Link to="/" className="font-display text-headline-md font-bold text-on-surface">
          SafeFrame
        </Link>
        <nav className="hidden md:flex items-center gap-lg ml-xl">
          <Link
            to="/dashboard"
            className="text-on-surface-variant hover:text-primary transition-colors font-label text-sm"
          >
            Dashboard
          </Link>
          <Link
            to="/upload"
            className="text-on-surface-variant hover:text-primary transition-colors font-label text-sm"
          >
            Upload
          </Link>
        </nav>
      </div>

      <div className="flex items-center gap-md">
        <a
          href="https://github.com"
          target="_blank"
          rel="noreferrer"
          className="material-symbols-outlined text-on-surface-variant hover:text-primary transition-colors"
          title="Source Code"
        >
          code
        </a>
        <Link
          to="/upload"
          onClick={onGetStartedClick}
          className="bg-primary-container text-on-primary-container px-lg py-sm rounded-lg font-label text-xs font-bold hover:opacity-90 active:scale-95 transition-all shadow-sm"
        >
          Get Started
        </Link>
      </div>
    </header>
  );
};
