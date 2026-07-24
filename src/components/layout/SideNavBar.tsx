import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { useAppStore } from '../../store/useAppStore';

export interface NavItem {
  label: string;
  path: string;
  icon: string;
}

export interface SideNavBarProps {
  onNewUploadClick?: () => void;
}

export const SideNavBar: React.FC<SideNavBarProps> = ({ onNewUploadClick }) => {
  const location = useLocation();
  const activeTaskId = useAppStore((state) => state.activeTaskId) || 'demo-task';

  const navItems: NavItem[] = [
    { label: 'Dashboard', path: '/dashboard', icon: 'dashboard' },
    { label: 'Upload', path: '/upload', icon: 'upload_file' },
    { label: 'Processing', path: `/processing/${activeTaskId}`, icon: 'memory' },
    { label: 'Results', path: `/results/${activeTaskId}`, icon: 'check_circle' },
  ];

  return (
    <aside
      className="fixed left-0 top-0 h-full w-[240px] bg-surface-container border-r border-outline-variant flex flex-col p-md gap-sm z-50 select-none"
      aria-label="Sidebar Navigation"
    >
      {/* Brand Header */}
      <div className="mb-lg px-sm">
        <Link to="/dashboard" className="block group">
          <h1 className="font-display text-headline-md font-bold text-primary transition-opacity group-hover:opacity-90 flex items-center gap-xs">
            <span className="material-symbols-outlined text-[24px]">verified_user</span>
            <span>SafeFrame AI</span>
          </h1>
          <p className="font-label-sm text-on-surface-variant opacity-70 mt-[2px]">
            Neural Vision Pipeline
          </p>
        </Link>
      </div>

      {/* Main Navigation Links */}
      <nav className="flex flex-col gap-xs flex-1">
        {navItems.map((item) => {
          const isActive =
            location.pathname === item.path ||
            (item.path.startsWith('/processing') && location.pathname.startsWith('/processing')) ||
            (item.path.startsWith('/results') && location.pathname.startsWith('/results'));
          return (
            <Link
              key={item.label}
              to={item.path}
              className={`rounded-lg px-md py-sm flex items-center gap-md transition-all duration-200 ${
                isActive
                  ? 'bg-secondary-container text-on-secondary-container font-semibold translate-x-1 shadow-active-glow'
                  : 'text-on-surface-variant hover:bg-surface-variant/60 hover:text-on-surface'
              }`}
            >
              <span
                className="material-symbols-outlined text-[20px]"
                style={{ fontVariationSettings: isActive ? "'FILL' 1" : "'FILL' 0" }}
              >
                {item.icon}
              </span>
              <span className="font-label-md text-label-md">{item.label}</span>
            </Link>
          );
        })}
      </nav>

      {/* Primary CTA & Back to Landing Action */}
      <div className="mt-auto flex flex-col gap-sm pt-md border-t border-outline-variant">
        <Link
          to="/upload"
          onClick={onNewUploadClick}
          className="bg-primary text-on-primary rounded-lg py-sm px-md font-label-md font-bold flex items-center justify-center gap-sm hover:opacity-90 hover:scale-[1.02] active:scale-95 transition-all shadow-glow"
        >
          <span className="material-symbols-outlined text-[18px]">add</span>
          <span>New Upload</span>
        </Link>

        <Link
          to="/"
          className="group text-on-surface-variant hover:text-primary transition-colors px-md py-xs flex items-center gap-md rounded-lg hover:bg-surface-variant/40 text-left w-full cursor-pointer"
        >
          <span className="material-symbols-outlined text-[20px] group-hover:-translate-x-1 transition-transform">
            arrow_back
          </span>
          <span className="font-label-sm text-label-sm">Back to Landing</span>
        </Link>
      </div>
    </aside>
  );
};
