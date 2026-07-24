import React from 'react';
import { SideNavBar } from './SideNavBar';
import { TopAppBar } from './TopAppBar';

export interface AppLayoutProps {
  children: React.ReactNode;
  title?: string;
  activeTaskName?: string;
  isProcessing?: boolean;
  onCancelProcess?: () => void;
  hideTopNav?: boolean;
  noPadding?: boolean;
}

export const AppLayout: React.FC<AppLayoutProps> = ({
  children,
  title,
  activeTaskName,
  isProcessing,
  onCancelProcess,
  hideTopNav = false,
  noPadding = false,
}) => {
  return (
    <div className="flex min-h-screen bg-surface selection:bg-primary/30 selection:text-primary">
      {/* Fixed 240px Sidebar */}
      <SideNavBar />

      {/* Main Content Area */}
      <main className="ml-[240px] flex-1 flex flex-col min-h-screen overflow-y-auto relative">
        {!hideTopNav && (
          <TopAppBar
            title={title}
            activeTaskName={activeTaskName}
            isProcessing={isProcessing}
            onCancelProcess={onCancelProcess}
          />
        )}
        <div className={`flex-1 ${noPadding ? '' : 'p-xl'}`}>{children}</div>
      </main>
    </div>
  );
};
