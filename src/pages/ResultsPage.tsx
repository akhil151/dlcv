import React, { useEffect } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import { AppLayout } from '../components/layout/AppLayout';
import { SummaryBentoGrid } from '../components/results/SummaryBentoGrid';
import { ExportDownloadHub } from '../components/results/ExportDownloadHub';
import { useAppStore } from '../store/useAppStore';
import { api } from '../services/api';

export const ResultsPage: React.FC = () => {
  const navigate = useNavigate();
  const { taskId = 'demo-task' } = useParams();

  const { activeSummary, setActiveSummary } = useAppStore();

  useEffect(() => {
    if (!taskId) return;

    const fetchSummary = async () => {
      try {
        const summary = await api.getTaskSummary(taskId);
        setActiveSummary(summary);
      } catch (err) {
        console.error('[ResultsPage] Failed to fetch task summary:', err);
      }
    };

    fetchSummary();
  }, [taskId, setActiveSummary]);

  const handleDownloadMaster = () => {
    const downloadUrl = api.getDownloadUrl(taskId);
    window.open(downloadUrl, '_blank');
  };

  const handleDownloadAuditLog = () => {
    const auditUrl = api.getAuditLogUrl(taskId);
    window.open(auditUrl, '_blank');
  };

  const handleDownloadJsonReport = () => {
    if (!activeSummary?.evaluationReport) return;
    const jsonStr = JSON.stringify(activeSummary.evaluationReport, null, 2);
    const blob = new Blob([jsonStr], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `safeframe_evaluation_report_${taskId}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <AppLayout title="Redaction Results">
      <div className="max-w-5xl mx-auto w-full pb-20">
        {/* Header Section */}
        <header className="mb-xl text-center md:text-left flex flex-col md:flex-row md:items-end justify-between gap-md">
          <div>
            <div className="inline-flex items-center gap-sm px-md py-xs rounded-full bg-primary/10 border border-primary/20 text-primary mb-md">
              <span className="material-symbols-outlined text-[16px] filled">check_circle</span>
              <span className="font-label-sm uppercase tracking-widest">Verification Success</span>
            </div>
            <h2 className="font-display text-display mb-sm font-bold text-on-surface">
              Processing Complete
            </h2>
            <p className="font-body-lg text-on-surface-variant max-w-2xl">
              The redaction pipeline has finished. Your video has been analyzed, anonymized, and is
              now ready for secure export.
            </p>
          </div>

          <div className="flex flex-wrap gap-md shrink-0">
            {activeSummary?.evaluationReport && (
              <button
                type="button"
                onClick={handleDownloadJsonReport}
                className="border border-outline-variant text-on-surface px-lg h-14 rounded-xl font-label-md flex items-center gap-sm hover:bg-surface-variant transition-colors cursor-pointer"
              >
                <span className="material-symbols-outlined">analytics</span>
                <span>Evaluation Report (JSON)</span>
              </button>
            )}

            <button
              type="button"
              onClick={handleDownloadMaster}
              className="bg-primary text-on-primary px-xl h-14 rounded-xl font-label-md flex items-center gap-md hover:scale-[1.02] active:scale-95 transition-all shadow-lg shadow-primary/10 cursor-pointer font-bold"
            >
              <span className="material-symbols-outlined">download</span>
              <span>Download Protected Video</span>
            </button>
          </div>
        </header>

        {/* Bento Grid Summary */}
        <SummaryBentoGrid summary={activeSummary || undefined} />

        {/* Secondary Actions & Info */}
        <ExportDownloadHub
          onDownloadAuditLog={handleDownloadAuditLog}
          onReEditClick={() => navigate('/upload')}
        />
      </div>

        {/* Fixed Footer Anchor */}
        <footer className="fixed bottom-0 right-0 left-[240px] bg-surface-container-lowest border-t border-outline-variant py-md px-xl flex justify-between items-center z-40 select-none">
          <div className="flex items-center gap-xs text-on-surface-variant text-label-sm">
            <span className="w-2 h-2 rounded-full bg-emerald-400"></span>
            <span>Task #{taskId} Verification Complete • Ready for Export</span>
          </div>
          <p className="font-label-sm text-on-surface-variant opacity-70">
            © 2024 SafeFrame AI. Technical Authority in Video Privacy.
          </p>
        </footer>
      </div>
    </AppLayout>
  );
};
