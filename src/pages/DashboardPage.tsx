import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { AppLayout } from '../components/layout/AppLayout';
import { WelcomeBanner } from '../components/dashboard/WelcomeBanner';
import { StatOverviewGrid, type StatItem } from '../components/dashboard/StatOverviewGrid';
import { RecentVideoTable } from '../components/dashboard/RecentVideoTable';
import { api } from '../services/api';
import type { ReadinessSnapshot, RecentVideoItem } from '../types';

export const DashboardPage: React.FC = () => {
  const navigate = useNavigate();
  const [stats, setStats] = useState<StatItem[]>([
    {
      id: 'videos-processed',
      label: 'Verified Tasks',
      value: 0,
      subtext: 'Completed tasks',
      colorVariant: 'primary',
    },
    {
      id: 'objects-detected',
      label: 'Face Regions Masked',
      value: 0,
      subtext: 'Region-frames across completed tasks',
      colorVariant: 'secondary',
    },
    {
      id: 'avg-processing-time',
      label: 'Avg Processing Time',
      value: '—',
      subtext: 'Completed tasks only',
      colorVariant: 'tertiary',
    },
  ]);
  const [recentVideos, setRecentVideos] = useState<RecentVideoItem[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [isShowingAll, setIsShowingAll] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [readiness, setReadiness] = useState<ReadinessSnapshot | null>(null);

  useEffect(() => {
    let cancelled = false;

    const fetchDashboard = async () => {
      setIsLoading(true);
      try {
        const data = await api.getDashboardData();
        if (cancelled) return;
        setErrorMessage(null);
        if (data && data.stats) {
          setStats([
            {
              id: 'videos-processed',
              label: 'Verified Tasks',
              value: data.stats.videosProcessed ?? 0,
              subtext: 'Completed tasks',
              colorVariant: 'primary',
            },
            {
              id: 'objects-detected',
              label: 'Face Regions Masked',
              value: Number(data.stats.objectsDetected ?? 0).toLocaleString(),
              subtext: 'Region-frames across completed tasks',
              colorVariant: 'secondary',
            },
            {
              id: 'avg-processing-time',
              label: 'Avg Processing Time',
              value:
                data.stats.avgProcessingTimeSeconds > 0
                  ? `${data.stats.avgProcessingTimeSeconds}s`
                  : '—',
              subtext: 'Completed tasks only',
              colorVariant: 'tertiary',
            },
          ]);
        }
        if (data && data.recentVideos) {
          setRecentVideos(data.recentVideos);
        }
      } catch (err: any) {
        if (cancelled) return;
        console.error('[DashboardPage] Failed to fetch dashboard data:', err);
        setErrorMessage(err?.message || 'Unable to load dashboard data from the server.');
      } finally {
        if (!cancelled) setIsLoading(false);
      }
    };

    fetchDashboard();
    api
      .getReadiness()
      .then((snapshot) => {
        if (!cancelled) setReadiness(snapshot);
      })
      .catch(() => {
        if (!cancelled) setReadiness(null);
      });

    return () => {
      cancelled = true;
    };
  }, []);

  const engineLabel = readiness
    ? `${readiness.detector.name} ${readiness.detector.device.toUpperCase()} — ${
        readiness.ffmpeg.available ? 'FFmpeg ready' : 'FFmpeg unavailable'
      }`
    : 'Engine status unknown';

  return (
    <AppLayout title="Dashboard">
      <div className="max-w-container-max mx-auto w-full flex flex-col gap-xl">
        <WelcomeBanner
          tasksCompletedCount={Number(stats[0]?.value) || 0}
          engineLabel={engineLabel}
          engineReady={Boolean(readiness?.processingReady)}
          onUploadClick={() => navigate('/upload')}
        />

        {errorMessage && (
          <div className="p-md bg-error/10 border border-error/30 rounded-xl text-error flex items-center gap-sm text-sm">
            <span className="material-symbols-outlined text-[18px]">error</span>
            <span>Dashboard data unavailable: {errorMessage}</span>
          </div>
        )}

        <StatOverviewGrid stats={stats} />

        <RecentVideoTable
          videos={isShowingAll ? recentVideos : recentVideos.slice(0, 5)}
          isLoading={isLoading}
          onVideoClick={(video) => navigate(`/results/${video.id}`)}
          onViewAllClick={() => setIsShowingAll(!isShowingAll)}
        />

        <footer className="w-full py-lg px-xl flex flex-col md:flex-row justify-between items-center max-w-container-max mx-auto border-t border-outline-variant mt-xl text-label-sm">
          <div className="flex items-center gap-md">
            <span
              className={`w-2 h-2 rounded-full ${
                readiness?.processingReady ? 'bg-emerald-400' : 'bg-tertiary'
              }`}
            ></span>
            <span className="font-label-sm text-on-surface-variant">
              {readiness
                ? `${readiness.detector.name} (ONNX) • FFmpeg ${
                    readiness.ffmpeg.version || ''
                  } • CPU inference`
                : 'SafeFrame pipeline status unavailable'}
            </span>
          </div>
          <p className="font-label-sm text-on-surface-variant opacity-70 mt-sm md:mt-0">
            © 2024 SafeFrame AI. Technical Authority in Video Privacy.
          </p>
        </footer>
      </div>
    </AppLayout>
  );
};
