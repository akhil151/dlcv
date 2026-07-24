import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { AppLayout } from '../components/layout/AppLayout';
import { WelcomeBanner } from '../components/dashboard/WelcomeBanner';
import { StatOverviewGrid, type StatItem } from '../components/dashboard/StatOverviewGrid';
import { RecentVideoTable } from '../components/dashboard/RecentVideoTable';
import { api } from '../services/api';
import type { RecentVideoItem } from '../types';

export const DashboardPage: React.FC = () => {
  const navigate = useNavigate();
  const [stats, setStats] = useState<StatItem[]>([
    {
      id: 'videos-processed',
      label: 'Videos Processed',
      value: 0,
      subtext: 'Total ingest tasks',
      colorVariant: 'primary',
    },
    {
      id: 'objects-detected',
      label: 'Entities Redacted',
      value: 0,
      subtext: 'Faces & Plates',
      colorVariant: 'secondary',
    },
    {
      id: 'avg-processing-time',
      label: 'Avg Inference Time',
      value: '0s',
      subtext: 'CUDA acceleration',
      colorVariant: 'tertiary',
    },
  ]);
  const [recentVideos, setRecentVideos] = useState<RecentVideoItem[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [isShowingAll, setIsShowingAll] = useState(false);

  useEffect(() => {
    const fetchDashboard = async () => {
      setIsLoading(true);
      try {
        const data = await api.getDashboardData();
        if (data && data.stats) {
          setStats([
            {
              id: 'videos-processed',
              label: 'Videos Processed',
              value: data.stats.videosProcessed ?? 0,
              subtext: 'Total completed tasks',
              colorVariant: 'primary',
            },
            {
              id: 'objects-detected',
              label: 'Entities Redacted',
              value: Number(data.stats.objectsDetected ?? 0).toLocaleString(),
              subtext: 'Across all streams',
              colorVariant: 'secondary',
            },
            {
              id: 'avg-processing-time',
              label: 'Avg Inference Time',
              value: `${data.stats.avgProcessingTimeSeconds ?? 0}s`,
              subtext: 'TensorRT pipeline',
              colorVariant: 'tertiary',
            },
          ]);
        }
        if (data && data.recentVideos) {
          setRecentVideos(data.recentVideos);
        }
      } catch (err) {
        console.error('[DashboardPage] Failed to fetch dashboard data:', err);
      } finally {
        setIsLoading(false);
      }
    };

    fetchDashboard();
  }, []);

  const handleViewAll = () => {
    setIsShowingAll(!isShowingAll);
  };

  return (
    <AppLayout title="Dashboard">
      <div className="max-w-container-max mx-auto w-full flex flex-col gap-xl">
        {/* Welcome Hero Control Center Banner */}
        <WelcomeBanner
          tasksCompletedCount={Number(stats[0]?.value) || 0}
          onUploadClick={() => navigate('/upload')}
        />

        {/* Aggregate Statistic Metric Cards */}
        <StatOverviewGrid stats={stats} />

        {/* Recent Processing Table */}
        <RecentVideoTable
          videos={isShowingAll ? recentVideos : recentVideos.slice(0, 5)}
          isLoading={isLoading}
          onVideoClick={(video) => navigate(`/results/${video.id}`)}
          onViewAllClick={handleViewAll}
        />

        {/* Dashboard Technical Footer */}
        <footer className="w-full py-lg px-xl flex flex-col md:flex-row justify-between items-center max-w-container-max mx-auto border-t border-outline-variant mt-xl text-label-sm">
          <div className="flex items-center gap-md">
            <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
            <span className="font-label-sm text-on-surface-variant">
              SafeFrame Neural Pipeline • YOLOv11 TensorRT Active
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
