import axios from 'axios';
import type {
  VideoMetadata,
  StartRedactionRequest,
  RedactionTask,
  RedactionSummary,
  RecentVideoItem,
} from '../types';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000/api/v1';

export const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 30000,
});

apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    const message = error.response?.data?.message || error.message || 'An API error occurred';
    console.error('[API Error]:', message);
    return Promise.reject(new Error(message));
  }
);

export const api = {
  // Health check
  healthCheck: async (): Promise<{ status: string }> => {
    const res = await apiClient.get('/health');
    return res.data;
  },

  // Upload raw video file
  uploadVideo: async (
    file: File,
    onProgress?: (pct: number) => void
  ): Promise<VideoMetadata> => {
    const formData = new FormData();
    formData.append('file', file);

    const res = await apiClient.post('/videos/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      onUploadProgress: (evt) => {
        if (evt.total && onProgress) {
          onProgress(Math.round((evt.loaded * 100) / evt.total));
        }
      },
    });

    const d = res.data;
    return {
      videoId: d.video_id,
      filename: d.filename,
      fileSizeBytes: d.file_size_bytes,
      format: d.format,
      resolution: d.resolution,
      durationSeconds: d.duration_seconds,
      estimatedProcessingSeconds: d.estimated_processing_seconds,
    };
  },

  // Start redaction process
  startRedaction: async (req: StartRedactionRequest): Promise<{ taskId: string }> => {
    const payload = {
      video_id: req.videoId,
      target_categories: req.targetCategories,
      blur_style: req.blurStyle,
      blur_intensity: req.blurIntensity,
    };
    const res = await apiClient.post<{ task_id: string }>('/redact/start', payload);
    return { taskId: res.data.task_id };
  },

  // Get task status & telemetry
  getTaskStatus: async (taskId: string): Promise<RedactionTask> => {
    const res = await apiClient.get(`/redact/status/${taskId}`);
    const d = res.data;
    return {
      taskId: d.task_id,
      videoId: d.video_id,
      filename: d.filename,
      status: d.status,
      progressPct: d.progress_pct,
      targetCategories: d.target_categories,
      blurStyle: d.blur_style,
      blurIntensity: d.blur_intensity,
      metrics: {
        fps: d.metrics.fps,
        elapsedSeconds: d.metrics.elapsed_seconds,
        remainingSeconds: d.metrics.remaining_seconds,
        throughputFrames: d.metrics.throughput_frames,
        totalFrames: d.metrics.total_frames,
        processedFrames: d.metrics.processed_frames,
      },
      recentLogs: d.recent_logs || [],
      createdAt: d.created_at || new Date().toISOString(),
    };
  },

  // Get task completion summary & evaluation report
  getTaskSummary: async (taskId: string): Promise<RedactionSummary> => {
    const res = await apiClient.get(`/redact/results/${taskId}`);
    const d = res.data;
    return {
      taskId: d.task_id,
      facesBlurred: d.faces_blurred,
      licensePlatesRedacted: d.license_plates_redacted,
      screensHidden: d.screens_hidden,
      idCardsFound: d.id_cards_found,
      phonesFound: d.phones_found || 0,
      totalObjectsCount: d.total_objects_count,
      processingTimeSeconds: d.processing_time_seconds,
      originalVideoUrl: `${API_BASE_URL}/videos/download/${taskId}?raw=true`,
      protectedVideoUrl: `${API_BASE_URL}/videos/download/${taskId}`,
      auditLogUrl: `${API_BASE_URL}/videos/download/${taskId}?audit=true`,
      evaluationReport: d.evaluation_report
        ? {
            taskId: d.evaluation_report.task_id,
            totalFramesEvaluated: d.evaluation_report.total_frames_evaluated,
            evaluatedAt: d.evaluation_report.evaluated_at,
            benchmarkPassed: d.evaluation_report.benchmark_passed,
            detectionMetrics: {
              map50: d.evaluation_report.detection_metrics.map_50,
              map5095: d.evaluation_report.detection_metrics.map_50_95,
              precision: d.evaluation_report.detection_metrics.precision,
              recall: d.evaluation_report.detection_metrics.recall,
              f1Score: d.evaluation_report.detection_metrics.f1_score,
            },
            trackingMetrics: {
              mota: d.evaluation_report.tracking_metrics.mota,
              motp: d.evaluation_report.tracking_metrics.motp,
              idf1: d.evaluation_report.tracking_metrics.idf1,
              idSwitches: d.evaluation_report.tracking_metrics.id_switches,
              mostlyTrackedCount: d.evaluation_report.tracking_metrics.mostly_tracked_count,
              mostlyLostCount: d.evaluation_report.tracking_metrics.mostly_lost_count,
            },
            privacyMetrics: {
              privacyLeakageRatePct: d.evaluation_report.privacy_metrics.privacy_leakage_rate_pct,
              meanLeakDurationFrames: d.evaluation_report.privacy_metrics.mean_leak_duration_frames,
              ghostRecoveryRatePct: d.evaluation_report.privacy_metrics.ghost_recovery_rate_pct,
              avgGhostLifetimeFrames: d.evaluation_report.privacy_metrics.avg_ghost_lifetime_frames,
              blurCoveragePct: d.evaluation_report.privacy_metrics.blur_coverage_pct,
              framesProtectedPct: d.evaluation_report.privacy_metrics.frames_protected_pct,
            },
            performanceMetrics: {
              avgFps: d.evaluation_report.performance_metrics.avg_fps,
              detectionLatencyMs: d.evaluation_report.performance_metrics.detection_latency_ms,
              trackingLatencyMs: d.evaluation_report.performance_metrics.tracking_latency_ms,
              reidLatencyMs: d.evaluation_report.performance_metrics.reid_latency_ms,
              ghostLatencyMs: d.evaluation_report.performance_metrics.ghost_latency_ms,
              blurLatencyMs: d.evaluation_report.performance_metrics.blur_latency_ms,
              pipelineLatencyMs: d.evaluation_report.performance_metrics.pipeline_latency_ms,
              peakMemoryMb: d.evaluation_report.performance_metrics.peak_memory_mb,
            },
          }
        : undefined,
    };
  },

  // Get dashboard statistics and recent video activity
  getDashboardData: async (): Promise<{
    stats: {
      videosProcessed: number;
      objectsDetected: number;
      avgProcessingTimeSeconds: number;
    };
    recentVideos: RecentVideoItem[];
  }> => {
    const res = await apiClient.get('/dashboard/overview');
    const d = res.data;
    return {
      stats: {
        videosProcessed: d.stats.videos_processed,
        objectsDetected: d.stats.objects_detected,
        avgProcessingTimeSeconds: d.stats.avg_processing_time_seconds,
      },
      recentVideos: (d.recent_videos || []).map((v: any) => ({
        id: v.id,
        filename: v.filename,
        fileSizeMb: v.file_size_mb,
        targets: v.targets,
        thumbnailUrl: v.thumbnail_url,
        createdAt: v.created_at,
        duration: v.duration,
        status: v.status,
      })),
    };
  },

  getDownloadUrl: (taskId: string, isRaw = false) =>
    `${API_BASE_URL}/videos/download/${taskId}${isRaw ? '?raw=true' : ''}`,

  getAuditLogUrl: (taskId: string) => `${API_BASE_URL}/videos/download/${taskId}?audit=true`,
};
