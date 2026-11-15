import axios from 'axios';
import type {
  VideoMetadata,
  StartRedactionRequest,
  RedactionTask,
  RedactionSummary,
  RecentVideoItem,
  ReadinessSnapshot,
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
    const message =
      error.response?.data?.message ||
      error.response?.data?.detail?.[0]?.msg ||
      error.response?.data?.detail ||
      error.message ||
      'An API error occurred';
    console.error('[API Error]:', message);
    return Promise.reject(new Error(String(message)));
  }
);

const metricsFromApi = (m: Record<string, unknown> | undefined) => ({
  fps: Number(m?.fps ?? 0),
  elapsedSeconds: Number(m?.elapsed_seconds ?? 0),
  remainingSeconds: Number(m?.remaining_seconds ?? 0),
  throughputFrames: Number(m?.throughput_frames ?? 0),
  totalFrames: Number(m?.total_frames ?? 0),
  processedFrames: Number(m?.processed_frames ?? 0),
  framesWithDetections: Number(m?.frames_with_detections ?? 0),
  masksTotal: Number(m?.masks_total ?? 0),
  newTracksTotal: Number(m?.new_tracks_total ?? 0),
  detectionLatencyMs: Number(m?.detection_latency_ms ?? 0),
  blurLatencyMs: Number(m?.blur_latency_ms ?? 0),
  avgProcessingFps: Number(m?.avg_processing_fps ?? 0),
  debugFramesSaved: Number(m?.debug_frames_saved ?? 0),
});

export const api = {
  healthCheck: async (): Promise<{ status: string }> => {
    const res = await apiClient.get('/health');
    return res.data;
  },

  /** Real dependency readiness: face detector weights + FFmpeg. */
  getReadiness: async (): Promise<ReadinessSnapshot> => {
    const res = await apiClient.get('/health/readiness');
    const d = res.data;
    return {
      status: d.status,
      process: d.process,
      modelsLoaded: Boolean(d.models_loaded),
      detector: {
        name: d.detector?.name ?? 'YUNET',
        available: Boolean(d.detector?.available),
        status: d.detector?.status ?? 'UNKNOWN',
        device: d.detector?.device ?? 'cpu',
      },
      plateDetector: {
        name: d.plate_detector?.name ?? 'RTDETR_R18_PLATE',
        available: Boolean(d.plate_detector?.available),
        status: d.plate_detector?.status ?? 'UNKNOWN',
        device: d.plate_detector?.device ?? 'cpu',
      },
      licensePlatesReady: Boolean(d.license_plates_ready),
      documentDetector: {
        name: d.document_detector?.name ?? 'DOCALIGNER_FASTVIT_SA24',
        available: Boolean(d.document_detector?.available),
        status: d.document_detector?.status ?? 'UNKNOWN',
        device: d.document_detector?.device ?? 'cpu',
      },
      documentsReady: Boolean(d.documents_ready),
      laptopDetector: {
        name: d.laptop_detector?.name ?? 'RTDETR_R18VD_COCO',
        available: Boolean(d.laptop_detector?.available),
        status: d.laptop_detector?.status ?? 'UNKNOWN',
        device: d.laptop_detector?.device ?? 'cpu',
      },
      laptopScreensReady: Boolean(d.laptop_screens_ready),
      phoneDetector: {
        name: d.phone_detector?.name ?? 'RTDETR_R18VD_COCO_PHONE',
        available: Boolean(d.phone_detector?.available),
        status: d.phone_detector?.status ?? 'UNKNOWN',
        device: d.phone_detector?.device ?? 'cpu',
      },
      phoneScreensReady: Boolean(d.phone_screens_ready),
      ffmpeg: {
        available: Boolean(d.ffmpeg?.available),
        version: d.ffmpeg?.version ?? null,
      },
      processingReady: Boolean(d.processing_ready),
      reasons: Array.isArray(d.reasons) ? d.reasons : [],
    };
  },

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
      // Large uploads can take much longer than the default request timeout.
      timeout: 15 * 60 * 1000,
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
      fps: Number(d.fps ?? 0),
      frameCount: Number(d.frame_count ?? 0),
      estimatedProcessingSeconds: d.estimated_processing_seconds,
    };
  },

  startRedaction: async (req: StartRedactionRequest): Promise<{ taskId: string }> => {
    const payload = {
      video_id: req.videoId,
      target_categories: req.targetCategories,
      blur_style: req.blurStyle,
      blur_intensity: req.blurIntensity,
      debug_mode: req.debugMode ?? false,
    };
    const res = await apiClient.post<{ task_id: string }>('/redact/start', payload);
    return { taskId: res.data.task_id };
  },

  getTaskStatus: async (taskId: string): Promise<RedactionTask> => {
    const res = await apiClient.get(`/redact/status/${taskId}`);
    const d = res.data;
    return {
      taskId: d.task_id,
      videoId: d.video_id,
      filename: d.filename,
      status: d.status,
      progressPct: d.progress_pct,
      targetCategories: d.target_categories ?? [],
      blurStyle: d.blur_style,
      blurIntensity: d.blur_intensity,
      metrics: metricsFromApi(d.metrics),
      recentLogs: d.recent_logs || [],
      createdAt: d.created_at || '',
      debugMode: Boolean(d.debug_mode),
      outputVerified: d.output_verified ?? undefined,
      failureReason: d.failure_reason ?? null,
    };
  },

  cancelTask: async (taskId: string): Promise<RedactionTask> => {
    const res = await apiClient.post(`/redact/cancel/${taskId}`);
    const d = res.data;
    return {
      taskId: d.task_id,
      videoId: d.video_id,
      filename: d.filename,
      status: d.status,
      progressPct: d.progress_pct,
      targetCategories: d.target_categories ?? [],
      blurStyle: d.blur_style,
      blurIntensity: d.blur_intensity,
      metrics: metricsFromApi(d.metrics),
      recentLogs: d.recent_logs || [],
      createdAt: d.created_at || '',
      debugMode: Boolean(d.debug_mode),
      outputVerified: d.output_verified ?? undefined,
      failureReason: d.failure_reason ?? null,
    };
  },

  getTaskSummary: async (taskId: string): Promise<RedactionSummary> => {
    const res = await apiClient.get(`/redact/results/${taskId}`);
    const d = res.data;
    return {
      taskId: d.task_id,
      facesBlurred: d.faces_blurred,
      licensePlatesRedacted: d.license_plates_redacted ?? 0,
      documentsRedacted: d.documents_redacted ?? 0,
      laptopScreensRedacted: d.laptop_screens_redacted ?? 0,
      laptopScreenQuadRegions: d.laptop_screen_quad_regions ?? 0,
      laptopScreenFallbackRegions: d.laptop_screen_fallback_regions ?? 0,
      laptopScreenQuadPct: d.laptop_screen_quad_pct ?? 0,
      laptopScreenBoundary: d.laptop_screen_boundary ?? 'not_applicable',
      phoneScreensRedacted: d.phone_screens_redacted ?? 0,
      phoneScreenQuadRegions: d.phone_screen_quad_regions ?? 0,
      phoneScreenFallbackRegions: d.phone_screen_fallback_regions ?? 0,
      phoneScreenQuadPct: d.phone_screen_quad_pct ?? 0,
      phoneScreenBoundary: d.phone_screen_boundary ?? 'not_applicable',
      totalObjectsCount: d.total_objects_count,
      processingTimeSeconds: d.processing_time_seconds,
      framesProcessed: d.frames_processed ?? 0,
      framesWithDetections: d.frames_with_detections ?? 0,
      detectionFramePct: d.detection_frame_pct ?? 0,
      avgProcessingFps: d.avg_processing_fps ?? 0,
      verifiedFrameCount: d.verified_frame_count ?? undefined,
      outputResolution: d.output_resolution ?? undefined,
      audioPreserved: d.audio_preserved ?? undefined,
      blurStyle: d.blur_style ?? undefined,
      blurIntensity: d.blur_intensity ?? undefined,
      originalVideoUrl: d.original_video_url
        ? `${API_BASE_URL}${d.original_video_url.replace(/^\/api\/v1/, '')}`
        : `${API_BASE_URL}/videos/original/${taskId}`,
      protectedVideoUrl: `${API_BASE_URL}/videos/download/${taskId}`,
    };
  },

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

  getDownloadUrl: (taskId: string) => `${API_BASE_URL}/videos/download/${taskId}`,

  getOriginalUrl: (taskId: string) => `${API_BASE_URL}/videos/original/${taskId}`,
};
