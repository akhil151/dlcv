export type RedactionCategory =
  | 'faces'
  | 'license_plates'
  | 'id_cards'
  | 'laptop_screens'
  | 'phone_screens';

/**
 * Face redaction is the only validated capability. The other categories are
 * shown in the UI as "Coming Later" and are never sent to the backend.
 */
export const SUPPORTED_CATEGORIES: RedactionCategory[] = ['faces'];

/** Client-side pre-check only; the backend validates the actual container. */
export const ACCEPTED_VIDEO_EXTENSIONS = ['.mp4', '.avi', '.mov', '.mkv', '.webm'];

export type BlurStyle = 'gaussian' | 'pixelate' | 'blackout';

export type TaskStatus = 'PENDING' | 'UPLOADING' | 'PROCESSING' | 'COMPLETED' | 'FAILED' | 'CANCELLED';

export interface VideoMetadata {
  videoId: string;
  filename: string;
  fileSizeBytes: number;
  format: string;
  resolution: string;
  durationSeconds: number;
  fps: number;
  frameCount: number;
  estimatedProcessingSeconds: number;
}

export interface BoundingBox {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface DetectedEntity {
  id: string;
  category: RedactionCategory;
  label: string;
  confidence: number;
  bbox: BoundingBox;
}

export interface ProcessingMetrics {
  fps: number;
  elapsedSeconds: number;
  remainingSeconds: number;
  throughputFrames: number;
  totalFrames: number;
  processedFrames: number;
  framesWithDetections: number;
  masksTotal: number;
  newTracksTotal: number;
  detectionLatencyMs: number;
  blurLatencyMs: number;
  avgProcessingFps: number;
  debugFramesSaved: number;
}

export interface RedactionSummary {
  taskId: string;
  /** Face regions masked across all frames (region-frames). */
  facesBlurred: number;
  totalObjectsCount: number;
  processingTimeSeconds: number;
  framesProcessed: number;
  framesWithDetections: number;
  detectionFramePct: number;
  avgProcessingFps: number;
  verifiedFrameCount?: number;
  outputResolution?: string;
  audioPreserved?: boolean;
  blurStyle?: string;
  blurIntensity?: number;
  originalVideoUrl?: string;
  protectedVideoUrl?: string;
}

export interface RedactionTask {
  taskId: string;
  videoId: string;
  filename: string;
  status: TaskStatus;
  progressPct: number;
  targetCategories: RedactionCategory[];
  blurStyle: BlurStyle;
  blurIntensity: number;
  metrics: ProcessingMetrics;
  recentLogs: string[];
  createdAt: string;
  debugMode: boolean;
  /** Backend-reported verification flag. Undefined until the task is terminal. */
  outputVerified?: boolean;
  /** Present for FAILED/CANCELLED tasks; null otherwise. */
  failureReason?: string | null;
}

export interface RecentVideoItem {
  id: string;
  filename: string;
  fileSizeMb: number;
  targets: string[];
  thumbnailUrl: string;
  createdAt: string;
  duration: string;
  status: 'COMPLETED' | 'PROCESSING' | 'FAILED' | 'CANCELLED';
}

export interface StartRedactionRequest {
  videoId: string;
  targetCategories: RedactionCategory[];
  blurStyle: BlurStyle;
  blurIntensity: number;
  debugMode?: boolean;
}

export interface FrameTelemetryPacket {
  taskId: string;
  frameIndex: number;
  totalFrames: number;
  progressPct: number;
  fps: number;
  currentStage: string;
  activeDetections: number;
  activeTracks: number;
  ghostBoxCount: number;
  elapsedTime: number;
  estimatedRemainingTime: number;
  logLine?: string;
  status: string;
  detectedObjects: DetectedEntity[];
}

export interface ReadinessSnapshot {
  status: 'ready' | 'unavailable';
  process: string;
  modelsLoaded: boolean;
  detector: {
    name: string;
    available: boolean;
    status: string;
    device: string;
  };
  ffmpeg: {
    available: boolean;
    version?: string | null;
  };
  processingReady: boolean;
  reasons: string[];
}
