export type RedactionCategory =
  | 'faces'
  | 'license_plates'
  | 'id_cards'
  | 'passports'
  | 'laptop_screens'
  | 'phone_screens';

/**
 * Face, license-plate, identity-document, laptop-screen and phone-screen
 * redaction are the validated capabilities.
 *
 * ID cards and passports share one validated, class-agnostic document
 * detector, so both are operational but are not type-separated. Laptop and
 * phone screens use a COCO device detector plus a display-quad refinement
 * stage; when the boundary cannot be separated the backend masks the whole
 * device box and says so in the summary.
 */
export const SUPPORTED_CATEGORIES: RedactionCategory[] = [
  'faces',
  'license_plates',
  'id_cards',
  'passports',
  'laptop_screens',
  'phone_screens',
];

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
  /** License-plate regions masked across all frames (region-frames). */
  licensePlatesRedacted: number;
  /** Document regions masked across all frames (ID cards + passports). */
  documentsRedacted: number;
  /** Laptop-screen regions masked across all frames (region-frames). */
  laptopScreensRedacted: number;
  /** Screen regions masked with a refined display quadrilateral. */
  laptopScreenQuadRegions: number;
  /** Screen regions that fell back to masking the whole device box. */
  laptopScreenFallbackRegions: number;
  /** Share of laptop-screen regions that used a refined display quad (0-100). */
  laptopScreenQuadPct: number;
  /** 'refined_quad' | 'device_box_fallback' | 'mixed' | 'not_applicable'. */
  laptopScreenBoundary: string;
  /** Phone-screen regions masked across all frames (region-frames). */
  phoneScreensRedacted: number;
  /** Phone-screen regions masked with a refined display quadrilateral. */
  phoneScreenQuadRegions: number;
  /** Phone-screen regions that fell back to masking the whole device box. */
  phoneScreenFallbackRegions: number;
  /** Share of phone-screen regions that used a refined display quad (0-100). */
  phoneScreenQuadPct: number;
  /** 'refined_quad' | 'device_box_fallback' | 'mixed' | 'not_applicable'. */
  phoneScreenBoundary: string;
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
  plateDetector?: {
    name: string;
    available: boolean;
    status: string;
    device: string;
  };
  licensePlatesReady?: boolean;
  documentDetector?: {
    name: string;
    available: boolean;
    status: string;
    device: string;
  };
  documentsReady?: boolean;
  laptopDetector?: {
    name: string;
    available: boolean;
    status: string;
    device: string;
  };
  laptopScreensReady?: boolean;
  phoneDetector?: {
    name: string;
    available: boolean;
    status: string;
    device: string;
  };
  phoneScreensReady?: boolean;
  ffmpeg: {
    available: boolean;
    version?: string | null;
  };
  processingReady: boolean;
  reasons: string[];
}
