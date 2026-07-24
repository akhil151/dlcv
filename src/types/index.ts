export type RedactionCategory =
  | 'faces'
  | 'license_plates'
  | 'id_cards'
  | 'laptop_screens'
  | 'phone_screens';

export type BlurStyle = 'gaussian' | 'pixelate' | 'blackout' | 'feathered';

export type TaskStatus = 'PENDING' | 'UPLOADING' | 'PROCESSING' | 'COMPLETED' | 'FAILED';

export type PipelineStage =
  | 'OPENING_VIDEO'
  | 'READING_FRAME'
  | 'DETECTION'
  | 'TRACKING'
  | 'REIDENTIFICATION'
  | 'GHOST_BOX'
  | 'BLUR'
  | 'ENCODING'
  | 'COMPLETED';

export interface VideoMetadata {
  videoId: string;
  filename: string;
  fileSizeBytes: number;
  format: string;
  resolution: string;
  durationSeconds: number;
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
}

export interface DetectionMetrics {
  map50: number;
  map5095: number;
  precision: number;
  recall: number;
  f1Score: number;
}

export interface TrackingMetrics {
  mota: number;
  motp: number;
  idf1: number;
  idSwitches: number;
  mostlyTrackedCount: number;
  mostlyLostCount: number;
}

export interface PrivacyMetrics {
  privacyLeakageRatePct: number;
  meanLeakDurationFrames: number;
  ghostRecoveryRatePct: number;
  avgGhostLifetimeFrames: number;
  blurCoveragePct: number;
  framesProtectedPct: number;
}

export interface PerformanceMetrics {
  avgFps: number;
  detectionLatencyMs: number;
  trackingLatencyMs: number;
  reidLatencyMs: number;
  ghostLatencyMs: number;
  blurLatencyMs: number;
  pipelineLatencyMs: number;
  peakMemoryMb: number;
}

export interface EvaluationReport {
  taskId: string;
  totalFramesEvaluated: number;
  evaluatedAt: string;
  benchmarkPassed: boolean;
  detectionMetrics: DetectionMetrics;
  trackingMetrics: TrackingMetrics;
  privacyMetrics: PrivacyMetrics;
  performanceMetrics: PerformanceMetrics;
}

export interface RedactionSummary {
  taskId: string;
  facesBlurred: number;
  licensePlatesRedacted: number;
  screensHidden: number;
  idCardsFound: number;
  phonesFound: number;
  totalObjectsCount: number;
  processingTimeSeconds: number;
  originalVideoUrl?: string;
  protectedVideoUrl?: string;
  auditLogUrl?: string;
  evaluationReport?: EvaluationReport;
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
  summary?: RedactionSummary;
  createdAt: string;
}

export interface RecentVideoItem {
  id: string;
  filename: string;
  fileSizeMb: number;
  targets: string[];
  thumbnailUrl: string;
  createdAt: string;
  duration: string;
  status: 'COMPLETED' | 'PROCESSING' | 'FAILED';
}

export interface StartRedactionRequest {
  videoId: string;
  targetCategories: RedactionCategory[];
  blurStyle: BlurStyle;
  blurIntensity: number;
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
