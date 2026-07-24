import { create } from 'zustand';
import type {
  RedactionCategory,
  BlurStyle,
  VideoMetadata,
  RedactionTask,
  FrameTelemetryPacket,
  RedactionSummary,
} from '../types';
import type { WebSocketStatus } from '../services/websocket';

interface AppState {
  // Upload & Configuration Draft State
  selectedFile: File | null;
  videoMetadata: VideoMetadata | null;
  uploadProgressPct: number;
  selectedCategories: RedactionCategory[];
  blurStyle: BlurStyle;
  blurIntensity: number;

  // Active Task & Telemetry State
  activeTaskId: string | null;
  activeTask: RedactionTask | null;
  wsStatus: WebSocketStatus;
  latestTelemetry: FrameTelemetryPacket | null;
  activeSummary: RedactionSummary | null;

  // Actions
  setSelectedFile: (file: File | null) => void;
  setVideoMetadata: (metadata: VideoMetadata | null) => void;
  setUploadProgressPct: (pct: number) => void;
  toggleCategory: (category: RedactionCategory) => void;
  setSelectedCategories: (categories: RedactionCategory[]) => void;
  setBlurStyle: (style: BlurStyle) => void;
  setBlurIntensity: (intensity: number) => void;

  setActiveTaskId: (taskId: string | null) => void;
  setActiveTask: (task: RedactionTask | null) => void;
  setWsStatus: (status: WebSocketStatus) => void;
  setLatestTelemetry: (telemetry: FrameTelemetryPacket | null) => void;
  setActiveSummary: (summary: RedactionSummary | null) => void;
  resetUploadDraft: () => void;
}

const DEFAULT_CATEGORIES: RedactionCategory[] = [
  'faces',
  'license_plates',
  'id_cards',
  'laptop_screens',
  'phone_screens',
];

export const useAppStore = create<AppState>((set) => ({
  selectedFile: null,
  videoMetadata: null,
  uploadProgressPct: 0,
  selectedCategories: DEFAULT_CATEGORIES,
  blurStyle: 'gaussian',
  blurIntensity: 80,

  activeTaskId: null,
  activeTask: null,
  wsStatus: 'DISCONNECTED',
  latestTelemetry: null,
  activeSummary: null,

  setSelectedFile: (file) => set({ selectedFile: file }),
  setVideoMetadata: (metadata) => set({ videoMetadata: metadata }),
  setUploadProgressPct: (pct) => set({ uploadProgressPct: pct }),

  toggleCategory: (category) =>
    set((state) => {
      const exists = state.selectedCategories.includes(category);
      const updated = exists
        ? state.selectedCategories.filter((c) => c !== category)
        : [...state.selectedCategories, category];
      return { selectedCategories: updated };
    }),

  setSelectedCategories: (categories) => set({ selectedCategories: categories }),
  setBlurStyle: (style) => set({ blurStyle: style }),
  setBlurIntensity: (intensity) => set({ blurIntensity: intensity }),

  setActiveTaskId: (taskId) => set({ activeTaskId: taskId }),
  setActiveTask: (task) => set({ activeTask: task }),
  setWsStatus: (status) => set({ wsStatus: status }),
  setLatestTelemetry: (telemetry) => set({ latestTelemetry: telemetry }),
  setActiveSummary: (summary) => set({ activeSummary: summary }),

  resetUploadDraft: () =>
    set({
      selectedFile: null,
      videoMetadata: null,
      uploadProgressPct: 0,
      selectedCategories: DEFAULT_CATEGORIES,
      blurStyle: 'gaussian',
      blurIntensity: 80,
    }),
}));
