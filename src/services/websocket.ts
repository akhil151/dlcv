import type { FrameTelemetryPacket } from '../types';

export type WebSocketStatus = 'CONNECTING' | 'CONNECTED' | 'DISCONNECTED' | 'ERROR';

export class WebSocketManager {
  private ws: WebSocket | null = null;
  private taskId: string | null = null;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectIntervalMs = 2000;
  private onTelemetryCallback: ((packet: FrameTelemetryPacket) => void) | null = null;
  private onStatusChangeCallback: ((status: WebSocketStatus) => void) | null = null;
  private isIntentionalDisconnect = false;

  constructor() {}

  public connect(
    taskId: string,
    onTelemetry: (packet: FrameTelemetryPacket) => void,
    onStatusChange?: (status: WebSocketStatus) => void
  ) {
    this.taskId = taskId;
    this.onTelemetryCallback = onTelemetry;
    this.onStatusChangeCallback = onStatusChange || null;
    this.isIntentionalDisconnect = false;
    this.reconnectAttempts = 0;

    this.initSocket();
  }

  private initSocket() {
    if (!this.taskId) return;

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = import.meta.env.VITE_WS_BASE_URL || `${protocol}//localhost:8000/ws/redact`;
    const url = `${host}/${this.taskId}`;

    this.notifyStatus('CONNECTING');

    try {
      this.ws = new WebSocket(url);

      this.ws.onopen = () => {
        this.reconnectAttempts = 0;
        this.notifyStatus('CONNECTED');
        console.log(`[WebSocketManager] Connected to task stream: ${url}`);
      };

      this.ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (this.onTelemetryCallback) {
            // Map snake_case backend keys to camelCase frontend schema
            const packet: FrameTelemetryPacket = {
              taskId: data.task_id || this.taskId || '',
              frameIndex: data.frame_index || 0,
              totalFrames: data.total_frames || 0,
              progressPct: data.progress_pct || 0,
              fps: data.fps || 0,
              currentStage: data.current_stage || 'Processing',
              activeDetections: data.active_detections || 0,
              activeTracks: data.active_tracks || 0,
              ghostBoxCount: data.ghost_box_count || 0,
              elapsedTime: data.elapsed_time || 0,
              estimatedRemainingTime: data.estimated_remaining_time || 0,
              logLine: data.log_line || '',
              status: data.status || 'PROCESSING',
              detectedObjects: (data.detected_objects || []).map((d: any) => ({
                id: d.id || '',
                category: d.category || 'faces',
                label: d.label || '',
                confidence: d.confidence || 0,
                bbox: d.bbox || { x: 0, y: 0, width: 0, height: 0 },
              })),
            };
            this.onTelemetryCallback(packet);
          }
        } catch (err) {
          console.error('[WebSocketManager] Error parsing packet JSON:', err);
        }
      };

      this.ws.onerror = (err) => {
        console.warn('[WebSocketManager] Socket error encountered:', err);
        this.notifyStatus('ERROR');
      };

      this.ws.onclose = () => {
        this.notifyStatus('DISCONNECTED');
        if (!this.isIntentionalDisconnect && this.reconnectAttempts < this.maxReconnectAttempts) {
          this.reconnectAttempts++;
          console.log(
            `[WebSocketManager] Socket closed. Reconnecting attempt ${this.reconnectAttempts}/${this.maxReconnectAttempts} in ${this.reconnectIntervalMs}ms...`
          );
          setTimeout(() => this.initSocket(), this.reconnectIntervalMs);
        }
      };
    } catch (err) {
      console.error('[WebSocketManager] Failed to create WebSocket connection:', err);
      this.notifyStatus('ERROR');
    }
  }

  public disconnect() {
    this.isIntentionalDisconnect = true;
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
    this.notifyStatus('DISCONNECTED');
  }

  private notifyStatus(status: WebSocketStatus) {
    if (this.onStatusChangeCallback) {
      this.onStatusChangeCallback(status);
    }
  }
}

export const wsManager = new WebSocketManager();
