import type { FrameTelemetryPacket } from '../types';

export type WebSocketStatus = 'CONNECTING' | 'CONNECTED' | 'DISCONNECTED' | 'ERROR';

export class WebSocketManager {
  private ws: WebSocket | null = null;
  private taskId: string | null = null;
  private reconnectAttempts = 0;
  private readonly maxReconnectAttempts = 5;
  private readonly reconnectIntervalMs = 2000;
  private reconnectTimer: number | null = null;
  private onTelemetryCallback: ((packet: FrameTelemetryPacket) => void) | null = null;
  private onStatusChangeCallback: ((status: WebSocketStatus) => void) | null = null;
  private isIntentionalDisconnect = false;

  public connect(
    taskId: string,
    onTelemetry: (packet: FrameTelemetryPacket) => void,
    onStatusChange?: (status: WebSocketStatus) => void
  ) {
    // Never leave an older socket (or its reconnect timer) alive.
    this.closeSocket();
    this.clearReconnectTimer();

    this.taskId = taskId;
    this.onTelemetryCallback = onTelemetry;
    this.onStatusChangeCallback = onStatusChange || null;
    this.isIntentionalDisconnect = false;
    this.reconnectAttempts = 0;

    this.initSocket();
  }

  private clearReconnectTimer() {
    if (this.reconnectTimer !== null) {
      window.clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
  }

  private closeSocket() {
    if (this.ws) {
      const socket = this.ws;
      this.ws = null;
      socket.onopen = null;
      socket.onmessage = null;
      socket.onerror = null;
      socket.onclose = null;
      try {
        socket.close();
      } catch {
        /* already closed */
      }
    }
  }

  private initSocket() {
    if (!this.taskId || this.isIntentionalDisconnect) return;

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = import.meta.env.VITE_WS_BASE_URL || `${protocol}//localhost:8000/ws/redact`;
    const url = `${host}/${this.taskId}`;

    this.notifyStatus('CONNECTING');

    try {
      const socket = new WebSocket(url);
      this.ws = socket;

      socket.onopen = () => {
        this.reconnectAttempts = 0;
        this.notifyStatus('CONNECTED');
        console.log(`[WebSocketManager] Connected to task stream: ${url}`);
      };

      socket.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (this.onTelemetryCallback) {
            // Map snake_case backend keys to the camelCase frontend schema.
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
              // The backend only sends packets for real processing events; the
              // terminal state itself is read from the REST status endpoint.
              status: data.status || 'PROCESSING',
              detectedObjects: (data.detected_objects || []).map((d: any) => ({
                id: String(d.id ?? ''),
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

      socket.onerror = () => {
        this.notifyStatus('ERROR');
      };

      socket.onclose = () => {
        if (this.ws !== socket) return; // superseded by a newer connection
        this.ws = null;
        this.notifyStatus('DISCONNECTED');
        if (!this.isIntentionalDisconnect && this.reconnectAttempts < this.maxReconnectAttempts) {
          this.reconnectAttempts += 1;
          console.log(
            `[WebSocketManager] Reconnecting ${this.reconnectAttempts}/${this.maxReconnectAttempts} in ${this.reconnectIntervalMs}ms`
          );
          this.reconnectTimer = window.setTimeout(() => {
            this.reconnectTimer = null;
            this.initSocket();
          }, this.reconnectIntervalMs);
        }
      };
    } catch (err) {
      console.error('[WebSocketManager] Failed to create WebSocket connection:', err);
      this.notifyStatus('ERROR');
    }
  }

  public disconnect() {
    this.isIntentionalDisconnect = true;
    this.clearReconnectTimer();
    this.closeSocket();
    this.taskId = null;
    this.onTelemetryCallback = null;
    this.notifyStatus('DISCONNECTED');
    this.onStatusChangeCallback = null;
  }

  private notifyStatus(status: WebSocketStatus) {
    if (this.onStatusChangeCallback) {
      this.onStatusChangeCallback(status);
    }
  }
}

export const wsManager = new WebSocketManager();
