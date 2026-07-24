from typing import Dict, List
from fastapi import WebSocket
from backend.core.logging import logger

class ConnectionManager:
    def __init__(self):
        # task_id -> List[WebSocket]
        self.active_connections: Dict[str, List[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, task_id: str):
        await websocket.accept()
        if task_id not in self.active_connections:
            self.active_connections[task_id] = []
        self.active_connections[task_id].append(websocket)
        logger.info(f"WebSocket client connected to task '{task_id}'. Total: {len(self.active_connections[task_id])}")

    def disconnect(self, websocket: WebSocket, task_id: str):
        if task_id in self.active_connections:
            if websocket in self.active_connections[task_id]:
                self.active_connections[task_id].remove(websocket)
            if not self.active_connections[task_id]:
                del self.active_connections[task_id]
        logger.info(f"WebSocket client disconnected from task '{task_id}'")

    async def broadcast_to_task(self, task_id: str, data: dict):
        if task_id in self.active_connections:
            for connection in self.active_connections[task_id]:
                try:
                    await connection.send_json(data)
                except Exception as e:
                    logger.warning(f"Error broadcasting WS to client on task '{task_id}': {e}")

ws_manager = ConnectionManager()
