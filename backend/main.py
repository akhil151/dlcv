import asyncio
import contextlib

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from backend.config.settings import settings
from backend.core.logging import setup_logging, logger
from backend.core.exceptions import (
    SafeFrameException,
    safeframe_exception_handler,
    global_exception_handler,
)
from backend.api.v1.router import api_v1_router
from backend.websocket.connection_manager import ws_manager
from backend.services.ai import model_manager

# Initialize logger
setup_logging()


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    """Loads the face detector at startup.

    Warmup runs off the event loop so the API accepts requests immediately. A
    detector failure is recorded in its status and surfaced by /health/readiness;
    it is never replaced by a substitute detector.
    """
    logger.info("Starting %s backend service...", settings.PROJECT_NAME)
    await asyncio.to_thread(model_manager.warmup)
    yield
    logger.info("Shutting down SafeFrame AI backend service...")


app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Backend API platform for SafeFrame AI Privacy Redaction Engine",
    version="1.0.0",
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    docs_url=f"{settings.API_V1_STR}/docs",
    redoc_url=f"{settings.API_V1_STR}/redoc",
    lifespan=lifespan,
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register Exception Handlers
app.add_exception_handler(SafeFrameException, safeframe_exception_handler)
app.add_exception_handler(Exception, global_exception_handler)

# Include API v1 Routers
app.include_router(api_v1_router, prefix=settings.API_V1_STR)


# WebSocket Telemetry Endpoint
@app.websocket("/ws/redact/{task_id}")
async def websocket_redaction_telemetry(websocket: WebSocket, task_id: str):
    await ws_manager.connect(websocket, task_id)
    try:
        while True:
            # Keep socket alive and receive any incoming commands from client
            data = await websocket.receive_text()
            logger.debug(f"Received WS message on task '{task_id}': {data}")
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket, task_id)
    except Exception as e:
        logger.warning(f"WebSocket error on task '{task_id}': {e}")
        ws_manager.disconnect(websocket, task_id)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
    )
