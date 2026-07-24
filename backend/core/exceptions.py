from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse
from backend.core.logging import logger

class SafeFrameException(Exception):
    def __init__(self, message: str, status_code: int = status.HTTP-400-BAD-REQUEST):
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)

class TaskNotFoundException(SafeFrameException):
    def __init__(self, task_id: str):
        super().__init__(
            message=f"Redaction task '{task_id}' not found.",
            status_code=status.HTTP_404_NOT_FOUND
        )

class VideoNotFoundException(SafeFrameException):
    def __init__(self, video_id: str):
        super().__init__(
            message=f"Video resource '{video_id}' not found.",
            status_code=status.HTTP_404_NOT_FOUND
        )

class FileUploadException(SafeFrameException):
    def __init__(self, detail: str):
        super().__init__(
            message=f"File upload failed: {detail}",
            status_code=status.HTTP_400_BAD_REQUEST
        )

async def safeframe_exception_handler(request: Request, exc: SafeFrameException):
    logger.warning(f"Domain Exception [{exc.status_code}] on {request.url.path}: {exc.message}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"success": False, "message": exc.message, "status_code": exc.status_code}
    )

async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled Exception on {request.url.path}: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"success": False, "message": "An internal server error occurred.", "status_code": 500}
    )
