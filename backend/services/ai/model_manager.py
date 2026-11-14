import os
import sys
import time
import threading
from typing import Dict, Any, Optional, Tuple
from backend.config.ai_settings import ai_settings
from backend.core.logging import logger

# .../backend/services/ai/model_manager.py -> project root (four levels up).
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

class ModelManager:
    """
    Thread-safe Singleton ModelManager for SafeFrame AI Engine.

    Responsibilities:
    - Load the face detector (YuNet ONNX) exactly once.
    - Report explicit LOADED / UNAVAILABLE / ERROR status for it.

    The bundled YOLO11m weights and the OSNet Re-ID model are deliberately not
    loaded: they are not used by any active stage (YOLO has no face semantics
    and Re-ID is not wired into association). Loading them only added startup
    cost and memory, and made readiness reports misleading.
    """
    _instance: Optional['ModelManager'] = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls) -> 'ModelManager':
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(ModelManager, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if getattr(self, '_initialized', False):
            return

        with self._lock:
            if getattr(self, '_initialized', False):
                return

            self._initialized = True
            # Keep imports and API routing lightweight. Hardware probing and
            # model loading occur only when a processing job is preflighted or
            # startup warmup explicitly requests them.
            self.device, self.is_cuda, self.vram_gb = "cpu", False, 0.0
            self.models: Dict[str, Any] = {}
            self.models_loaded: bool = False
            self.is_warmed_up: bool = False

    def _resolve_hardware_device(self) -> Tuple[str, bool, float]:
        """Returns the inference device actually used by the active detector.

        YuNet runs through OpenCV's DNN backend on CPU in this build; no CUDA
        execution path is active, so none is claimed.
        """
        logger.info("[ModelManager] Detector device: CPU (OpenCV DNN, YuNet ONNX).")
        return "cpu", False, 0.0

    def resolve_yunet_path(self) -> str:
        """Resolves the YuNet weight path irrespective of the working directory.

        An absolute configured path is used as-is. A relative path is resolved
        against the project root first, then against the current working
        directory as a fallback.
        """
        configured = ai_settings.YUNET_MODEL_PATH
        if os.path.isabs(configured):
            return configured
        project_candidate = os.path.join(PROJECT_ROOT, configured)
        if os.path.isfile(project_candidate):
            return project_candidate
        cwd_candidate = os.path.abspath(configured)
        if os.path.isfile(cwd_candidate):
            return cwd_candidate
        return project_candidate

    def _load_yunet(self) -> Tuple[Optional[Any], str, str, str]:
        """Initializes YuNet and returns (model, status, detail, resolved_path).

        status is one of LOADED / UNAVAILABLE / ERROR. Failures are explicit and
        never trigger a substitute detector.
        """
        resolved = self.resolve_yunet_path()
        try:
            import cv2
        except Exception as exc:  # pragma: no cover - OpenCV is a hard dependency
            return None, "UNAVAILABLE", f"OpenCV is not importable: {exc}", resolved

        if not os.path.isfile(resolved):
            return (None, "UNAVAILABLE",
                    f"YuNet ONNX weights not found at '{resolved}'. "
                    "Set YUNET_MODEL_PATH or place face_detection_yunet_2023mar.onnx in the project root.",
                    resolved)
        if os.path.getsize(resolved) == 0:
            return None, "UNAVAILABLE", f"YuNet ONNX weights at '{resolved}' are empty (0 bytes).", resolved
        if not hasattr(cv2, "FaceDetectorYN"):
            return (None, "UNAVAILABLE",
                    "This OpenCV build does not expose cv2.FaceDetectorYN; the YuNet detector cannot be constructed.",
                    resolved)

        try:
            logger.info(f"[ModelManager] Loading YuNet ONNX model from '{resolved}'...")
            model = cv2.FaceDetectorYN.create(
                model=resolved,
                config="",
                input_size=(320, 320),
                score_threshold=ai_settings.SECONDARY_FACE_CONF_THRESHOLD,
                nms_threshold=0.3,
                top_k=5000,
            )
            if model is None:
                return None, "ERROR", "cv2.FaceDetectorYN.create returned no detector.", resolved
            logger.info("[ModelManager] Successfully Loaded YuNet")
            return model, "LOADED", "ok", resolved
        except Exception as exc:
            return None, "ERROR", f"YuNet initialization failed for '{resolved}': {exc}", resolved

    def detector_readiness(self) -> Dict[str, Any]:
        """Cheap, non-loading snapshot of the face-detector dependency.

        If models are already loaded the cached handle is reported. Otherwise a
        filesystem-only probe is performed (no YOLO/OSNet loading, no inference).
        """
        handle = self.models.get("secondary_face")
        if handle is not None:
            return {
                "available": handle.get("status") == "LOADED",
                "status": handle.get("status"),
                "detail": handle.get("detail", ""),
                "weights_path": handle.get("weights_path"),
            }
        resolved = self.resolve_yunet_path()
        if not os.path.isfile(resolved):
            return {"available": False, "status": "UNAVAILABLE",
                    "detail": f"YuNet ONNX weights not found at '{resolved}'.", "weights_path": resolved}
        if os.path.getsize(resolved) == 0:
            return {"available": False, "status": "UNAVAILABLE",
                    "detail": f"YuNet ONNX weights at '{resolved}' are empty (0 bytes).", "weights_path": resolved}
        return {"available": True, "status": "PRESENT",
                "detail": "weights present; detector not yet initialized", "weights_path": resolved}

    def _load_models(self) -> None:
        """Initializes the only active model: the YuNet face detector."""
        if self.models_loaded:
            return
        self.device, self.is_cuda, self.vram_gb = self._resolve_hardware_device()
        logger.info("[ModelManager] Initializing face detector (%s)...", self.device)
        start_time = time.time()

        # A missing/invalid weight file must never silently degrade to fabricated
        # detections: the status is recorded and the detector stays unavailable.
        yunet_model_obj, yunet_status, yunet_detail, yunet_path = self._load_yunet()

        self.models["secondary_face"] = {
            "model": yunet_model_obj,
            "name": ai_settings.FACE_DETECTOR_MODEL.upper(),
            "status": yunet_status,
            "detail": yunet_detail,
            "weights_path": yunet_path,
            "device": "cpu",
        }

        elapsed = round(time.time() - start_time, 3)
        self.models_loaded = True
        logger.info("==================================================")
        logger.info(f"[ModelManager] Detector: YuNet ({self.models['secondary_face']['status']})")
        logger.info(f"[ModelManager] Device: {self.device}")
        logger.info(f"[ModelManager] Initialization completed in {elapsed}s.")
        logger.info("==================================================")

    def warmup(self) -> None:
        """Loads the detector at startup and runs one inference so the first task
        does not pay the initialization cost.

        A failure is recorded explicitly by the detector status; it is never
        masked by a substitute detector.
        """
        if self.is_warmed_up:
            return

        with self._lock:
            if self.is_warmed_up:
                return

            self._load_models()
            logger.info("[ModelManager] Warming up YuNet detector...")
            warmup_start = time.time()
            try:
                import numpy as np

                handle = self.models.get("secondary_face", {})
                detector = handle.get("model")
                if detector is not None:
                    detector.setInputSize((640, 480))
                    detector.detect(np.zeros((480, 640, 3), dtype=np.uint8))
                self.is_warmed_up = True
                logger.info("[ModelManager] Detector warmup completed in %ss",
                            round(time.time() - warmup_start, 3))
            except Exception as exc:
                logger.warning("[ModelManager] Detector warmup failed: %s", exc)

    def get_model(self, model_key: str) -> Dict[str, Any]:
        self._load_models()
        if model_key not in self.models:
            raise KeyError(f"Model handle '{model_key}' not found in ModelManager. Available: {list(self.models.keys())}")
        return self.models[model_key]

    def get_device_info(self) -> Dict[str, Any]:
        return {
            "device": self.device,
            "is_cuda": self.is_cuda,
            "vram_gb": self.vram_gb,
            "half_precision": self.is_cuda and ai_settings.HALF_PRECISION,
            "is_warmed_up": self.is_warmed_up,
            "models_cached": list(self.models.keys()),
        }

model_manager = ModelManager()
