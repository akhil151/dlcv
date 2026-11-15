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
    - Load the license-plate detector (RT-DETR ONNX) once, lazily, only when a
      job actually requests plates.
    - Report explicit LOADED / UNAVAILABLE / ERROR status for each.

    The bundled YOLO11m weights and the OSNet Re-ID model are deliberately not
    loaded: they are not used by any active stage (YOLO has no face/plate
    semantics and Re-ID is not wired into association). Loading them only added
    startup cost and memory, and made readiness reports misleading.
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
            # Plate detector state is separate so a face-only job never pays
            # the 80 MB load / initialization cost.
            self.plate_model: Optional[Any] = None
            self.plate_status: str = "UNAVAILABLE"
            self.plate_detail: str = "not initialized"
            self.plate_weights_path: Optional[str] = None
            self.plate_output_names: Tuple[str, ...] = ()
            # Identity-document detector state is separate as well: a face/plate
            # job never pays the 83 MB DocAligner load cost.
            self.document_model: Optional[Any] = None
            self.document_status: str = "UNAVAILABLE"
            self.document_detail: str = "not initialized"
            self.document_weights_path: Optional[str] = None
            self.document_output_names: Tuple[str, ...] = ()
            # Laptop-screen (device proposal) detector state is separate so a
            # face/plate/document job never pays this ~83 MB load cost. The
            # detector uses onnxruntime; the graph is the RT-DETR R18 COCO
            # export pinned by scripts/download_laptop_screen_model.py.
            self.laptop_model: Optional[Any] = None
            self.laptop_status: str = "UNAVAILABLE"
            self.laptop_detail: str = "not initialized"
            self.laptop_weights_path: Optional[str] = None
            self.laptop_output_names: Tuple[str, ...] = ()
            self.laptop_load_seconds: Optional[float] = None
            # Phone-screen (device proposal) detector state is separate so a
            # face/plate/document/laptop job never pays this ~83 MB load cost. The
            # detector uses onnxruntime; the graph is the RT-DETR R18 COCO
            # export pinned by scripts/download_phone_screen_model.py.
            self.phone_model: Optional[Any] = None
            self.phone_status: str = "UNAVAILABLE"
            self.phone_detail: str = "not initialized"
            self.phone_weights_path: Optional[str] = None
            self.phone_output_names: Tuple[str, ...] = ()
            self.phone_load_seconds: Optional[float] = None

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

    def resolve_plate_path(self) -> str:
        """Resolves the plate detector weight path irrespective of cwd."""
        configured = ai_settings.PLATE_MODEL_PATH
        if os.path.isabs(configured):
            return configured
        project_candidate = os.path.join(PROJECT_ROOT, configured)
        if os.path.isfile(project_candidate):
            return project_candidate
        cwd_candidate = os.path.abspath(configured)
        if os.path.isfile(cwd_candidate):
            return cwd_candidate
        return project_candidate

    def _load_plate_detector(self) -> Tuple[Optional[Any], str, str, str, Tuple[str, ...]]:
        """Initializes the plate ONNX graph.

        Returns (model, status, detail, resolved_path, output_names). Failures
        are explicit and never trigger a substitute detector.
        """
        resolved = self.resolve_plate_path()
        try:
            import cv2
        except Exception as exc:
            return None, "UNAVAILABLE", f"OpenCV is not importable: {exc}", resolved, ()

        if not os.path.isfile(resolved):
            return (None, "UNAVAILABLE",
                    f"License-plate detector weights not found at '{resolved}'. "
                    "Run 'python scripts/download_plate_model.py' or set PLATE_MODEL_PATH.",
                    resolved, ())
        if os.path.getsize(resolved) == 0:
            return None, "UNAVAILABLE", f"License-plate weights at '{resolved}' are empty (0 bytes).", resolved, ()

        try:
            logger.info(f"[ModelManager] Loading RT-DETR plate detector from '{resolved}'...")
            model = cv2.dnn.readNetFromONNX(resolved)
            if model is None:
                return None, "ERROR", "cv2.dnn.readNetFromONNX returned no network.", resolved, ()
            output_names = tuple(model.getUnconnectedOutLayersNames())
            required = {"logits", "pred_boxes"}
            if not required.issubset(set(output_names)):
                return (None, "ERROR",
                        f"plate detector outputs {list(output_names)} do not match the expected "
                        f"{sorted(required)} graph",
                        resolved, output_names)
            logger.info("[ModelManager] Successfully Loaded RT-DETR plate detector")
            return model, "LOADED", "ok", resolved, output_names
        except Exception as exc:
            return None, "ERROR", f"Plate detector initialization failed for '{resolved}': {exc}", resolved, ()

    def ensure_plate_detector(self) -> Dict[str, Any]:
        """Loads the plate detector once and returns its handle.

        Called only when a job requests the license_plates category.
        """
        with self._lock:
            if self.plate_status == "LOADED" and self.plate_model is not None:
                return self._plate_handle()
            if self.plate_status == "ERROR":
                return self._plate_handle()
            model, status, detail, path, names = self._load_plate_detector()
            self.plate_model = model
            self.plate_status = status
            self.plate_detail = detail
            self.plate_weights_path = path
            self.plate_output_names = names
            return self._plate_handle()

    def _plate_handle(self) -> Dict[str, Any]:
        return {
            "model": self.plate_model,
            "name": ai_settings.PLATE_DETECTOR_MODEL.upper(),
            "status": self.plate_status,
            "detail": self.plate_detail,
            "weights_path": self.plate_weights_path,
            "output_names": self.plate_output_names,
            "device": "cpu",
        }

    def plate_detector_readiness(self) -> Dict[str, Any]:
        """Cheap snapshot of the plate detector dependency (no loading)."""
        if self.plate_status in ("LOADED", "ERROR"):
            return {
                "available": self.plate_status == "LOADED",
                "status": self.plate_status,
                "detail": self.plate_detail,
                "weights_path": self.plate_weights_path,
            }
        resolved = self.resolve_plate_path()
        if not os.path.isfile(resolved):
            return {"available": False, "status": "UNAVAILABLE",
                    "detail": f"License-plate detector weights not found at '{resolved}'.",
                    "weights_path": resolved}
        if os.path.getsize(resolved) == 0:
            return {"available": False, "status": "UNAVAILABLE",
                    "detail": f"License-plate weights at '{resolved}' are empty (0 bytes).",
                    "weights_path": resolved}
        return {"available": True, "status": "PRESENT",
                "detail": "weights present; detector not yet initialized", "weights_path": resolved}

    # ------------------------------------------------------- identity documents

    def resolve_document_path(self) -> str:
        """Resolves the document detector weight path irrespective of cwd."""
        configured = ai_settings.DOCUMENT_MODEL_PATH
        if os.path.isabs(configured):
            return configured
        project_candidate = os.path.join(PROJECT_ROOT, configured)
        if os.path.isfile(project_candidate):
            return project_candidate
        cwd_candidate = os.path.abspath(configured)
        if os.path.isfile(cwd_candidate):
            return cwd_candidate
        return project_candidate

    def _load_document_detector(self) -> Tuple[Optional[Any], str, str, str, Tuple[str, ...]]:
        """Initializes the DocAligner document-corner ONNX graph.

        Returns (model, status, detail, resolved_path, output_names). Failures
        are explicit and never trigger a substitute detector: a document job
        either runs this validated graph or is refused.
        """
        resolved = self.resolve_document_path()
        try:
            import cv2
        except Exception as exc:  # pragma: no cover - OpenCV is a hard dependency
            return None, "UNAVAILABLE", f"OpenCV is not importable: {exc}", resolved, ()

        if not os.path.isfile(resolved):
            return (None, "UNAVAILABLE",
                    f"Identity-document detector weights not found at '{resolved}'. "
                    "Run 'python scripts/download_document_model.py' or set DOCUMENT_MODEL_PATH.",
                    resolved, ())
        if os.path.getsize(resolved) == 0:
            return None, "UNAVAILABLE", f"Identity-document weights at '{resolved}' are empty (0 bytes).", resolved, ()

        try:
            logger.info("[ModelManager] Loading DocAligner document detector from '%s'...", resolved)
            model = cv2.dnn.readNetFromONNX(resolved)
            if model is None:
                return None, "ERROR", "cv2.dnn.readNetFromONNX returned no network.", resolved, ()
            output_names = tuple(model.getUnconnectedOutLayersNames())
            if "heatmap" not in output_names:
                return (None, "ERROR",
                        f"document detector outputs {list(output_names)} do not match the expected "
                        "['heatmap'] graph",
                        resolved, output_names)
            # Constructor-level validation: one deterministic zero-frame forward
            # proves the graph runs in this OpenCV build and that the corner
            # heatmap shape is the one the decoder expects.
            import numpy as np
            size = ai_settings.DOCUMENT_DETECTOR_INPUT_SIZE
            expected = ai_settings.DOCUMENT_HEATMAP_OUTPUT_SIZE
            model.setInput(np.zeros((1, 3, size, size), dtype=np.float32))
            probe = model.forward()
            if probe.shape != (1, 4, expected, expected):
                return (None, "ERROR",
                        f"document detector heatmap shape {probe.shape} != expected "
                        f"(1, 4, {expected}, {expected})",
                        resolved, output_names)
            logger.info("[ModelManager] Successfully Loaded DocAligner document detector")
            return model, "LOADED", "ok", resolved, output_names
        except Exception as exc:
            return None, "ERROR", f"Document detector initialization failed for '{resolved}': {exc}", resolved, ()

    def ensure_document_detector(self) -> Dict[str, Any]:
        """Loads the document detector once and returns its handle.

        Called only when a job requests id_cards and/or passports.
        """
        with self._lock:
            if self.document_status in ("LOADED", "ERROR"):
                return self._document_handle()
            model, status, detail, path, names = self._load_document_detector()
            self.document_model = model
            self.document_status = status
            self.document_detail = detail
            self.document_weights_path = path
            self.document_output_names = names
            return self._document_handle()

    def _document_handle(self) -> Dict[str, Any]:
        return {
            "model": self.document_model,
            "name": ai_settings.DOCUMENT_DETECTOR_MODEL,
            "status": self.document_status,
            "detail": self.document_detail,
            "weights_path": self.document_weights_path,
            "output_names": self.document_output_names,
            "device": "cpu",
        }

    def document_detector_readiness(self) -> Dict[str, Any]:
        """Cheap snapshot of the document detector dependency (no loading)."""
        if self.document_status in ("LOADED", "ERROR"):
            return {
                "available": self.document_status == "LOADED",
                "status": self.document_status,
                "detail": self.document_detail,
                "weights_path": self.document_weights_path,
            }
        resolved = self.resolve_document_path()
        if not os.path.isfile(resolved):
            return {"available": False, "status": "UNAVAILABLE",
                    "detail": f"Identity-document detector weights not found at '{resolved}'.",
                    "weights_path": resolved}
        if os.path.getsize(resolved) == 0:
            return {"available": False, "status": "UNAVAILABLE",
                    "detail": f"Identity-document weights at '{resolved}' are empty (0 bytes).",
                    "weights_path": resolved}
        return {"available": True, "status": "PRESENT",
                "detail": "weights present; detector not yet initialized", "weights_path": resolved}

    # -------------------------------------------------------- laptop screens

    def resolve_laptop_path(self) -> str:
        """Resolves the laptop-screen detector weight path irrespective of cwd."""
        configured = ai_settings.LAPTOP_MODEL_PATH
        if os.path.isabs(configured):
            return configured
        project_candidate = os.path.join(PROJECT_ROOT, configured)
        if os.path.isfile(project_candidate):
            return project_candidate
        cwd_candidate = os.path.abspath(configured)
        if os.path.isfile(cwd_candidate):
            return cwd_candidate
        return project_candidate

    def _load_laptop_detector(self) -> Tuple[Optional[Any], str, str, str, Tuple[str, ...]]:
        """Initializes the RT-DETR R18 COCO ONNX graph with onnxruntime.

        Returns (session, status, detail, resolved_path, output_names). Failures
        are explicit and never trigger a substitute detector: a laptop-screen
        job either runs this validated graph or is refused before the task is
        created.
        """
        resolved = self.resolve_laptop_path()
        try:
            import onnxruntime  # noqa: F401
        except Exception as exc:
            return (None, "UNAVAILABLE",
                    "onnxruntime is not installed; run 'pip install -r backend/requirements.txt'.",
                    resolved, ())

        if not os.path.isfile(resolved):
            return (None, "UNAVAILABLE",
                    f"Laptop-screen detector weights not found at '{resolved}'. "
                    "Run 'python scripts/download_laptop_screen_model.py' or set LAPTOP_MODEL_PATH.",
                    resolved, ())
        if os.path.getsize(resolved) == 0:
            return None, "UNAVAILABLE", f"Laptop-screen weights at '{resolved}' are empty (0 bytes).", resolved, ()

        try:
            import numpy as np
            import onnxruntime as ort

            logger.info("[ModelManager] Loading RT-DETR laptop detector from '%s'...", resolved)
            started = time.time()
            options = ort.SessionOptions()
            options.log_severity_level = 3
            session = ort.InferenceSession(
                resolved, sess_options=options, providers=["CPUExecutionProvider"])
            output_names = tuple(output.name for output in session.get_outputs())
            if not {"logits", "pred_boxes"}.issubset(set(output_names)):
                return (None, "ERROR",
                        f"laptop detector outputs {list(output_names)} do not match the expected "
                        "['logits', 'pred_boxes'] graph",
                        resolved, output_names)
            size = ai_settings.LAPTOP_DETECTOR_INPUT_SIZE
            probe = session.run(None, {
                session.get_inputs()[0].name: np.zeros((1, 3, size, size), dtype=np.float32)})
            by_name = dict(zip(output_names, probe))
            logits = np.asarray(by_name["logits"])
            boxes = np.asarray(by_name["pred_boxes"])
            if logits.ndim != 3 or boxes.ndim != 3 or logits.shape[:2] != boxes.shape[:2]:
                return (None, "ERROR",
                        f"laptop detector output shapes logits={logits.shape} "
                        f"boxes={boxes.shape} are not a COCO detection pair",
                        resolved, output_names)
            self.laptop_load_seconds = round(time.time() - started, 3)
            logger.info("[ModelManager] Successfully Loaded RT-DETR laptop detector")
            return session, "LOADED", "ok", resolved, output_names
        except Exception as exc:
            return None, "ERROR", f"Laptop detector initialization failed for '{resolved}': {exc}", resolved, ()

    def ensure_laptop_detector(self) -> Dict[str, Any]:
        """Loads the laptop detector once and returns its handle.

        Called only when a job requests the laptop_screens category.
        """
        with self._lock:
            if self.laptop_status in ("LOADED", "ERROR"):
                return self._laptop_handle()
            model, status, detail, path, names = self._load_laptop_detector()
            self.laptop_model = model
            self.laptop_status = status
            self.laptop_detail = detail
            self.laptop_weights_path = path
            self.laptop_output_names = names
            return self._laptop_handle()

    def _laptop_handle(self) -> Dict[str, Any]:
        return {
            "model": self.laptop_model,
            "name": ai_settings.LAPTOP_DETECTOR_MODEL,
            "status": self.laptop_status,
            "detail": self.laptop_detail,
            "weights_path": self.laptop_weights_path,
            "output_names": self.laptop_output_names,
            "load_seconds": self.laptop_load_seconds,
            "device": "cpu",
        }

    def laptop_detector_readiness(self) -> Dict[str, Any]:
        """Cheap snapshot of the laptop detector dependency (no loading)."""
        if self.laptop_status in ("LOADED", "ERROR"):
            return {
                "available": self.laptop_status == "LOADED",
                "status": self.laptop_status,
                "detail": self.laptop_detail,
                "weights_path": self.laptop_weights_path,
            }
        resolved = self.resolve_laptop_path()
        if not os.path.isfile(resolved):
            return {"available": False, "status": "UNAVAILABLE",
                    "detail": f"Laptop-screen detector weights not found at '{resolved}'.",
                    "weights_path": resolved}
        if os.path.getsize(resolved) == 0:
            return {"available": False, "status": "UNAVAILABLE",
                    "detail": f"Laptop-screen weights at '{resolved}' are empty (0 bytes).",
                    "weights_path": resolved}
        try:
            import onnxruntime  # noqa: F401
        except Exception:
            return {"available": False, "status": "UNAVAILABLE",
                    "detail": "onnxruntime is not installed in this environment.",
                    "weights_path": resolved}
        return {"available": True, "status": "PRESENT",
                "detail": "weights present; detector not yet initialized", "weights_path": resolved}

    # --------------------------------------------------------- phone screens

    def resolve_phone_path(self) -> str:
        """Resolves the phone-screen detector weight path irrespective of cwd."""
        configured = ai_settings.PHONE_MODEL_PATH
        if os.path.isabs(configured):
            return configured
        project_candidate = os.path.join(PROJECT_ROOT, configured)
        if os.path.isfile(project_candidate):
            return project_candidate
        cwd_candidate = os.path.abspath(configured)
        if os.path.isfile(cwd_candidate):
            return cwd_candidate
        return project_candidate

    def _load_phone_detector(self) -> Tuple[Optional[Any], str, str, str, Tuple[str, ...]]:
        """Initializes the RT-DETR R18 COCO ONNX graph with onnxruntime for phones.

        Returns (session, status, detail, resolved_path, output_names). Failures
        are explicit and never trigger a substitute detector: a phone-screen
        job either runs this validated graph or is refused before the task is
        created.
        """
        resolved = self.resolve_phone_path()
        try:
            import onnxruntime  # noqa: F401
        except Exception as exc:
            return (None, "UNAVAILABLE",
                    "onnxruntime is not installed; run 'pip install -r backend/requirements.txt'.",
                    resolved, ())

        if not os.path.isfile(resolved):
            return (None, "UNAVAILABLE",
                    f"Phone-screen detector weights not found at '{resolved}'. "
                    "Run 'python scripts/download_phone_screen_model.py' or set PHONE_MODEL_PATH.",
                    resolved, ())
        if os.path.getsize(resolved) == 0:
            return None, "UNAVAILABLE", f"Phone-screen weights at '{resolved}' are empty (0 bytes).", resolved, ()

        try:
            import numpy as np
            import onnxruntime as ort

            logger.info("[ModelManager] Loading RT-DETR phone detector from '%s'...", resolved)
            started = time.time()
            options = ort.SessionOptions()
            options.log_severity_level = 3
            session = ort.InferenceSession(
                resolved, sess_options=options, providers=["CPUExecutionProvider"])
            output_names = tuple(output.name for output in session.get_outputs())
            if not {"logits", "pred_boxes"}.issubset(set(output_names)):
                return (None, "ERROR",
                        f"phone detector outputs {list(output_names)} do not match the expected "
                        "['logits', 'pred_boxes'] graph",
                        resolved, output_names)
            size = ai_settings.PHONE_DETECTOR_INPUT_SIZE
            probe = session.run(None, {
                session.get_inputs()[0].name: np.zeros((1, 3, size, size), dtype=np.float32)})
            by_name = dict(zip(output_names, probe))
            logits = np.asarray(by_name["logits"])
            boxes = np.asarray(by_name["pred_boxes"])
            if logits.ndim != 3 or boxes.ndim != 3 or logits.shape[:2] != boxes.shape[:2]:
                return (None, "ERROR",
                        f"phone detector output shapes logits={logits.shape} "
                        f"boxes={boxes.shape} are not a COCO detection pair",
                        resolved, output_names)
            self.phone_load_seconds = round(time.time() - started, 3)
            logger.info("[ModelManager] Successfully Loaded RT-DETR phone detector")
            return session, "LOADED", "ok", resolved, output_names
        except Exception as exc:
            return None, "ERROR", f"Phone detector initialization failed for '{resolved}': {exc}", resolved, ()

    def ensure_phone_detector(self) -> Dict[str, Any]:
        """Loads the phone detector once and returns its handle.

        Called only when a job requests the phone_screens category.
        """
        with self._lock:
            if self.phone_status in ("LOADED", "ERROR"):
                return self._phone_handle()
            model, status, detail, path, names = self._load_phone_detector()
            self.phone_model = model
            self.phone_status = status
            self.phone_detail = detail
            self.phone_weights_path = path
            self.phone_output_names = names
            return self._phone_handle()

    def _phone_handle(self) -> Dict[str, Any]:
        return {
            "model": self.phone_model,
            "name": ai_settings.PHONE_DETECTOR_MODEL,
            "status": self.phone_status,
            "detail": self.phone_detail,
            "weights_path": self.phone_weights_path,
            "output_names": self.phone_output_names,
            "load_seconds": self.phone_load_seconds,
            "device": "cpu",
        }

    def phone_detector_readiness(self) -> Dict[str, Any]:
        """Cheap snapshot of the phone detector dependency (no loading)."""
        if self.phone_status in ("LOADED", "ERROR"):
            return {
                "available": self.phone_status == "LOADED",
                "status": self.phone_status,
                "detail": self.phone_detail,
                "weights_path": self.phone_weights_path,
            }
        resolved = self.resolve_phone_path()
        if not os.path.isfile(resolved):
            return {"available": False, "status": "UNAVAILABLE",
                    "detail": f"Phone-screen detector weights not found at '{resolved}'.",
                    "weights_path": resolved}
        if os.path.getsize(resolved) == 0:
            return {"available": False, "status": "UNAVAILABLE",
                    "detail": f"Phone-screen weights at '{resolved}' are empty (0 bytes).",
                    "weights_path": resolved}
        try:
            import onnxruntime  # noqa: F401
        except Exception:
            return {"available": False, "status": "UNAVAILABLE",
                    "detail": "onnxruntime is not installed in this environment.",
                    "weights_path": resolved}
        return {"available": True, "status": "PRESENT",
                "detail": "weights present; detector not yet initialized", "weights_path": resolved}

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
