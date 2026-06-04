"""
Portrait Detection Engine architecture for optimizing subject tracking
"""

import numpy as np
import cv2
from dataclasses import dataclass
from typing import Optional, List
from utils.logger import debug_log

@dataclass
class DetectionResult:
    x_center: float
    y_center: Optional[float] = None
    confidence: float = 1.0


class PortraitDetectionEngine:
    """Base interface for detection engines"""
    def initialize(self):
        """Initialize models, weights, etc."""
        pass
        
    def detect(self, frame: np.ndarray, orig_w: int, orig_h: int) -> Optional[DetectionResult]:
        """Detect subject in frame and return normalized or absolute centers"""
        raise NotImplementedError
        
    def release(self):
        """Free resources"""
        pass
        
    def get_name(self) -> str:
        return self.__class__.__name__


class OpenCVFastEngine(PortraitDetectionEngine):
    """Fallback OpenCV Haar Cascade Engine (Fast but less accurate)"""
    def __init__(self):
        self.face_cascade = None
        
    def initialize(self):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
    def detect(self, frame: np.ndarray, orig_w: int, orig_h: int) -> Optional[DetectionResult]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(50, 50))
        
        if len(faces) > 0:
            largest = max(faces, key=lambda f: f[2] * f[3])
            x_center = largest[0] + largest[2] / 2.0
            y_center = largest[1] + largest[3] / 2.0
            return DetectionResult(x_center=x_center, y_center=y_center)
        return None


class MediaPipeQualityEngine(PortraitDetectionEngine):
    """MediaPipe Face Mesh Engine (Quality but CPU intensive)"""
    def __init__(self):
        self.mp_face_mesh = None
        self.face_mesh = None
        
    def initialize(self):
        import mediapipe as mp
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
    def detect(self, frame: np.ndarray, orig_w: int, orig_h: int) -> Optional[DetectionResult]:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]
            nose = landmarks.landmark[1]
            return DetectionResult(x_center=nose.x * orig_w, y_center=nose.y * orig_h)
        return None
        
    def release(self):
        if self.face_mesh:
            self.face_mesh.close()


class YoloFastEngine(PortraitDetectionEngine):
    """YOLOv8 Engine (Extremely fast if GPU available, accurate)"""
    def __init__(self, model_path: str = "", allow_download: bool = False):
        self.model = None
        self.model_path = model_path
        self.allow_download = allow_download
        
    def initialize(self):
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError("YOLO requires the 'ultralytics' package. Install it with: pip install ultralytics")
            
        import torch
        from pathlib import Path

        # Avoid implicit model downloads during app startup or normal processing.
        # Users can opt in by providing a model path or allowing the lightweight
        # default model download in performance settings.
        selected_model = self.model_path or ('yolov8n.pt' if self.allow_download else '')
        if not selected_model:
            raise FileNotFoundError("YOLO model path is not configured")
        if selected_model != 'yolov8n.pt' and not Path(selected_model).exists():
            raise FileNotFoundError(f"YOLO model not found: {selected_model}")

        self.model = YOLO(selected_model)
        
        # Auto-select device
        if torch.cuda.is_available():
            self.model.to('cuda')
            debug_log("[Portrait] YOLO using CUDA (GPU)")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.model.to('mps')
            debug_log("[Portrait] YOLO using MPS (Mac GPU)")
        else:
            debug_log("[Portrait] YOLO using CPU (No CUDA/MPS detected)")
            
    def detect(self, frame: np.ndarray, orig_w: int, orig_h: int) -> Optional[DetectionResult]:
        import torch
        # Class 0 is 'person' in COCO dataset
        results = self.model.predict(frame, classes=[0], verbose=False, conf=0.3)
        
        if len(results) > 0 and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            # Find largest person box
            largest_idx = torch.argmax(boxes.conf) if hasattr(boxes.conf, 'shape') and len(boxes.conf.shape) > 0 else 0
            
            box = boxes[largest_idx].xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = box
            
            x_center = (x1 + x2) / 2.0
            y_center = (y1 + y2) / 2.0
            
            return DetectionResult(x_center=x_center, y_center=y_center)
        return None


class EngineManager:
    """Manages the selection and execution of detection engines"""
    def __init__(self, mode: str = "hybrid_auto", settings: Optional[dict] = None):
        self.mode = mode
        self.engine: Optional[PortraitDetectionEngine] = None
        self.profile = "balanced"
        self.settings = settings or {}
        
    def get_detection_interval(self) -> int:
        configured_interval = self.settings.get("detection_interval")
        if isinstance(configured_interval, int) and configured_interval > 0:
            return configured_interval

        if self.profile == "quality":
            return 5
        elif self.profile == "fast":
            return 30
        return 10 # balanced
        
    def setup_engine(self, profile: str = "balanced") -> str:
        self.profile = profile
        
        engines_to_try = []
        if self.mode == "yolo_fast":
            engines_to_try = [YoloFastEngine]
        elif self.mode == "mediapipe_quality":
            engines_to_try = [MediaPipeQualityEngine]
        elif self.mode == "opencv_fast":
            engines_to_try = [OpenCVFastEngine]
        else:
            # hybrid_auto tries YOLO first, then MediaPipe, then OpenCV
            engines_to_try = [YoloFastEngine, MediaPipeQualityEngine, OpenCVFastEngine]
            
        for engine_cls in engines_to_try:
            try:
                if engine_cls is YoloFastEngine:
                    engine = engine_cls(
                        model_path=self.settings.get("yolo_model_path", ""),
                        allow_download=bool(self.settings.get("allow_yolo_download", False))
                    )
                else:
                    engine = engine_cls()
                engine.initialize()
                self.engine = engine
                debug_log(f"[Portrait] Initialized engine: {self.engine.get_name()}")
                return self.engine.get_name()
            except Exception as e:
                debug_log(f"[Portrait] Engine {engine_cls.__name__} failed to initialize: {e}")
                
        raise RuntimeError("No suitable portrait detection engine could be initialized.")
        
    def process_pass_1(self, cap, total_frames: int, orig_w: int, orig_h: int, crop_w: int, is_cancelled_callback, progress_callback) -> List[int]:
        """Runs the interval-based detection pass and returns an array of crop positions per frame"""
        crop_positions = []
        current_target = orig_w / 2
        
        frame_count = 0
        detected_frames = 0
        skipped_frames = 0
        last_log_time = 0
        import time
        
        interval = self.get_detection_interval()
        debug_log(f"[Portrait] Starting Pass 1 with {self.engine.get_name()}, detecting every {interval} frames")
        
        # Exponential Moving Average for smoothing
        alpha = 0.2 
        
        while True:
            if is_cancelled_callback():
                cap.release()
                if self.engine:
                    self.engine.release()
                raise Exception("Cancelled by user")
                
            ret, frame = cap.read()
            if not ret:
                break
                
            # Only detect every N frames
            if frame_count % interval == 0:
                result = self.engine.detect(frame, orig_w, orig_h)
                detected_frames += 1
                if result:
                    # Smooth tracking using EMA
                    current_target = (alpha * result.x_center) + ((1 - alpha) * current_target)
            else:
                skipped_frames += 1
            
            crop_x = int(current_target - crop_w / 2)
            crop_x = max(0, min(crop_x, orig_w - crop_w))
            crop_positions.append(crop_x)
            
            frame_count += 1
            
            current_time = time.time()
            if frame_count % 30 == 0 or (current_time - last_log_time) > 2:
                progress = (frame_count / max(1, total_frames)) * 0.4
                progress_callback(progress)
                last_log_time = current_time
                
        if self.engine:
            self.engine.release()
            
        debug_log(f"[Portrait] Pass 1 complete. Total: {frame_count}, Detected: {detected_frames}, Skipped: {skipped_frames}")
        return crop_positions
