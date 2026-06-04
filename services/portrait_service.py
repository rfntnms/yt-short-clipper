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
    width: Optional[float] = None
    height: Optional[float] = None


class PortraitDetectionEngine:
    """Base interface for detection engines"""
    def initialize(self):
        """Initialize models, weights, etc."""
        pass
        
    def detect(self, frame: np.ndarray, orig_w: int, orig_h: int) -> Optional[DetectionResult]:
        """Detect subject in frame and return normalized or absolute centers"""
        raise NotImplementedError

    def detect_all(self, frame: np.ndarray, orig_w: int, orig_h: int) -> List[DetectionResult]:
        result = self.detect(frame, orig_w, orig_h)
        return [result] if result else []
        
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
        
    def detect_all(self, frame: np.ndarray, orig_w: int, orig_h: int) -> List[DetectionResult]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(50, 50))

        detections = []
        for x, y, w, h in faces:
            detections.append(DetectionResult(
                x_center=x + w / 2.0,
                y_center=y + h / 2.0,
                confidence=float(w * h),
                width=float(w),
                height=float(h),
            ))
        return detections

    def detect(self, frame: np.ndarray, orig_w: int, orig_h: int) -> Optional[DetectionResult]:
        detections = self.detect_all(frame, orig_w, orig_h)
        if detections:
            return max(detections, key=lambda d: d.confidence)
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
            max_num_faces=3,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
    def detect_all(self, frame: np.ndarray, orig_w: int, orig_h: int) -> List[DetectionResult]:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        detections = []
        if results.multi_face_landmarks:
            for landmarks in results.multi_face_landmarks:
                nose = landmarks.landmark[1]
                xs = [point.x * orig_w for point in landmarks.landmark]
                ys = [point.y * orig_h for point in landmarks.landmark]
                detections.append(DetectionResult(
                    x_center=nose.x * orig_w,
                    y_center=nose.y * orig_h,
                    confidence=1.0,
                    width=max(xs) - min(xs),
                    height=max(ys) - min(ys),
                ))
        return detections

    def detect(self, frame: np.ndarray, orig_w: int, orig_h: int) -> Optional[DetectionResult]:
        detections = self.detect_all(frame, orig_w, orig_h)
        return detections[0] if detections else None
        
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
            
    def detect_all(self, frame: np.ndarray, orig_w: int, orig_h: int) -> List[DetectionResult]:
        # Class 0 is 'person' in COCO dataset
        results = self.model.predict(frame, classes=[0], verbose=False, conf=0.3)

        detections = []
        if len(results) > 0 and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            confidences = boxes.conf.cpu().numpy() if hasattr(boxes.conf, "cpu") else []
            for idx, box_tensor in enumerate(boxes.xyxy):
                box = box_tensor.cpu().numpy()
                x1, y1, x2, y2 = box
                confidence = float(confidences[idx]) if idx < len(confidences) else 1.0
                detections.append(DetectionResult(
                    x_center=(x1 + x2) / 2.0,
                    y_center=(y1 + y2) / 2.0,
                    confidence=confidence,
                    width=float(x2 - x1),
                    height=float(y2 - y1),
                ))
        return detections

    def detect(self, frame: np.ndarray, orig_w: int, orig_h: int) -> Optional[DetectionResult]:
        detections = self.detect_all(frame, orig_w, orig_h)
        if detections:
            return max(detections, key=lambda d: d.confidence)
        return None


class EngineManager:
    """Manages the selection and execution of detection engines"""
    def __init__(self, mode: str = "hybrid_auto", settings: Optional[dict] = None):
        self.mode = mode
        self.engine: Optional[PortraitDetectionEngine] = None
        self.profile = "balanced"
        self.settings = settings or {}
        self.framing_mode = self.settings.get("speaker_framing_mode", "center_speaker")
        if self.framing_mode not in ("center_speaker", "active_speaker"):
            self.framing_mode = "center_speaker"
        
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

    def _candidate_size_score(self, result: DetectionResult, orig_w: int, orig_h: int) -> float:
        if not result.width or not result.height:
            return 0.0
        frame_area = max(1.0, float(orig_w * orig_h))
        return min(1.0, float(result.width * result.height) / frame_area * 12.0)

    def _candidate_center_score(self, result: DetectionResult, orig_w: int) -> float:
        return max(0.0, 1.0 - abs(result.x_center - orig_w / 2) / max(1.0, orig_w / 2))

    def _candidate_continuity_score(self, result: DetectionResult, locked_x: float, orig_w: int) -> float:
        return max(0.0, 1.0 - abs(result.x_center - locked_x) / max(1.0, orig_w / 3))

    def _initial_candidate_score(self, result: DetectionResult, orig_w: int, orig_h: int) -> float:
        center_score = self._candidate_center_score(result, orig_w)
        size_score = self._candidate_size_score(result, orig_w, orig_h)
        confidence = min(1.0, float(result.confidence or 0.0))
        if self.framing_mode == "active_speaker":
            return (confidence * 0.45) + (size_score * 0.35) + (center_score * 0.20)
        return (center_score * 0.55) + (size_score * 0.25) + (confidence * 0.20)

    def _locked_candidate_score(self, result: DetectionResult, locked_x: float, orig_w: int, orig_h: int) -> float:
        continuity_score = self._candidate_continuity_score(result, locked_x, orig_w)
        center_score = self._candidate_center_score(result, orig_w)
        size_score = self._candidate_size_score(result, orig_w, orig_h)
        confidence = min(1.0, float(result.confidence or 0.0))
        if self.framing_mode == "active_speaker":
            return (continuity_score * 0.45) + (confidence * 0.30) + (size_score * 0.20) + (center_score * 0.05)
        return (continuity_score * 0.50) + (center_score * 0.30) + (size_score * 0.15) + (confidence * 0.05)

    def _select_candidate(
        self,
        detections: List[DetectionResult],
        locked_x: Optional[float],
        pending_x: Optional[float],
        stable_switch_count: int,
        orig_w: int,
        orig_h: int,
    ):
        if not detections:
            return None, pending_x, 0, False

        if locked_x is None:
            selected = max(detections, key=lambda d: self._initial_candidate_score(d, orig_w, orig_h))
            return selected, None, 0, False

        switch_distance = orig_w * 0.18
        near_lock = [d for d in detections if abs(d.x_center - locked_x) <= switch_distance]
        far_from_lock = [d for d in detections if abs(d.x_center - locked_x) > switch_distance]

        selected = max(near_lock, key=lambda d: self._locked_candidate_score(d, locked_x, orig_w, orig_h)) if near_lock else None

        if not far_from_lock:
            return selected or DetectionResult(x_center=locked_x), None, 0, False

        challenger = max(far_from_lock, key=lambda d: self._initial_candidate_score(d, orig_w, orig_h))
        current_score = self._initial_candidate_score(selected, orig_w, orig_h) if selected else 0.0
        challenger_score = self._initial_candidate_score(challenger, orig_w, orig_h)
        challenge_margin = 0.08 if self.framing_mode == "center_speaker" else 0.03

        if challenger_score <= current_score + challenge_margin:
            return selected or DetectionResult(x_center=locked_x), None, 0, False

        if pending_x is not None and abs(challenger.x_center - pending_x) <= switch_distance:
            stable_switch_count += 1
        else:
            pending_x = challenger.x_center
            stable_switch_count = 1

        required_switches = 3 if self.framing_mode == "center_speaker" else 2
        if stable_switch_count >= required_switches:
            return challenger, None, 0, True

        hold = selected or DetectionResult(x_center=locked_x, y_center=challenger.y_center, confidence=challenger.confidence)
        return hold, pending_x, stable_switch_count, False
        
    def process_pass_1(self, cap, total_frames: int, orig_w: int, orig_h: int, crop_w: int, is_cancelled_callback, progress_callback) -> List[int]:
        """Runs the interval-based detection pass and returns an array of crop positions per frame"""
        crop_positions = []
        current_target = orig_w / 2
        locked_x = None
        pending_x = None
        stable_switch_count = 0
        missed_detections = 0
        target_switches = 0
        
        frame_count = 0
        detected_frames = 0
        skipped_frames = 0
        last_log_time = 0
        import time
        
        interval = self.get_detection_interval()
        debug_log(
            f"[Portrait] Starting Pass 1 with {self.engine.get_name()}, "
            f"detecting every {interval} frames, framing={self.framing_mode}"
        )
        
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
                detections = self.engine.detect_all(frame, orig_w, orig_h)
                detected_frames += 1
                result, pending_x, stable_switch_count, switched = self._select_candidate(
                    detections,
                    locked_x,
                    pending_x,
                    stable_switch_count,
                    orig_w,
                    orig_h,
                )
                if result:
                    missed_detections = 0
                    if locked_x is None or switched:
                        target_switches += 1 if locked_x is not None else 0
                        locked_x = result.x_center
                    else:
                        locked_x = result.x_center
                    # Smooth tracking using EMA
                    current_target = (alpha * result.x_center) + ((1 - alpha) * current_target)
                else:
                    missed_detections += 1
                    if missed_detections > 5:
                        current_target = (0.05 * (orig_w / 2)) + (0.95 * current_target)
                        locked_x = current_target
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
            
        debug_log(
            f"[Portrait] Pass 1 complete. Total: {frame_count}, Detected: {detected_frames}, "
            f"Skipped: {skipped_frames}, Switches: {target_switches}, Missed: {missed_detections}"
        )
        return crop_positions
