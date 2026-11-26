from __future__ import annotations
from abc import ABC, abstractmethod
import cv2
import numpy as np
from .config import ValidationConfig
from .models import HandDetection

class HandDetector(ABC):
    @abstractmethod
    def detect(self, frame: np.ndarray) -> list[HandDetection]: ...

class MediaPipeHandDetector(HandDetector):
    """MediaPipe hand-landmark adapter; initialized once per video."""
    def __init__(self, config: ValidationConfig, max_hands: int = 4) -> None:
        try:
            import mediapipe as mp
        except ImportError as exc:
            raise RuntimeError("MediaPipe is missing. Install requirements.txt.") from exc
        if not hasattr(mp, "solutions"):
            raise RuntimeError("This MediaPipe installation does not expose the Solutions API. Install mediapipe==0.10.21.")
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False, max_num_hands=max_hands, model_complexity=1,
            min_detection_confidence=config.hand_detection_confidence,
            min_tracking_confidence=config.hand_tracking_confidence)
    def detect(self, frame: np.ndarray) -> list[HandDetection]:
        height, width = frame.shape[:2]
        result = self._hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        detections = []
        for index, item in enumerate(result.multi_hand_landmarks or []):
            points = [(float(p.x), float(p.y)) for p in item.landmark]
            xs, ys = [p[0] for p in points], [p[1] for p in points]
            bbox = (max(0,int(min(xs)*width)), max(0,int(min(ys)*height)),
                    min(width-1,int(max(xs)*width)), min(height-1,int(max(ys)*height)))
            label, score = "Unknown", 0.0
            handedness = result.multi_handedness or []
            if index < len(handedness) and handedness[index].classification:
                label = handedness[index].classification[0].label
                score = float(handedness[index].classification[0].score)
            detections.append(HandDetection(label, score, points, bbox))
        return detections
    def close(self) -> None:
        self._hands.close()

class YoloSceneDetector:
    """One YOLO inference per sampled frame for people and configurable COCO objects."""
    DEFAULT_UNWANTED = {"cell phone","remote","scissors","bottle","cup","book","laptop","keyboard","mouse","backpack","handbag","suitcase","umbrella"}
    def __init__(self, person_confidence: float, object_confidence: float, unwanted_names: list[str] | None):
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("Ultralytics is missing. Install requirements.txt.") from exc
        self.model = YOLO("yolo11n.pt")
        self.person_confidence = person_confidence
        self.object_confidence = object_confidence
        self.unwanted = {name.strip().lower() for name in (unwanted_names or sorted(self.DEFAULT_UNWANTED)) if name.strip()}
    def detect(self, frame: np.ndarray) -> tuple[int, list[str], list[tuple[str, tuple[int,int,int,int]]]]:
        results = self.model.predict(source=frame, conf=min(self.person_confidence,self.object_confidence),
                                     verbose=False, imgsz=640)
        people, found, boxes = 0, set(), []
        if not results or results[0].boxes is None: return people, [], boxes
        names = results[0].names
        for box in results[0].boxes:
            class_id = int(box.cls[0])
            name = str(names[class_id]).lower()
            confidence = float(box.conf[0])
            if name == "person" and confidence >= self.person_confidence:
                people += 1
            elif name in self.unwanted and confidence >= self.object_confidence:
                found.add(name)
                coords = tuple(int(v) for v in box.xyxy[0].tolist())
                boxes.append((name, coords))
        return people, sorted(found), boxes

class YoloPersonDetector:
    """Compatibility adapter for callers needing only a person count."""
    def __init__(self, confidence: float = 0.45, model_name: str = "yolo11n.pt"):
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("Ultralytics is missing. Install requirements.txt.") from exc
        self.model = YOLO(model_name); self.confidence = confidence
    def count_people(self, frame: np.ndarray) -> int:
        result = self.model.predict(source=frame, classes=[0], conf=self.confidence, verbose=False)
        return len(result[0].boxes) if result and result[0].boxes is not None else 0
