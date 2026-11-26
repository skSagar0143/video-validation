from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import cv2
import numpy as np

from .config import ValidationConfig
from .models import HandDetection


class HandDetector(ABC):
    @abstractmethod
    def detect(self, frame: np.ndarray) -> list[HandDetection]:
        raise NotImplementedError


class PersonDetector(ABC):
    @abstractmethod
    def count_people(self, frame: np.ndarray) -> int:
        raise NotImplementedError


class MediaPipeHandDetector(HandDetector):
    """MediaPipe hand-landmark adapter. Model resources are initialized once."""

    def __init__(self, config: ValidationConfig, max_hands: int = 4) -> None:
        try:
            import mediapipe as mp
        except ImportError as exc:
            raise RuntimeError("MediaPipe is missing. Install requirements.txt.") from exc

        self._mp = mp
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            model_complexity=1,
            min_detection_confidence=config.hand_detection_confidence,
            min_tracking_confidence=config.hand_tracking_confidence,
        )

    def detect(self, frame: np.ndarray) -> list[HandDetection]:
        height, width = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self._hands.process(rgb)
        detections: list[HandDetection] = []
        if not result.multi_hand_landmarks:
            return detections

        handedness_list = result.multi_handedness or []
        for index, hand_landmarks in enumerate(result.multi_hand_landmarks):
            points = [(float(p.x), float(p.y)) for p in hand_landmarks.landmark]
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            bbox = (
                max(0, int(min(xs) * width)),
                max(0, int(min(ys) * height)),
                min(width - 1, int(max(xs) * width)),
                min(height - 1, int(max(ys) * height)),
            )
            label, score = "Unknown", 0.0
            if index < len(handedness_list) and handedness_list[index].classification:
                classification = handedness_list[index].classification[0]
                label, score = classification.label, float(classification.score)
            detections.append(HandDetection(label, score, points, bbox))
        return detections

    def close(self) -> None:
        self._hands.close()


class YoloPersonDetector(PersonDetector):
    """Ultralytics COCO person-class detector (class id 0)."""

    def __init__(self, confidence: float = 0.45, model_name: str = "yolo11n.pt") -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("Ultralytics is missing. Install requirements.txt.") from exc
        self._model = YOLO(model_name)
        self._confidence = confidence

    def count_people(self, frame: np.ndarray) -> int:
        results = self._model.predict(
            source=frame, classes=[0], conf=self._confidence, verbose=False
        )
        if not results or results[0].boxes is None:
            return 0
        return len(results[0].boxes)


@dataclass
class DetectionBundle:
    hand_detector: HandDetector
    person_detector: PersonDetector | None = None

    def close(self) -> None:
        close = getattr(self.hand_detector, "close", None)
        if callable(close):
            close()
