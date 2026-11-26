from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

@dataclass
class ValidationConfig:
    sample_fps: float = 5.0
    expected_hands: int = 2
    maximum_hand_missing_seconds: float = 2.0
    maximum_extra_hand_seconds: float = 0.5
    allowed_people: int = 1
    person_confidence: float = 0.45
    hand_detection_confidence: float = 0.55
    hand_tracking_confidence: float = 0.5
    enable_person_detection: bool = True
    enable_object_detection: bool = True
    unwanted_object_names: list[str] | None = None
    object_confidence: float = 0.45
    enable_hand_motion_check: bool = False
    minimum_hand_motion_score: float = 0.003
    stationary_hand_seconds: float = 8.0
    enable_working_area: bool = False
    working_area: dict[str, float] | None = None
    enable_blur_check: bool = True
    minimum_blur_score: float = 25.0
    enable_black_frame_check: bool = True
    black_frame_mean_threshold: float = 8.0
    write_annotated_video: bool = True
    event_merge_gap_seconds: float = 0.6

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "ValidationConfig":
        allowed = cls.__dataclass_fields__.keys()
        return cls(**{k: v for k, v in values.items() if k in allowed})

    @classmethod
    def from_json(cls, path: str | Path) -> "ValidationConfig":
        with Path(path).open("r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))

    def validate(self) -> None:
        if not 0 < self.sample_fps <= 30: raise ValueError("Sample FPS must be greater than 0 and at most 30.")
        if not 1 <= self.expected_hands <= 4: raise ValueError("Expected hands must be between 1 and 4.")
        if min(self.maximum_hand_missing_seconds, self.maximum_extra_hand_seconds, self.stationary_hand_seconds) < 0:
            raise ValueError("Duration thresholds cannot be negative.")
        if not 0 <= self.allowed_people <= 10: raise ValueError("Allowed people must be between 0 and 10.")
        for name, value in (("Person", self.person_confidence), ("Hand detection", self.hand_detection_confidence),
                            ("Hand tracking", self.hand_tracking_confidence), ("Object", self.object_confidence)):
            if not 0 < value <= 1: raise ValueError(f"{name} confidence must be between 0 and 1.")
        if self.minimum_hand_motion_score < 0: raise ValueError("Minimum hand motion score cannot be negative.")
        if self.enable_working_area:
            area = self.working_area or {}
            x, y, w, h = (float(area.get(k, default)) for k, default in (("x",0),("y",0),("width",1),("height",1)))
            if min(x,y,w,h) < 0 or x+w > 1 or y+h > 1 or w == 0 or h == 0:
                raise ValueError("Working area must be a valid normalized rectangle inside the frame.")
