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
        if self.sample_fps <= 0 or self.sample_fps > 30:
            raise ValueError("Sample FPS must be between 0 and 30.")
        if not 1 <= self.expected_hands <= 4:
            raise ValueError("Expected hands must be between 1 and 4.")
        if self.maximum_hand_missing_seconds < 0 or self.maximum_extra_hand_seconds < 0:
            raise ValueError("Duration thresholds cannot be negative.")
        if not 0 <= self.allowed_people <= 10:
            raise ValueError("Allowed people must be between 0 and 10.")
        if not 0 < self.person_confidence <= 1:
            raise ValueError("Person confidence must be between 0 and 1.")
        if not 0 < self.hand_detection_confidence <= 1:
            raise ValueError("Hand confidence must be between 0 and 1.")
        if self.enable_working_area:
            area = self.working_area or {}
            x, y = float(area.get("x", 0)), float(area.get("y", 0))
            w, h = float(area.get("width", 1)), float(area.get("height", 1))
            if min(x, y, w, h) < 0 or x + w > 1 or y + h > 1 or w == 0 or h == 0:
                raise ValueError("Working area must be a valid normalized rectangle inside the frame.")
