from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any

@dataclass(frozen=True)
class HandDetection:
    handedness: str
    score: float
    landmarks: list[tuple[float, float]]
    bbox: tuple[int, int, int, int]

@dataclass(frozen=True)
class FrameAnalysis:
    timestamp: float
    frame_index: int
    hands: list[HandDetection] = field(default_factory=list)
    person_count: int = 0
    blur_score: float | None = None
    brightness_mean: float | None = None
    hands_outside_working_area: int = 0
    unwanted_objects: list[str] = field(default_factory=list)
    hand_motion_score: float | None = None

@dataclass(frozen=True)
class ValidationEvent:
    rule_id: str
    title: str
    severity: str
    start_seconds: float
    end_seconds: float
    details: str
    peak_value: float | None = None
    evidence_frame: int | None = None
    @property
    def duration_seconds(self) -> float:
        return max(0.0, self.end_seconds - self.start_seconds)

@dataclass
class VideoMetadata:
    filename: str
    width: int
    height: int
    fps: float
    frame_count: int
    duration_seconds: float

@dataclass
class VideoValidationResult:
    metadata: VideoMetadata
    status: str
    analyzed_frames: int
    events: list[ValidationEvent]
    annotated_video_path: str | None = None
    evidence_paths: list[str] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)
