from __future__ import annotations

from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from .config import ValidationConfig
from .detectors import MediaPipeHandDetector, YoloPersonDetector
from .models import FrameAnalysis, VideoValidationResult
from .rules import RuleFactory, build_events
from .video_reader import OpenCVVideoReader


ProgressCallback = Callable[[float, str], None]


class VideoValidationEngine:
    """Application service orchestrating frame sampling, detectors and rules."""

    def __init__(self, config: ValidationConfig) -> None:
        config.validate()
        self.config = config
        self.reader = OpenCVVideoReader()

    def analyze(
        self,
        video_path: str | Path,
        output_dir: str | Path,
        progress: ProgressCallback | None = None,
    ) -> VideoValidationResult:
        video_path, output_dir = Path(video_path), Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        metadata = self.reader.inspect(video_path)
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        hand_detector = MediaPipeHandDetector(self.config, max_hands=max(4, self.config.expected_hands + 2))
        person_detector = None
        if self.config.enable_person_detection:
            person_detector = YoloPersonDetector(self.config.person_confidence)
        writer = None
        annotated_path = output_dir / f"{video_path.stem}_annotated.mp4"
        analyses: list[FrameAnalysis] = []
        evidence_paths: list[str] = []
        original_fps = metadata.fps if metadata.fps > 0 else 25.0
        sample_every = max(1, int(round(original_fps / self.config.sample_fps)))
        effective_sample_fps = original_fps / sample_every
        frame_index = 0
        sampled = 0
        rules = RuleFactory.create(self.config)

        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                if frame_index % sample_every != 0:
                    frame_index += 1
                    continue

                timestamp = frame_index / original_fps
                hands = hand_detector.detect(frame)
                person_count = person_detector.count_people(frame) if person_detector else 0
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                brightness = float(np.mean(gray))
                outside = self._count_hands_outside_area(hands)
                analysis = FrameAnalysis(
                    timestamp, frame_index, hands, person_count,
                    blur_score, brightness, outside,
                )
                analyses.append(analysis)
                annotated = self._annotate(frame.copy(), analysis)
                if self.config.write_annotated_video:
                    if writer is None:
                        height, width = frame.shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        writer = cv2.VideoWriter(str(annotated_path), fourcc, effective_sample_fps, (width, height))
                    writer.write(annotated)

                # Keep a small set of sampled evidence images; final event timestamps are in the report.
                if sampled < 12 and (sampled == 0 or sampled % max(1, int(self.config.sample_fps * 5)) == 0):
                    evidence = output_dir / f"evidence_{sampled:04d}.jpg"
                    cv2.imwrite(str(evidence), annotated)
                    evidence_paths.append(str(evidence))

                sampled += 1
                if progress and metadata.frame_count:
                    progress(min(0.99, frame_index / metadata.frame_count), f"Analyzing {timestamp:.1f}s")
                frame_index += 1
        finally:
            capture.release()
            hand_detector.close()
            if writer is not None:
                writer.release()

        events = build_events(analyses, self.config, rules, 1.0 / max(effective_sample_fps, 0.1))
        fail_count = sum(event.severity == "FAIL" for event in events)
        review_count = sum(event.severity == "REVIEW" for event in events)
        status = "FAIL" if fail_count else ("REVIEW" if review_count else "PASS")
        if progress:
            progress(1.0, "Preparing report")

        return VideoValidationResult(
            metadata=metadata,
            status=status,
            analyzed_frames=len(analyses),
            events=events,
            annotated_video_path=str(annotated_path) if writer is not None and annotated_path.exists() else None,
            evidence_paths=evidence_paths,
            summary={
                "fail_events": fail_count,
                "review_events": review_count,
                "sample_fps": round(effective_sample_fps, 2),
                "expected_hands": self.config.expected_hands,
                "allowed_people": self.config.allowed_people,
            },
        )

    def _count_hands_outside_area(self, hands) -> int:
        if not self.config.enable_working_area:
            return 0
        area = self.config.working_area or {"x": 0, "y": 0, "width": 1, "height": 1}
        x1, y1 = area["x"], area["y"]
        x2, y2 = x1 + area["width"], y1 + area["height"]
        count = 0
        for hand in hands:
            if any(not (x1 <= x <= x2 and y1 <= y <= y2) for x, y in hand.landmarks):
                count += 1
        return count

    def _annotate(self, frame: np.ndarray, analysis: FrameAnalysis) -> np.ndarray:
        for hand in analysis.hands:
            x1, y1, x2, y2 = hand.bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (40, 220, 40), 2)
            for x, y in hand.landmarks:
                cv2.circle(frame, (int(x * frame.shape[1]), int(y * frame.shape[0])), 2, (0, 200, 255), -1)
            cv2.putText(frame, f"{hand.handedness} {hand.score:.2f}", (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 220, 40), 2)
        cv2.putText(frame, f"Hands: {len(analysis.hands)} | People: {analysis.person_count} | {analysis.timestamp:.1f}s",
                    (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)
        if self.config.enable_working_area:
            area = self.config.working_area or {"x": 0, "y": 0, "width": 1, "height": 1}
            h, w = frame.shape[:2]
            p1 = (int(area["x"] * w), int(area["y"] * h))
            p2 = (int((area["x"] + area["width"]) * w), int((area["y"] + area["height"]) * h))
            cv2.rectangle(frame, p1, p2, (255, 100, 0), 2)
        return frame
