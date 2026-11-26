from __future__ import annotations

from pathlib import Path
import cv2

from .models import VideoMetadata


class OpenCVVideoReader:
    """Small adapter around OpenCV VideoCapture."""

    def inspect(self, path: str | Path) -> VideoMetadata:
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise ValueError(f"Could not open video: {path}")
        try:
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = float(capture.get(cv2.CAP_PROP_FPS) or 0)
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            safe_fps = fps if fps > 0 else 0.0
            duration = frame_count / safe_fps if safe_fps else 0.0
            return VideoMetadata(Path(path).name, width, height, safe_fps, frame_count, duration)
        finally:
            capture.release()
