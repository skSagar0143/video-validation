from __future__ import annotations

import json
from pathlib import Path
import pandas as pd

from .models import VideoValidationResult


def result_to_dict(result: VideoValidationResult) -> dict:
    return {
        "video": {
            "filename": result.metadata.filename,
            "width": result.metadata.width,
            "height": result.metadata.height,
            "fps": result.metadata.fps,
            "frame_count": result.metadata.frame_count,
            "duration_seconds": result.metadata.duration_seconds,
        },
        "status": result.status,
        "analyzed_frames": result.analyzed_frames,
        "summary": result.summary,
        "events": [
            {
                "rule_id": e.rule_id,
                "title": e.title,
                "severity": e.severity,
                "start_seconds": round(e.start_seconds, 3),
                "end_seconds": round(e.end_seconds, 3),
                "duration_seconds": round(e.duration_seconds, 3),
                "details": e.details,
                "peak_value": e.peak_value,
            }
            for e in result.events
        ],
        "annotated_video_path": result.annotated_video_path,
        "evidence_paths": result.evidence_paths,
    }


def write_reports(result: VideoValidationResult, output_dir: str | Path) -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_stem = Path(result.metadata.filename).stem
    json_path = output_dir / f"{safe_stem}_report.json"
    csv_path = output_dir / f"{safe_stem}_events.csv"
    payload = result_to_dict(result)
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    rows = payload["events"]
    columns = ["rule_id", "title", "severity", "start_seconds", "end_seconds",
               "duration_seconds", "details", "peak_value"]
    pd.DataFrame(rows, columns=columns).to_csv(csv_path, index=False)
    return json_path, csv_path
