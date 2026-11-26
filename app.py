from __future__ import annotations

from pathlib import Path
import tempfile
import zipfile

import pandas as pd
import streamlit as st

from video_validation.config import ValidationConfig
from video_validation.engine import VideoValidationEngine
from video_validation.reporting import result_to_dict, write_reports


st.set_page_config(page_title="Video Validation Studio", page_icon="🎥", layout="wide")
st.title("🎥 Video Validation Studio")
st.caption("Local-first AI quality checks for clothes folding, plate organization and other table-work videos.")
st.info("Videos are analyzed on this computer. AI detections can be uncertain: inspect REVIEW events and evidence before rejecting a clip.")

with st.sidebar:
    st.header("Validation settings")
    profile = st.selectbox("Video profile", ["General table work", "Clothes organization", "Plate organization"])
    expected_hands = st.number_input("Expected hands", min_value=1, max_value=4, value=2)
    sample_fps = st.slider("AI analysis FPS", min_value=1, max_value=12, value=5)
    missing_seconds = st.number_input("Hand missing grace period (seconds)", min_value=0.0, max_value=10.0, value=2.0, step=0.5)
    extra_seconds = st.number_input("Extra hand threshold (seconds)", min_value=0.0, max_value=5.0, value=0.5, step=0.1)
    enable_people = st.checkbox("Detect additional people", value=True)
    allowed_people = st.number_input("Allowed people in frame", min_value=0, max_value=10, value=1)
    person_confidence = st.slider("Person detection confidence", 0.1, 0.95, 0.45, 0.05)
    enable_roi = st.checkbox("Check hands against working area", value=False)
    if enable_roi:
        st.caption("Normalized rectangle: x/y start and width/height as a fraction of the frame.")
        roi_x = st.slider("Area left (x)", 0.0, 0.8, 0.05, 0.01)
        roi_y = st.slider("Area top (y)", 0.0, 0.8, 0.05, 0.01)
        roi_w = st.slider("Area width", 0.1, 1.0 - roi_x, min(0.9, 1.0 - roi_x), 0.01)
        roi_h = st.slider("Area height", 0.1, 1.0 - roi_y, min(0.9, 1.0 - roi_y), 0.01)
        roi = {"x": roi_x, "y": roi_y, "width": roi_w, "height": roi_h}
    else:
        roi = None
    st.subheader("Video quality")
    enable_blur = st.checkbox("Detect blurry frames", value=True)
    blur_threshold = st.number_input("Minimum sharpness score", min_value=0.0, max_value=500.0, value=25.0)
    enable_black = st.checkbox("Detect black frames", value=True)
    black_threshold = st.number_input("Black frame brightness threshold", min_value=0.0, max_value=50.0, value=8.0)
    save_annotated = st.checkbox("Create annotated MP4", value=True)

uploads = st.file_uploader(
    "Select one or more videos",
    type=["mp4", "mov", "avi", "mkv", "m4v"],
    accept_multiple_files=True,
    help="Choose multiple videos to process them in one batch.",
)
if uploads:
    st.write(f"**{len(uploads)} video(s) selected**")
    for item in uploads:
        st.write(f"• {item.name} — {item.size / (1024 * 1024):.1f} MB")

if st.button("▶ Analyze selected videos", type="primary", disabled=not uploads, use_container_width=True):
    config = ValidationConfig(
        sample_fps=float(sample_fps),
        expected_hands=int(expected_hands),
        maximum_hand_missing_seconds=float(missing_seconds),
        maximum_extra_hand_seconds=float(extra_seconds),
        allowed_people=int(allowed_people),
        person_confidence=float(person_confidence),
        enable_person_detection=enable_people,
        enable_working_area=enable_roi,
        working_area=roi,
        enable_blur_check=enable_blur,
        minimum_blur_score=float(blur_threshold),
        enable_black_frame_check=enable_black,
        black_frame_mean_threshold=float(black_threshold),
        write_annotated_video=save_annotated,
    )
    try:
        config.validate()
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    batch_results = []
    archive_paths = []
    progress_area = st.container()
    with tempfile.TemporaryDirectory(prefix="video_validation_") as temp_root:
        root = Path(temp_root)
        for index, uploaded in enumerate(uploads, start=1):
            st.divider()
            st.subheader(f"{index}/{len(uploads)} — {uploaded.name}")
            input_path = root / uploaded.name
            # Avoid path traversal from uploaded filenames.
            input_path = root / Path(uploaded.name).name
            input_path.write_bytes(uploaded.getvalue())
            output_dir = root / f"result_{index}"
            progress = st.progress(0.0)
            status_text = st.empty()

            def update_progress(value: float, message: str) -> None:
                progress.progress(value)
                status_text.caption(message)

            try:
                engine = VideoValidationEngine(config)
                result = engine.analyze(input_path, output_dir, update_progress)
                json_path, csv_path = write_reports(result, output_dir)
                payload = result_to_dict(result)
                batch_results.append(payload)
                archive_paths.extend([json_path, csv_path])
                if result.annotated_video_path:
                    archive_paths.append(Path(result.annotated_video_path))
                archive_paths.extend(Path(p) for p in result.evidence_paths)

                color = {"PASS": "green", "FAIL": "red", "REVIEW": "orange"}[result.status]
                st.markdown(f"### :{color}[{result.status}]")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Duration", f"{result.metadata.duration_seconds:.1f}s")
                col2.metric("Analyzed frames", result.analyzed_frames)
                col3.metric("FAIL events", result.summary["fail_events"])
                col4.metric("REVIEW events", result.summary["review_events"])

                if result.events:
                    event_rows = [{
                        "Severity": e.severity,
                        "Issue": e.title,
                        "Start": f"{e.start_seconds:.2f}s",
                        "End": f"{e.end_seconds:.2f}s",
                        "Duration": f"{e.duration_seconds:.2f}s",
                        "Details": e.details,
                    } for e in result.events]
                    st.dataframe(pd.DataFrame(event_rows), use_container_width=True, hide_index=True)
                else:
                    st.success("No configured validation violations were detected.")

                with st.expander("Video metadata and detection settings"):
                    st.json({"metadata": payload["video"], "summary": result.summary})

                if result.annotated_video_path and Path(result.annotated_video_path).exists():
                    st.video(result.annotated_video_path)
                if result.evidence_paths:
                    st.caption("Sampled evidence frames")
                    evidence_cols = st.columns(min(4, len(result.evidence_paths)))
                    for evidence_index, evidence in enumerate(result.evidence_paths):
                        with evidence_cols[evidence_index % len(evidence_cols)]:
                            st.image(evidence, caption=Path(evidence).name, use_container_width=True)

                st.download_button(
                    "Download JSON report",
                    data=json_path.read_bytes(),
                    file_name=json_path.name,
                    mime="application/json",
                    key=f"json_{index}",
                )
                st.download_button(
                    "Download CSV events",
                    data=csv_path.read_bytes(),
                    file_name=csv_path.name,
                    mime="text/csv",
                    key=f"csv_{index}",
                )
                if result.annotated_video_path and Path(result.annotated_video_path).exists():
                    st.download_button(
                        "Download annotated MP4",
                        data=Path(result.annotated_video_path).read_bytes(),
                        file_name=Path(result.annotated_video_path).name,
                        mime="video/mp4",
                        key=f"video_{index}",
                    )
            except Exception as exc:
                st.error(f"Could not analyze {uploaded.name}: {exc}")
                st.exception(exc)

        if batch_results:
            st.divider()
            st.header("Batch summary")
            summary_df = pd.DataFrame([{
                "Video": item["video"]["filename"],
                "Status": item["status"],
                "Duration (s)": round(item["video"]["duration_seconds"], 2),
                "Analyzed frames": item["analyzed_frames"],
                "FAIL events": item["summary"]["fail_events"],
                "REVIEW events": item["summary"]["review_events"],
            } for item in batch_results])
            st.dataframe(summary_df, use_container_width=True, hide_index=True)
            st.download_button(
                "Download batch JSON",
                data=__import__("json").dumps(batch_results, indent=2).encode("utf-8"),
                file_name="batch_validation_report.json",
                mime="application/json",
            )
            zip_path = root / "validation_outputs.zip"
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
                for path in archive_paths:
                    if path.exists():
                        archive.write(path, arcname=path.name)
            st.download_button(
                "Download all reports and evidence (ZIP)",
                data=zip_path.read_bytes(),
                file_name="validation_outputs.zip",
                mime="application/zip",
            )

st.divider()
st.caption("MVP limitations: person detection is not identity recognition; a missing-hand detection may be caused by occlusion or model error; object-specific checks need dedicated classes or custom training.")
