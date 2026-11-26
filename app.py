from __future__ import annotations
from pathlib import Path
import json, tempfile, zipfile
import pandas as pd
import streamlit as st
from video_validation.config import ValidationConfig
from video_validation.engine import VideoValidationEngine
from video_validation.reporting import result_to_dict, write_reports
from video_validation.detectors import YoloSceneDetector

st.set_page_config(page_title="Video Validation Studio",page_icon="🎥",layout="wide")
st.title("🎥 Video Validation Studio")
st.caption("Local AI-assisted checks for clothes folding, plate organization and other table-work videos.")
st.info("Videos are processed locally. Detection is probabilistic; inspect REVIEW events and evidence.")
with st.sidebar:
    st.header("Hand rules")
    st.selectbox("Workflow profile",["General table work","Clothes organization","Plate organization"])
    expected=st.number_input("Expected hands",1,4,2)
    sample_fps=st.slider("Analysis FPS",1,12,5)
    missing=st.number_input("Missing-hand grace period (seconds)",0.0,10.0,2.0,0.5)
    extra=st.number_input("Extra-hand grace period (seconds)",0.0,5.0,0.5,0.1)
    st.subheader("People and objects")
    enable_people=st.checkbox("Detect additional people",True)
    allowed_people=st.number_input("Allowed people",0,10,1)
    person_conf=st.slider("Person confidence",0.1,0.95,0.45,0.05)
    enable_objects=st.checkbox("Detect unwanted objects",True)
    defaults=sorted(YoloSceneDetector.DEFAULT_UNWANTED)
    unwanted=st.multiselect("Unwanted COCO object classes",defaults,default=defaults)
    object_conf=st.slider("Object confidence",0.1,0.95,0.45,0.05)
    st.caption("Only classes known to the pretrained COCO model are available here.")
    st.subheader("Hand movement")
    motion_enabled=st.checkbox("Flag prolonged stationary hands (experimental)",False)
    min_motion=st.number_input("Minimum normalized movement",0.0,0.1,0.003,0.001,format="%.3f")
    stationary_seconds=st.number_input("Stationary duration (seconds)",1.0,120.0,8.0,1.0)
    st.subheader("Working area")
    enable_roi=st.checkbox("Check hands against work area",False)
    if enable_roi:
        rx=st.slider("Area left",0.0,0.8,0.05,0.01); ry=st.slider("Area top",0.0,0.8,0.05,0.01)
        rw=st.slider("Area width",0.1,1.0-rx,min(0.9,1.0-rx),0.01)
        rh=st.slider("Area height",0.1,1.0-ry,min(0.9,1.0-ry),0.01)
        roi={"x":rx,"y":ry,"width":rw,"height":rh}
    else: roi=None
    st.subheader("Video quality")
    blur_enabled=st.checkbox("Detect blurry frames",True)
    blur=st.number_input("Minimum sharpness score",0.0,500.0,25.0)
    black_enabled=st.checkbox("Detect black frames",True)
    black=st.number_input("Black-frame brightness threshold",0.0,50.0,8.0)
    annotated=st.checkbox("Generate annotated MP4",True)
uploads=st.file_uploader("Choose one or multiple videos",type=["mp4","mov","avi","mkv","m4v"],accept_multiple_files=True)
if uploads:
    st.write(f"**{len(uploads)} video(s) selected**")
    for file in uploads: st.write(f"• {file.name} — {file.size/(1024*1024):.1f} MB")
if st.button("▶ Analyze selected videos",type="primary",disabled=not uploads,use_container_width=True):
    config=ValidationConfig(sample_fps=float(sample_fps),expected_hands=int(expected),
        maximum_hand_missing_seconds=float(missing),maximum_extra_hand_seconds=float(extra),
        allowed_people=int(allowed_people),person_confidence=float(person_conf),
        enable_person_detection=enable_people,enable_object_detection=enable_objects,
        unwanted_object_names=unwanted,object_confidence=float(object_conf),
        enable_hand_motion_check=motion_enabled,minimum_hand_motion_score=float(min_motion),
        stationary_hand_seconds=float(stationary_seconds),enable_working_area=enable_roi,working_area=roi,
        enable_blur_check=blur_enabled,minimum_blur_score=float(blur),
        enable_black_frame_check=black_enabled,black_frame_mean_threshold=float(black),
        write_annotated_video=annotated)
    try: config.validate()
    except ValueError as exc: st.error(str(exc)); st.stop()
    payloads=[]; archive_items=[]
    with tempfile.TemporaryDirectory(prefix="video_validation_") as temp:
        root=Path(temp)
        for index,uploaded in enumerate(uploads,1):
            st.divider(); st.subheader(f"{index}/{len(uploads)} — {uploaded.name}")
            input_path=root/f"input_{index}_{Path(uploaded.name).name}"; input_path.write_bytes(uploaded.getvalue())
            output=root/f"result_{index}"; progress=st.progress(0.0); status=st.empty()
            def update(value,message): progress.progress(value); status.caption(message)
            try:
                result=VideoValidationEngine(config).analyze(input_path,output,update)
                json_path,csv_path=write_reports(result,output); payload=result_to_dict(result); payloads.append(payload)
                paths=[json_path,csv_path]+([Path(result.annotated_video_path)] if result.annotated_video_path else [])+[Path(p) for p in result.evidence_paths]
                archive_items.extend((p,f"video_{index}/{p.name}") for p in paths if p.exists())
                st.subheader(f"Result: {result.status}")
                a,b,c,d=st.columns(4); a.metric("Duration",f"{result.metadata.duration_seconds:.1f}s")
                b.metric("Analyzed frames",result.analyzed_frames); c.metric("FAIL events",result.summary["fail_events"]); d.metric("REVIEW events",result.summary["review_events"])
                if result.events:
                    st.dataframe(pd.DataFrame([{"Severity":e.severity,"Rule":e.title,"Start":f"{e.start_seconds:.2f}s","End":f"{e.end_seconds:.2f}s","Duration":f"{e.duration_seconds:.2f}s","Details":e.details,"Evidence frame":e.evidence_frame} for e in result.events]),use_container_width=True,hide_index=True)
                else: st.success("No configured violations detected.")
                with st.expander("Metadata and full JSON"): st.json(payload)
                if result.annotated_video_path and Path(result.annotated_video_path).exists(): st.video(result.annotated_video_path)
                if result.evidence_paths:
                    cols=st.columns(min(4,len(result.evidence_paths)))
                    for j,p in enumerate(result.evidence_paths):
                        with cols[j%len(cols)]: st.image(p,caption=Path(p).name,use_container_width=True)
                st.download_button("Download JSON",json_path.read_bytes(),json_path.name,"application/json",key=f"json_{index}")
                st.download_button("Download CSV",csv_path.read_bytes(),csv_path.name,"text/csv",key=f"csv_{index}")
                if result.annotated_video_path and Path(result.annotated_video_path).exists():
                    p=Path(result.annotated_video_path); st.download_button("Download annotated MP4",p.read_bytes(),p.name,"video/mp4",key=f"mp4_{index}")
            except Exception as exc:
                st.error(f"Could not analyze {uploaded.name}: {exc}"); st.exception(exc)
        if payloads:
            st.divider(); st.header("Batch summary")
            st.dataframe(pd.DataFrame([{"Video":p["video"]["filename"],"Status":p["status"],"Duration (s)":round(p["video"]["duration_seconds"],2),"Analyzed frames":p["analyzed_frames"],"FAIL events":p["summary"]["fail_events"],"REVIEW events":p["summary"]["review_events"]} for p in payloads]),use_container_width=True,hide_index=True)
            st.download_button("Download batch JSON",json.dumps(payloads,indent=2).encode(),"batch_validation_report.json","application/json")
            archive=root/"validation_outputs.zip"
            with zipfile.ZipFile(archive,"w",zipfile.ZIP_DEFLATED) as z:
                for path,arcname in archive_items: z.write(path,arcname)
            st.download_button("Download all outputs (ZIP)",archive.read_bytes(),"validation_outputs.zip","application/zip")
st.divider()
st.caption("Limitations: COCO object classes only; unknown objects require a custom model. Hand-motion scoring is experimental. Person detection counts, not identities. Missing hands can be caused by occlusion or model misses.")
