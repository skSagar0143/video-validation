# Video Validation System

A local-first desktop-friendly video validation tool for task videos such as clothes folding and plate organization.

## MVP features

- Select one or multiple videos from a local file picker.
- Analyze videos locally; videos are not uploaded to a server.
- Detect hands with MediaPipe Hands and people with YOLO.
- Validate expected hand count, hands leaving the video frame, people entering the frame, blur, and black frames.
- Configure a rectangular working area and validation thresholds.
- Generate per-video JSON and CSV reports plus an annotated MP4 when enabled.
- Review event timestamps and evidence snapshots in the Streamlit interface.

> Detection is probabilistic. A missing-hand event is a visual signal, not proof that a hand physically left the scene. Occlusion and model errors can cause uncertainty. Use the REVIEW status and inspect evidence before rejecting training data.

## Requirements

- Python 3.10–3.11 recommended.
- Windows 10/11, macOS, or Linux.
- 8 GB RAM minimum; NVIDIA GPU optional.
- First run downloads the YOLO model and MediaPipe package assets as required. After dependencies and model weights are cached, analysis can run locally without a hosted AI API.

## Setup (Windows PowerShell)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

Open the local URL printed by Streamlit (normally http://localhost:8501).

## Setup (macOS/Linux)

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

## How to use

1. Start the app and choose one or more MP4, MOV, AVI, or MKV videos.
2. Select the validation profile and adjust the expected number of hands, sampling FPS, missing-hand grace period, and person detection confidence.
3. Optionally enable a normalized working-area rectangle (x/y/width/height as percentages).
4. Click **Analyze selected videos**.
5. Review PASS / FAIL / REVIEW results, event timestamps, and frame evidence.
6. Download the CSV/JSON report and annotated video for each video.

## Validation rules

- **Expected hand count:** flags sustained deviations from the expected count.
- **Hand outside frame:** flags sustained periods where a hand is not detected; the event is labelled as possible hand loss/occlusion because the model cannot always distinguish these causes.
- **Additional person:** flags person detections beyond the configured allowed count.
- **Working area:** flags detected hand landmarks outside the configured normalized rectangle.
- **Blur / black frame:** optional frame quality checks.

## Project structure

```text
video-validation/
  app.py
  requirements.txt
  config/
    default.json
  video_validation/
    __init__.py
    config.py
    models.py
    video_reader.py
    detectors.py
    rules.py
    engine.py
    reporting.py
  tests/
    test_rules.py
```

## Notes

- The initial release is a local Streamlit desktop-style web UI, not a hosted service.
- YOLO person detection uses the Ultralytics pretrained COCO model. It detects people, not identity; it cannot determine whether a person is authorized.
- MediaPipe handedness labels are estimates and can be affected by mirrored camera footage.
- Object-specific rules (e.g. mobile phone detection) need a defined object list and may require a custom-trained model for reliable results.
- Do not use this MVP as the sole basis for high-stakes decisions. Validate thresholds on representative labelled clips.
