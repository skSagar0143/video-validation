# Video Validation Studio

Local-first AI-assisted video validation for clothes folding, plate organization and similar table-work training data.

## Features
- Batch upload and local processing of multiple videos.
- MediaPipe hand landmarks: expected hand count, possible missing/occluded hands, and excess hands.
- YOLO person counting and configurable unwanted-object detection for supported COCO classes.
- Optional experimental stationary-hand heuristic, working-area check, blur and black-frame checks.
- Timestamped PASS / FAIL / REVIEW reports, evidence images, annotated MP4, per-video JSON/CSV and batch ZIP/JSON.
- Strategy-based validation rules and replaceable detector adapters.

## Requirements
- Python 3.11 recommended; Windows 10/11, macOS or Linux.
- 8 GB RAM minimum; 16 GB+ recommended. NVIDIA GPU optional.
- Internet is required for package installation and first YOLO model download. Inference runs locally after weights are cached.
- MediaPipe 0.10.21 is pinned for the Solutions Hands API.

## Windows PowerShell
```powershell
git clone https://github.com/skSagar0143/video-validation.git
cd video-validation
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```
Open http://localhost:8501. If PowerShell blocks activation, use Command Prompt and run .venv\Scripts\activate.bat.

## Usage
1. Select one or more MP4, MOV, AVI, MKV or M4V files.
2. Configure expected hands and grace periods.
3. Enable person detection and set allowed people.
4. Enable unwanted-object detection and choose COCO object classes to flag.
5. Optionally enable stationary-hand review, work-area and video-quality checks.
6. Click Analyze selected videos; inspect events, evidence and annotated clips.
7. Download per-video reports or the batch ZIP.

## Detection interpretation and limitations
- Missing hands are REVIEW events: could be exit, occlusion, or model miss.
- Extra hands and additional people are FAIL events, but false detections are possible. Person detection is counting, not identity recognition.
- Unwanted-object checks support only selected classes from the pretrained COCO model. They do not detect arbitrary unknown objects; add a custom-trained detector for domain-specific objects.
- Stationary-hand detection is an experimental centroid-motion heuristic and can flag intentional pauses.
- Working-area detection checks all detected landmarks; calibrate the rectangle to avoid false flags.
- Blur and black-frame rules are image heuristics; tune thresholds for your camera and lighting.

## Tests
Run pytest -q. Unit tests cover rules and configuration; they do not replace testing on representative camera videos.

## Privacy
Video processing and inference run on your computer. First-run dependency/model downloads require internet. Do not use automated output as the sole basis for high-stakes decisions; inspect evidence and validate thresholds with labelled clips.
