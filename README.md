# Video Validation Studio

Local-first AI-assisted video validation for clothes folding, plate organization and similar table-work training data.

## Run with Docker (no local Python installation required)

**Prerequisite:** Docker Desktop installed and running.

### Option A — Pull the published GitHub image

The GitHub Actions workflow builds the image and publishes it to GitHub Container Registry (GHCR) whenever Docker-related application files are pushed to `main`.

In PowerShell:

```powershell
docker pull ghcr.io/sksagar0143/video-validation:latest
docker run -d --name video-validation -p 8501:8501 --restart unless-stopped -v video-validation-models:/root/.config/Ultralytics ghcr.io/sksagar0143/video-validation:latest
```

Open **http://localhost:8501**.

To view logs:
```powershell
docker logs -f video-validation
```

To stop and remove the container:
```powershell
docker stop video-validation
docker rm video-validation
```

The first run downloads the YOLO model weights inside the container. The named Docker volume preserves that model cache between container replacements.

**GHCR package access:** If Docker reports that the package is private or access is denied, open the repository's Packages page on GitHub, open the `video-validation` package settings and change visibility to Public. Alternatively, authenticate with a GitHub token that has `read:packages` permission using `docker login ghcr.io -u YOUR_GITHUB_USERNAME`.

### Option B — Build the image on your computer from the cloned source

```powershell
git clone https://github.com/skSagar0143/video-validation.git
cd video-validation
docker compose build
docker compose up -d
```

Then open **http://localhost:8501**. This builds the image locally and does not require Python on Windows.

### Docker notes
- The image uses Python 3.11 and CPU-only PyTorch; you do not need Python or CUDA installed on Windows.
- Video analysis runs inside Docker on your computer. The CPU image is compatible with ordinary Docker Desktop installations; NVIDIA GPU acceleration is not configured in this default image.
- The container exposes port 8501. If that port is already in use, change the left side of the port mapping, e.g. `-p 8502:8501`, and open http://localhost:8502.
- Upload limit is configured to 2 GB. Actual limits depend on available RAM/disk and Docker Desktop resource allocation.
- YOLO model files are persisted in the named volume `video-validation-models` (or `yolo-model-cache` with Compose).
- The Docker image is built for `linux/amd64`.

## Features
- Batch upload and local processing of multiple videos.
- MediaPipe hand landmarks: expected hand count, possible missing/occluded hands, and excess hands.
- YOLO person counting and configurable unwanted-object detection for supported COCO classes.
- Optional experimental stationary-hand heuristic, working-area check, blur and black-frame checks.
- Timestamped PASS / FAIL / REVIEW reports, evidence images, annotated MP4, per-video JSON/CSV and batch ZIP/JSON.
- Strategy-based validation rules and replaceable detector adapters.

## Requirements for non-Docker setup
- Python 3.11 recommended; Windows 10/11, macOS or Linux.
- 8 GB RAM minimum; 16 GB+ recommended.
- Internet is required for package installation and first YOLO model download.

## Detection interpretation and limitations
- Missing hands are REVIEW events: could be exit, occlusion, or model miss.
- Extra hands and additional people are FAIL events, but false detections are possible. Person detection is counting, not identity recognition.
- Unwanted-object checks support only selected classes from the pretrained COCO model. They do not detect arbitrary unknown objects; add a custom-trained detector for domain-specific objects.
- Stationary-hand detection is an experimental centroid-motion heuristic and can flag intentional pauses.
- Working-area detection checks all detected landmarks; calibrate the rectangle to avoid false flags.
- Blur and black-frame rules are image heuristics; tune thresholds for your camera and lighting.

## Tests
Run `pytest -q` in a Python environment. Unit tests cover rules and configuration; they do not replace testing on representative camera videos.

## Privacy
Video processing and inference run on your computer. First-run dependency/model downloads require internet. Do not use automated output as the sole basis for high-stakes decisions; inspect evidence and validate thresholds with labelled clips.
