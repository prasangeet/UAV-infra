
# CUBIT Inspect — Inference Service

This directory contains the local Python inference service used by the CUBIT Inspect Tauri desktop application, along with the model-training, evaluation, dataset-preparation, and standalone inference utilities.

The API is implemented in `src/api/app.py`. The recommended launcher is `main.py`, which binds the service to `127.0.0.1:8001`. The React frontend uses this port by default.

## Model checkpoint

The API defaults to the fine-tuned checkpoint tracked in this repository:

`runs/detect/cubit_det_yolo11n_1024_finetune50/weights/best.pt`

The path is resolved relative to the server directory. To use another checkpoint, set `CUBIT_MODEL` to its absolute path before starting the service.

**Fish shell**

```fish
set -gx CUBIT_MODEL "/absolute/path/to/best.pt"
set -gx CUBIT_DEVICE "0"
```

**Bash / Zsh**

```bash
export CUBIT_MODEL="/absolute/path/to/best.pt"
export CUBIT_DEVICE="0"
```

Set `CUBIT_DEVICE` to `cpu` to force CPU inference. When it is not set, the API selects CUDA device `0` if PyTorch reports CUDA availability; otherwise it uses CPU.

The starter configuration is in `.env.example`. The service reads process environment variables; it does not automatically load that file as dotenv configuration.

## Run the API

Install the Python dependencies:

```bash
cd server
uv sync
```

Start the recommended API entry point:

```bash
uv run main.py
```

The API is available at:

- Base URL: http://127.0.0.1:8001
- Interactive docs: http://127.0.0.1:8001/docs
- OpenAPI schema: http://127.0.0.1:8001/openapi.json

### Alternate launcher and port

The package also exposes the `cubit-api` command through `src/api/run.py`. That alternate entry point currently binds to port `8000`:

```bash
uv run cubit-api
```

The desktop frontend defaults to port `8001`. If you use the alternate launcher, set the frontend's `VITE_API_URL` to `http://127.0.0.1:8000` before starting Tauri.

## Configuration

| Variable | Default | Purpose |
|---|---:|---|
| `CUBIT_MODEL` | Fine-tuned checkpoint path above | Model weights to load |
| `CUBIT_DEVICE` | CUDA `0` when available, otherwise `cpu` | Inference device |
| `CUBIT_MAX_UPLOAD_MB` | `100` | Maximum size accepted for image/video uploads |
| `CUBIT_VIDEO_FRAME_STRIDE` | `2` | Run inference on every Nth video frame |
| `CUBIT_TORCH_THREADS` | `4` | PyTorch CPU thread limit; OpenCV thread count is also limited |

The model is loaded lazily on the first inference request and reused by the service. A lock serializes access to the shared model instance.

## API endpoints

### Health and model

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/health` | Service, checkpoint, model-loaded, CUDA, and device status |
| GET | `/api/model` | Checkpoint path and detector classes |

### Camera

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/cameras` | Discover and probe Linux camera devices |
| POST | `/api/camera/start` | Start camera capture and live YOLO inference |
| POST | `/api/camera/stop` | Stop camera capture |
| GET | `/api/camera/status` | Read worker state, FPS, latest detections, and errors |
| GET | `/api/camera/stream` | Stream annotated frames as MJPEG |

The camera start endpoint accepts `device`, `conf`, and `imgsz` query parameters. Example:

```text
POST /api/camera/start?device=/dev/video2&conf=0.25&imgsz=1024
```

The camera implementation is Linux/V4L2-specific. It scans `/dev/video*`, reads names from `/sys/class/video4linux`, checks read/write access, and probes whether OpenCV can retrieve a frame. It does not call browser `getUserMedia`; Python/OpenCV owns the camera device.

### Image inference

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/infer/image` | Accept a multipart image and return an annotated JPEG (base64) plus detections |

Query parameters:

- `conf` — confidence threshold (default `0.25`; range `0.0–1.0`).
- `imgsz` — YOLO inference image size (default `1024`; range `32–2048`).

Image decoding, model initialization, prediction, and annotation are dispatched to a worker thread.

### Video inference

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/infer/video` | Accept a multipart video upload and create an asynchronous scan job; returns HTTP `202` |
| GET | `/api/infer/video/{job_id}` | Return job status, progress, and summary |
| GET | `/api/infer/video/{job_id}/result` | Return the annotated MP4 after successful processing |

Supported extensions: MP4, MOV, M4V, AVI, MKV, WebM, MPG, and MPEG.

The video worker processes frames outside the FastAPI event loop. Its states include `queued`, `loading_model`, `processing`, `completed`, and `failed`. The result summary contains class counts, average confidence by class, number of frames processed and analysed, dimensions, FPS, and duration.

By default, the API runs detection on every second frame. It writes the last annotated result for intervening frames so that the output retains the source frame rate. This reduces sustained inference load but can miss defects appearing only between sampled frames.

- `CUBIT_VIDEO_FRAME_STRIDE=1` — analyse every frame.
- `CUBIT_VIDEO_FRAME_STRIDE=2` — analyse every second frame (default).
- `CUBIT_TORCH_THREADS=4` — limit PyTorch CPU threads (default).

The upload limit defaults to 100 MiB. The input file is removed after processing. Jobs are held in memory and are not restored after a backend restart; temporary scan files are kept under the operating system's temporary directory.

## Linux camera troubleshooting

Inspect device nodes and current group membership:

```fish
ls -l /dev/video*
groups
```

Optional V4L2 diagnostics (install the distribution's `v4l-utils` package if needed):

```fish
v4l2-ctl --list-devices
```

The selected user must have read/write access to the chosen video node. On systems where access is controlled by the `video` group, a user can be added with:

```fish
sudo usermod -aG video $USER
```

Log out of the desktop session and log back in for supplementary group changes to take effect. Do not run Tauri or the inference API as root to bypass camera permissions.

A device node may be present but unusable if it is metadata-only, busy in another application, or does not expose a capture stream. The API probes devices and reports the reason a node cannot be opened or cannot return a frame.

## Linux video playback

The desktop UI uses the Tauri WebView for source and annotated video preview. On Linux, playback depends on the WebKit media stack and the GStreamer plugins/codecs installed on the host. Install the media plugin packages appropriate for the distribution and the source video's container/codec if the WebView shows a permanent loading indicator or fails to start playback.

This is separate from backend video inference: OpenCV decodes the uploaded video for processing, while the WebView decodes media for UI preview.

## Related training and data utilities

Run these commands from `server/`:

| Purpose | Entry point |
|---|---|
| Download CUBIT-Det archives | `scripts/dataset_download.py` |
| Verify/extract archives | `scripts/extract_cubit_det.py` |
| Generate overlapping image tiles | `scripts/tile_cubit_det.py` |
| Visualize dataset annotations | `scripts/visualize_cubit_det.py` |
| Train YOLO | `src/training/train_yolo.py` |
| Configurable evaluation | `src/evaluation/evaluate.py` |
| Standalone image inference | `src/inference/infer_image.py` |
| Standalone video inference | `src/inference/infer_video.py` |

See the [repository README](../README.md) for the complete project overview, dataset workflow, training/evaluation instructions, recorded validation snapshot, and project metadata.

## Operational notes

- The API binds to loopback by default and is intended for a local desktop workflow.
- The detector currently has three classes: Crack, Spalling, and Moisture.
- Inference quality and speed depend on the selected image size, confidence threshold, hardware, and video frame stride.
- Predictions are visual assistance only and do not replace professional structural inspection.
