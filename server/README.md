# CUBIT Inspect inference service

FastAPI service used by the Tauri 2 + React desktop client.

## Model checkpoint

The fine-tuned checkpoint is intentionally not committed. Set `CUBIT_MODEL` to the absolute path of your existing fine-tuned `best.pt`.

Example (fish shell):

```fish
set -gx CUBIT_MODEL /mnt/volume/Projects/UAV-II/runs/detect/cubit_det_yolo11n_1024_finetune50/weights/best.pt
```

The API also checks `server/runs/detect/cubit_det_yolo11n_1024_finetune50/weights/best.pt` by default.

## Linux camera discovery

The desktop client does not assume that the webcam is camera index 0. The API scans every `/dev/videoN` node, reads its human-readable name from sysfs, checks device permissions, and briefly probes whether OpenCV can actually capture a frame. The UI shows usable devices in a selector.

Endpoints:

- `GET /api/cameras` — discover devices and return usability/permission diagnostics.
- `POST /api/camera/start?device=/dev/video2&conf=0.25&imgsz=1024` — start the selected device.
- `POST /api/camera/stop` — stop capture.
- `GET /api/camera/status` — latest detections, FPS and capture errors.
- `GET /api/camera/stream` — MJPEG annotated stream.
- `GET /api/health` and `GET /api/model`.
- `POST /api/infer/image` — single-image inference.

The Tauri WebView never calls browser `getUserMedia`. Python/OpenCV owns the V4L2 device, YOLO performs inference on the configured CUDA device, and React displays the MJPEG stream.


## Video processing and resource limits

Video scans run asynchronously so frame inference does not block the FastAPI event loop. The API exposes upload, job-status, and annotated-result endpoints:

- \`POST /api/infer/video?conf=0.25&imgsz=1024\` — upload a video and receive a job ID.
- \`GET /api/infer/video/{job_id}\` — read upload/processing status and frame progress.
- \`GET /api/infer/video/{job_id}/result\` — retrieve the annotated MP4 after completion.

Inference runs on every second frame by default to reduce sustained CPU/GPU load while preserving the output video's frame rate. The most recent annotated frame is held between inference samples. Tune this with:

- \`CUBIT_VIDEO_FRAME_STRIDE=1\` — infer every frame (highest load).
- \`CUBIT_VIDEO_FRAME_STRIDE=2\` — infer every second frame (default).
- \`CUBIT_TORCH_THREADS=4\` — limit PyTorch CPU intra-op threads (default 4).

The image inference endpoint also sends decoding, model initialization, prediction, and annotation to a worker thread rather than blocking the ASGI event loop.

## Troubleshooting

Inspect device nodes and the current user's groups:

```fish
ls -l /dev/video*
groups
```

Optional V4L2 diagnostics (install `v4l-utils` if the command is unavailable):

```fish
v4l2-ctl --list-devices
```

If a node is owned by group `video` and your user is not a member, add the user:

```fish
sudo usermod -aG video $USER
```

Log out of the desktop session and log back in so the new supplementary group is applied. Do not run Tauri or the inference API as root to work around device permissions.

If a node opens but cannot return a frame, it may be a metadata-only node, the camera may be held by another process, or the selected node may not expose a capture stream. Close other camera applications and use the camera selector to choose another usable node.

## Run

```fish
cd server
uv sync
set -gx CUBIT_MODEL /absolute/path/to/best.pt
set -gx CUBIT_DEVICE 0
uv run uvicorn src.api.app:app --host 127.0.0.1 --port 8000
```

Keep checkpoints, datasets, run outputs, and private captures local; do not commit them.
