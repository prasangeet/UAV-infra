# CUBIT Inspect inference service

FastAPI service used by the Tauri 2 + React desktop client.

## Model checkpoint

The fine-tuned checkpoint is intentionally not committed. Set `CUBIT_MODEL` to the absolute path of your existing fine-tuned `best.pt`.

Example (fish shell):

```fish
set -gx CUBIT_MODEL /mnt/volume/Projects/UAV-II/runs/detect/cubit_det_yolo11n_1024_finetune50/weights/best.pt
```

The API also checks `server/runs/detect/cubit_det_yolo11n_1024_finetune50/weights/best.pt` by default.

## Native camera architecture

The Tauri WebView does **not** request webcam access. Python/OpenCV opens the Linux V4L2 device, reads frames, runs YOLO inference on the configured CUDA device, and exposes the annotated output as an MJPEG stream.

Endpoints:
- `POST /api/camera/start?index=0&conf=0.25&imgsz=1024`
- `POST /api/camera/stop`
- `GET /api/camera/status`
- `GET /api/camera/stream` (MJPEG)
- `GET /api/health`
- `GET /api/model`
- `POST /api/infer/image`

## Linux camera permissions

Check camera device nodes:

```fish
ls -l /dev/video*
groups
```

If the device is owned by group `video` and your user is not in that group, add the user to it:

```fish
sudo usermod -aG video $USER
```

Log out and log back in for the new group membership to take effect. Do not run the desktop app or API as root as a workaround.

## Run

```fish
cd server
uv sync
set -gx CUBIT_MODEL /absolute/path/to/best.pt
set -gx CUBIT_DEVICE 0
uv run uvicorn src.api.app:app --host 127.0.0.1 --port 8000
```

The API binds to loopback only. Keep checkpoints, datasets, run outputs, and private captures local; do not commit them.
