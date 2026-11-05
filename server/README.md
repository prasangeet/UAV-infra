# CUBIT Inspect inference service

FastAPI service used by the Tauri 2 + React desktop client.

## Model checkpoint

The fine-tuned checkpoint is intentionally not committed. Set `CUBIT_MODEL` to the absolute path of your existing fine-tuned `best.pt`.

Example (fish shell):

```fish
set -gx CUBIT_MODEL /mnt/volume/Projects/UAV-II/runs/detect/cubit_det_yolo11n_1024_finetune50/weights/best.pt
```

## Camera permissions are Linux device permissions

The desktop app does not use a browser camera permission prompt. OpenCV runs inside the local Python/FastAPI process, and Linux controls that process' access to `/dev/videoN` using device-node ownership, ACLs, and supplementary groups.

The diagnostics endpoint reports the backend username, effective groups, device owner/group/mode, whether each node is readable/writable, whether OpenCV can open it and obtain a frame, and the OpenCV Video I/O build information.

- `GET /api/cameras` — discover/probe capture devices and return permission diagnostics.
- `POST /api/camera/start?device=/dev/video2&conf=0.25&imgsz=1024` — start selected device.
- `POST /api/camera/stop`
- `GET /api/camera/status`
- `GET /api/camera/stream` — MJPEG annotated stream.
- `GET /api/health` and `GET /api/model`.
- `POST /api/infer/image` — image inference.

### Guided camera access setup

From the `server/` directory, run:

```fish
bash scripts/setup_camera_access.sh
```

The script checks for camera nodes and the `video` group. It asks before adding the current account to that group; it does not change devices to world-writable and does not run the inference server as root. If group membership is changed, fully log out of the desktop session and log back in before restarting the API and Tauri app.

Manual diagnostics:

```fish
ls -l /dev/video*
groups
v4l2-ctl --list-devices
```

If device nodes are accessible but OpenCV cannot return frames, another process may be using a node, the node may be metadata-only, or the installed OpenCV build may lack V4L2 support. The desktop app displays per-device probe errors and backend diagnostics.

## Run

```fish
cd server
uv sync
set -gx CUBIT_MODEL /absolute/path/to/best.pt
set -gx CUBIT_DEVICE 0
uv run uvicorn src.api.app:app --host 127.0.0.1 --port 8000
```

Keep checkpoints, datasets, run outputs, and private captures local; do not commit them.
