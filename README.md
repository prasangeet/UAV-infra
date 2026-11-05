# UAV Infrastructure Damage Detection

Research and application workspace for UAV-assisted infrastructure damage detection using CUBIT-Det and YOLO.

## CUBIT Inspect

- `client/`: Tauri 2 + React + TypeScript desktop application.
- `server/`: model training, evaluation, inference scripts, and FastAPI inference service.
- Fine-tuned YOLO11n classes: Crack, Spalling, Moisture.

## Start the application

Terminal 1 — Python inference service:

```fish
cd server
uv sync
set -gx CUBIT_MODEL /absolute/path/to/cubit_det_yolo11n_1024_finetune50/weights/best.pt
set -gx CUBIT_DEVICE 0
uv run uvicorn src.api.app:app --host 127.0.0.1 --port 8000
```

Terminal 2 — Tauri desktop app:

```fish
cd client
pnpm install
pnpm tauri dev
```

## Linux camera access

The desktop app does not use a browser permission prompt. The Python/OpenCV service opens Linux V4L2 devices as the current user. The camera selector probes `/dev/videoN` nodes and shows device ownership, group/mode, backend group membership, and OpenCV capture errors.

To run the guided permission check/setup:

```fish
cd server
bash scripts/setup_camera_access.sh
```

It asks before adding your account to the `video` group and never makes camera nodes world-writable. If group membership changes, fully log out and back into your desktop session before restarting the API. Full details are in `server/README.md`.

The model checkpoint, datasets, run outputs, and private captures are not stored in Git.
