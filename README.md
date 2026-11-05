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

## Linux camera pipeline

The API discovers actual `/dev/videoN` nodes and tests which ones can return a frame. Select a usable device in the Camera control in the desktop app. It does not assume camera index 0 and does not use browser camera permissions.

For diagnostics, run `ls -l /dev/video*` and `groups`. If the video device belongs to group `video` and your user is not in that group, add the user with `sudo usermod -aG video $USER`, then log out and back in. See `server/README.md` for the full API and camera troubleshooting details.

The model checkpoint, datasets, run outputs, and private captures are not stored in Git.
