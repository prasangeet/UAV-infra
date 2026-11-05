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

## Camera pipeline

Camera capture is handled directly by Python/OpenCV through the Linux V4L2 interface. YOLO runs in the same local inference service, and the annotated frames are rendered by the Tauri React client using an MJPEG stream. Browser `getUserMedia` permissions are not used.

The model checkpoint, datasets, run outputs, and private captures are not stored in Git.
