# UAV Infrastructure Damage Detection

Research and application workspace for UAV-assisted infrastructure damage detection using CUBIT-Det and YOLO.

## CUBIT Inspect

The `client/` directory contains the Tauri 2 + React + TypeScript desktop application. The `server/` directory contains model training, evaluation, inference scripts, and the FastAPI service used by the desktop client.

### Stack

- Desktop: Tauri 2, React, TypeScript, Vite
- Inference API: FastAPI, OpenCV, Ultralytics YOLO
- Model: fine-tuned YOLO11n trained on CUBIT-Det (Crack, Spalling, Moisture)

### Development

Start the inference API in one terminal:

```fish
cd server
set -gx CUBIT_MODEL /absolute/path/to/best.pt
uv sync
uv run uvicorn src.api.app:app --host 127.0.0.1 --port 8000
```

Start the desktop client in another terminal:

```fish
cd client
pnpm install
pnpm tauri dev
```

The API loads model weights on the first inference request. Keep checkpoints, datasets, run outputs, and private captures outside Git.
