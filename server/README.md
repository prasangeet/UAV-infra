# CUBIT Inspect inference service

FastAPI service used by the Tauri 2 + React desktop client.

## Model checkpoint

The fine-tuned checkpoint is intentionally not committed. Set `CUBIT_MODEL` to the absolute path of your existing fine-tuned `best.pt`.

Example (fish shell):

```fish
set -gx CUBIT_MODEL /mnt/volume/Projects/UAV-II/runs/detect/cubit_det_yolo11n_1024_finetune50/weights/best.pt
```

The API also checks `server/runs/detect/cubit_det_yolo11n_1024_finetune50/weights/best.pt` by default.

## Run

```fish
cd server
uv sync
uv run uvicorn src.api.app:app --host 127.0.0.1 --port 8000
```

- Health: `GET /api/health`
- Model details: `GET /api/model`
- Image inference: `POST /api/infer/image`
- Live frame inference: `POST /api/infer/frame`

Both inference endpoints accept a multipart `file` field and optional `conf` and `imgsz` query parameters. Responses include the annotated JPEG (base64), image dimensions, and detections.

The API binds to loopback only. The desktop client captures camera frames locally and submits them to this service. Model checkpoints, datasets, run outputs, and private captures remain local and are not committed.
