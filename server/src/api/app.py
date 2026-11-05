from __future__ import annotations

import base64
import os
from pathlib import Path
from threading import Lock
from typing import Any

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = ROOT / "runs/detect/cubit_det_yolo11n_1024_finetune50/weights/best.pt"
MODEL_PATH = Path(os.getenv("CUBIT_MODEL", str(DEFAULT_MODEL))).expanduser()
DEVICE = os.getenv("CUBIT_DEVICE", "0" if torch.cuda.is_available() else "cpu")
MAX_UPLOAD_BYTES = int(os.getenv("CUBIT_MAX_UPLOAD_MB", "100")) * 1024 * 1024

app = FastAPI(title="CUBIT Inspect API", version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:1420",
        "http://127.0.0.1:1420",
        "tauri://localhost",
        "http://tauri.localhost",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

_model: YOLO | None = None
_model_lock = Lock()


def get_model() -> YOLO:
    """Load the model once, on the first inference request."""
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                if not MODEL_PATH.is_file():
                    raise HTTPException(
                        status_code=503,
                        detail=(
                            f"Model checkpoint not found: {MODEL_PATH}. "
                            "Set CUBIT_MODEL to the fine-tuned best.pt path."
                        ),
                    )
                _model = YOLO(str(MODEL_PATH))
                if _model.task != "detect":
                    _model = None
                    raise HTTPException(status_code=500, detail="Configured checkpoint is not a detection model.")
    return _model


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "model_loaded": _model is not None,
        "model_exists": MODEL_PATH.is_file(),
        "device": DEVICE,
        "cuda_available": torch.cuda.is_available(),
        "model_name": MODEL_PATH.name if MODEL_PATH.is_file() else None,
    }


@app.get("/api/model")
def model_info() -> dict[str, Any]:
    return {
        "checkpoint": str(MODEL_PATH),
        "exists": MODEL_PATH.is_file(),
        "device": DEVICE,
        "class_names": {0: "Crack", 1: "Spalling", 2: "Moisture"},
    }


async def run_image_inference(
    file: UploadFile,
    conf: float,
    imgsz: int,
) -> dict[str, Any]:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Upload an image file.")
    payload = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image exceeds the upload limit.")
    image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Could not decode uploaded image.")

    model = get_model()
    try:
        # Ultralytics predictors are stateful; serialize access for the single-GPU desktop use case.
        with _model_lock:
            result = model.predict(
                image,
                imgsz=imgsz,
                conf=conf,
                device=DEVICE,
                verbose=False,
            )[0]
        detections: list[dict[str, Any]] = []
        if result.boxes is not None:
            for box in result.boxes:
                class_id = int(box.cls.item())
                detections.append(
                    {
                        "class_id": class_id,
                        "class_name": result.names.get(class_id, str(class_id)),
                        "confidence": float(box.conf.item()),
                        "xyxy": [round(float(value), 2) for value in box.xyxy[0].tolist()],
                    }
                )
        annotated = result.plot()
        success, encoded = cv2.imencode(
            ".jpg",
            annotated,
            [cv2.IMWRITE_JPEG_QUALITY, 88],
        )
        if not success:
            raise HTTPException(status_code=500, detail="Could not encode annotated result.")
        return {
            "image": base64.b64encode(encoded.tobytes()).decode("ascii"),
            "width": int(image.shape[1]),
            "height": int(image.shape[0]),
            "detections": detections,
            "count": len(detections),
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc


@app.post("/api/infer/image")
async def infer_image(
    file: UploadFile = File(...),
    conf: float = Query(0.25, ge=0.0, le=1.0),
    imgsz: int = Query(1024, ge=32, le=2048),
) -> dict[str, Any]:
    return await run_image_inference(file, conf, imgsz)


@app.post("/api/infer/frame")
async def infer_frame(
    file: UploadFile = File(...),
    conf: float = Query(0.25, ge=0.0, le=1.0),
    imgsz: int = Query(1024, ge=32, le=2048),
) -> dict[str, Any]:
    """Single-frame inference endpoint for the desktop live-camera loop."""
    return await run_image_inference(file, conf, imgsz)
