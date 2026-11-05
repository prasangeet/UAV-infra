from __future__ import annotations

import base64
import os
import time
from pathlib import Path
from threading import Event, Lock, Thread
from typing import Any, Iterator

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = ROOT / "runs/detect/cubit_det_yolo11n_1024_finetune50/weights/best.pt"
MODEL_PATH = Path(os.getenv("CUBIT_MODEL", str(DEFAULT_MODEL))).expanduser()
DEVICE = os.getenv("CUBIT_DEVICE", "0" if torch.cuda.is_available() else "cpu")
MAX_UPLOAD_BYTES = int(os.getenv("CUBIT_MAX_UPLOAD_MB", "100")) * 1024 * 1024

app = FastAPI(title="CUBIT Inspect API", version="0.3.0")
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
    """Load the configured model once, on first use."""
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
                candidate = YOLO(str(MODEL_PATH))
                if candidate.task != "detect":
                    raise HTTPException(
                        status_code=500,
                        detail="Configured checkpoint is not an object-detection model.",
                    )
                _model = candidate
    return _model


class CameraWorker:
    """Owns native camera access and performs inference outside the Tauri WebView."""

    def __init__(self) -> None:
        self._state_lock = Lock()
        self._capture: cv2.VideoCapture | None = None
        self._thread: Thread | None = None
        self._stop = Event()
        self._running = False
        self._latest_jpeg: bytes | None = None
        self._detections: list[dict[str, Any]] = []
        self._fps = 0.0
        self._error: str | None = None
        self._camera_index = 0
        self._conf = 0.25
        self._imgsz = 1024
        self._last_frame_at = 0.0

    def start(self, index: int, conf: float, imgsz: int) -> None:
        with self._state_lock:
            if self._running:
                return
            self._error = None
            self._latest_jpeg = None
            self._detections = []
            self._fps = 0.0
            self._camera_index = index
            self._conf = conf
            self._imgsz = imgsz
            self._stop.clear()

            # OpenCV handles the physical device; no getUserMedia/WebView permission is involved.
            capture = cv2.VideoCapture(index, cv2.CAP_V4L2)
            if not capture.isOpened():
                capture.release()
                capture = cv2.VideoCapture(index)
            if not capture.isOpened():
                capture.release()
                raise RuntimeError(
                    f"Could not open camera {index}. Check that /dev/video{index} exists "
                    "and your user has permission to access it (video group)."
                )

            capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self._capture = capture
            self._running = True
            self._thread = Thread(target=self._run, name="cubit-camera", daemon=True)
            self._thread.start()

    def _run(self) -> None:
        previous = time.perf_counter()
        try:
            model = get_model()
            while not self._stop.is_set():
                capture = self._capture
                if capture is None:
                    break
                ok, frame = capture.read()
                if not ok or frame is None:
                    with self._state_lock:
                        self._error = "Camera stopped returning frames."
                    break

                # Ultralytics predictor state is shared with image inference; serialize model calls.
                with _model_lock:
                    result = model.predict(
                        frame,
                        imgsz=self._imgsz,
                        conf=self._conf,
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
                    [cv2.IMWRITE_JPEG_QUALITY, 84],
                )
                if not success:
                    continue

                now = time.perf_counter()
                instantaneous_fps = 1.0 / max(now - previous, 1e-6)
                previous = now
                with self._state_lock:
                    # Smooth the displayed value to avoid a flickering FPS counter.
                    self._fps = instantaneous_fps if self._fps == 0 else (0.75 * self._fps + 0.25 * instantaneous_fps)
                    self._latest_jpeg = encoded.tobytes()
                    self._detections = detections
                    self._last_frame_at = now
        except Exception as exc:
            with self._state_lock:
                self._error = str(exc)
        finally:
            capture = self._capture
            self._capture = None
            if capture is not None:
                capture.release()
            with self._state_lock:
                self._running = False
                self._fps = 0.0

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread and thread.is_alive() and thread is not __import__("threading").current_thread():
            thread.join(timeout=3.0)
        with self._state_lock:
            self._running = False
            self._fps = 0.0
        self._thread = None

    def status(self) -> dict[str, Any]:
        with self._state_lock:
            return {
                "running": self._running,
                "camera_index": self._camera_index,
                "fps": round(self._fps, 1),
                "detections": list(self._detections),
                "error": self._error,
                "has_frame": self._latest_jpeg is not None,
            }

    def stream(self) -> Iterator[bytes]:
        last_sent = None
        while True:
            with self._state_lock:
                running = self._running
                frame = self._latest_jpeg
            if frame is not None and frame is not last_sent:
                last_sent = frame
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Cache-Control: no-cache\r\n\r\n" + frame + b"\r\n"
                )
            elif not running:
                break
            time.sleep(0.03)


camera = CameraWorker()


@app.on_event("shutdown")
def shutdown_camera() -> None:
    camera.stop()


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


@app.post("/api/camera/start")
def start_camera(
    index: int = Query(0, ge=0, le=15),
    conf: float = Query(0.25, ge=0.0, le=1.0),
    imgsz: int = Query(1024, ge=32, le=2048),
) -> dict[str, Any]:
    # Fail before opening the camera if the fine-tuned checkpoint is not configured.
    get_model()
    try:
        camera.start(index, conf, imgsz)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"status": "starting", **camera.status()}


@app.post("/api/camera/stop")
def stop_camera() -> dict[str, Any]:
    camera.stop()
    return {"status": "stopped", **camera.status()}


@app.get("/api/camera/status")
def camera_status() -> dict[str, Any]:
    return camera.status()


@app.get("/api/camera/stream")
def camera_stream() -> StreamingResponse:
    return StreamingResponse(
        camera.stream(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-cache, no-store", "Pragma": "no-cache"},
    )


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
        with _model_lock:
            result = model.predict(
                image,
                imgsz=imgsz,
                conf=conf,
                device=DEVICE,
                verbose=False,
            )[0]
        detections = []
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
