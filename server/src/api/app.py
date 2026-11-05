from __future__ import annotations

import base64
import os
import time
from pathlib import Path
from threading import Event, Lock, Thread, current_thread
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

app = FastAPI(title="CUBIT Inspect API", version="0.4.0")
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
    """Load and validate the configured model once."""
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                if not MODEL_PATH.is_file():
                    raise HTTPException(
                        status_code=503,
                        detail=(
                            f"Model checkpoint not found: {MODEL_PATH}. "
                            "Set CUBIT_MODEL to your fine-tuned best.pt path."
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
    """Captures frames with OpenCV and runs YOLO outside the Tauri WebView."""

    def __init__(self) -> None:
        self._lifecycle_lock = Lock()
        self._state_lock = Lock()
        self._capture: cv2.VideoCapture | None = None
        self._thread: Thread | None = None
        self._stop_event = Event()
        self._running = False
        self._latest_jpeg: bytes | None = None
        self._frame_sequence = 0
        self._detections: list[dict[str, Any]] = []
        self._fps = 0.0
        self._error: str | None = None
        self._camera_index = 0
        self._frame_size = (0, 0)

    def start(self, index: int, conf: float, imgsz: int) -> None:
        with self._lifecycle_lock:
            with self._state_lock:
                if self._thread is not None and self._thread.is_alive():
                    if self._running:
                        raise RuntimeError("Camera is already running.")
                    raise RuntimeError("Camera worker is still shutting down. Try again shortly.")

            capture = cv2.VideoCapture(index, cv2.CAP_V4L2)
            if not capture.isOpened():
                capture.release()
                capture = cv2.VideoCapture(index)
            if not capture.isOpened():
                capture.release()
                raise RuntimeError(
                    f"Could not open camera {index} (/dev/video{index}). "
                    "Check that the device exists and your user can access it. "
                    "On Linux, inspect /dev/video* permissions and your video group membership."
                )

            capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self._stop_event.clear()

            with self._state_lock:
                self._capture = capture
                self._camera_index = index
                self._frame_size = (0, 0)
                self._latest_jpeg = None
                self._frame_sequence = 0
                self._detections = []
                self._fps = 0.0
                self._error = None
                self._running = True
                self._thread = Thread(
                    target=self._run,
                    args=(capture, conf, imgsz),
                    name="cubit-camera",
                    daemon=True,
                )
                self._thread.start()

    def _run(self, capture: cv2.VideoCapture, conf: float, imgsz: int) -> None:
        previous = time.perf_counter()
        try:
            model = get_model()
            while not self._stop_event.is_set():
                ok, frame = capture.read()
                if not ok or frame is None:
                    if not self._stop_event.is_set():
                        with self._state_lock:
                            self._error = "OpenCV could not read a frame from the camera."
                    break

                with _model_lock:
                    result = model.predict(
                        frame,
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
                                "xyxy": [round(float(v), 2) for v in box.xyxy[0].tolist()],
                            }
                        )

                annotated = result.plot()
                encoded_ok, encoded = cv2.imencode(
                    ".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 84]
                )
                if not encoded_ok:
                    continue

                now = time.perf_counter()
                instantaneous_fps = 1.0 / max(now - previous, 1e-6)
                previous = now
                with self._state_lock:
                    self._fps = (
                        instantaneous_fps
                        if self._fps == 0
                        else 0.75 * self._fps + 0.25 * instantaneous_fps
                    )
                    self._latest_jpeg = encoded.tobytes()
                    self._frame_sequence += 1
                    self._frame_size = (int(frame.shape[1]), int(frame.shape[0]))
                    self._detections = detections
        except Exception as exc:
            with self._state_lock:
                self._error = f"{type(exc).__name__}: {exc}"
        finally:
            capture.release()
            with self._state_lock:
                if self._capture is capture:
                    self._capture = None
                self._running = False
                self._fps = 0.0

    def stop(self) -> bool:
        with self._lifecycle_lock:
            with self._state_lock:
                self._stop_event.set()
                thread = self._thread
                capture = self._capture

            if capture is not None:
                capture.release()
            if thread is not None and thread.is_alive() and thread is not current_thread():
                thread.join(timeout=8.0)

            with self._state_lock:
                still_alive = thread is not None and thread.is_alive()
                if not still_alive:
                    self._thread = None
                    self._capture = None
                    self._running = False
                    self._fps = 0.0
                else:
                    self._error = "Camera worker is still stopping. Wait briefly before restarting."
                return not still_alive

    def status(self) -> dict[str, Any]:
        with self._state_lock:
            return {
                "running": self._running,
                "camera_index": self._camera_index,
                "fps": round(self._fps, 1),
                "detections": list(self._detections),
                "error": self._error,
                "has_frame": self._latest_jpeg is not None,
                "frame_sequence": self._frame_sequence,
                "frame_width": self._frame_size[0],
                "frame_height": self._frame_size[1],
            }

    def stream(self) -> Iterator[bytes]:
        last_sequence = -1
        while True:
            with self._state_lock:
                running = self._running
                frame = self._latest_jpeg
                sequence = self._frame_sequence

            if frame is not None and sequence != last_sequence:
                last_sequence = sequence
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Cache-Control: no-cache\r\n\r\n"
                    + frame
                    + b"\r\n"
                )
            elif not running:
                return
            time.sleep(0.02)


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
    get_model()
    try:
        camera.start(index, conf, imgsz)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"status": "starting", **camera.status()}


@app.post("/api/camera/stop")
def stop_camera() -> dict[str, Any]:
    stopped = camera.stop()
    if not stopped:
        raise HTTPException(status_code=503, detail=camera.status()["error"])
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
                        "xyxy": [round(float(v), 2) for v in box.xyxy[0].tolist()],
                    }
                )
        annotated = result.plot()
        encoded_ok, encoded = cv2.imencode(
            ".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 88]
        )
        if not encoded_ok:
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
