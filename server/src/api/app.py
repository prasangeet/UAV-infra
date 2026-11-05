from __future__ import annotations

import base64
import os
import time
import tempfile
from pathlib import Path
from uuid import uuid4
from threading import Event, Lock, Thread, current_thread
from typing import Any, Iterator

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from starlette.concurrency import run_in_threadpool
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = ROOT / "runs/detect/cubit_det_yolo11n_1024_finetune50/weights/best.pt"
MODEL_PATH = Path(os.getenv("CUBIT_MODEL", str(DEFAULT_MODEL))).expanduser()
DEVICE = os.getenv("CUBIT_DEVICE", "0" if torch.cuda.is_available() else "cpu")
MAX_UPLOAD_BYTES = int(os.getenv("CUBIT_MAX_UPLOAD_MB", "100")) * 1024 * 1024

app = FastAPI(title="CUBIT Inspect API", version="0.5.0")
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


def camera_device_nodes() -> list[Path]:
    """Return Linux V4L2 video nodes instead of assuming an OpenCV numeric index."""
    return sorted(
        (path for path in Path("/dev").glob("video[0-9]*") if path.exists()),
        key=lambda path: int(path.name.removeprefix("video")),
    )


def camera_name(device: Path) -> str:
    sysfs_name = Path("/sys/class/video4linux") / device.name / "name"
    try:
        return sysfs_name.read_text(encoding="utf-8").strip() or device.name
    except OSError:
        return device.name


def inspect_camera(device: Path) -> dict[str, Any]:
    """Probe a V4L2 node and report why it cannot be used."""
    readable = os.access(device, os.R_OK)
    writable = os.access(device, os.W_OK)
    result: dict[str, Any] = {
        "device": str(device),
        "name": camera_name(device),
        "readable": readable,
        "writable": writable,
        "openable": False,
        "error": None,
    }

    if not readable or not writable:
        result["error"] = (
            f"No read/write permission for {device}. Check its owner/group and add your user "
            "to the device group (usually 'video'), then log out and back in."
        )
        return result

    capture = cv2.VideoCapture(str(device), cv2.CAP_V4L2)
    if not capture.isOpened():
        capture.release()
        capture = cv2.VideoCapture(str(device), cv2.CAP_ANY)

    if not capture.isOpened():
        capture.release()
        result["error"] = (
            f"OpenCV could not open {device}. It may be a metadata-only node, already in use, "
            "or unsupported by the installed OpenCV video backend."
        )
        return result

    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    ok, frame = capture.read()
    capture.release()

    if not ok or frame is None:
        result["error"] = (
            f"OpenCV opened {device}, but it did not return an image frame. "
            "This may be a non-capture node or a camera format/backend issue."
        )
        return result

    result["openable"] = True
    result["width"] = int(frame.shape[1])
    result["height"] = int(frame.shape[0])
    result["error"] = None
    return result


@app.get("/api/cameras")
def list_cameras() -> dict[str, Any]:
    """Discover and probe every Linux /dev/video* node for actual frame capture."""
    nodes = camera_device_nodes()
    cameras = [inspect_camera(node) for node in nodes]
    usable = [item for item in cameras if item["openable"]]

    if not nodes:
        message = (
            "No /dev/video* devices exist. Check that the webcam is connected and detected "
            "by Linux. Inspect v4l2-ctl --list-devices and ls -l /dev/video*."
        )
    elif not usable:
        details = "; ".join(
            f'{item["device"]}: {item["error"]}' for item in cameras if item["error"]
        )
        message = f"Found {len(nodes)} video device node(s), but none can capture frames. {details}"
    else:
        message = f"Found {len(usable)} usable camera capture device(s)."

    return {
        "cameras": cameras,
        "count": len(cameras),
        "usable_count": len(usable),
        "message": message,
    }


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
        self._camera_device: str | None = None
        self._frame_size = (0, 0)

    def start(self, device: str, conf: float, imgsz: int) -> None:
        allowed_devices = {str(path) for path in camera_device_nodes()}
        if device not in allowed_devices:
            raise RuntimeError(
                f"{device!r} is not a discovered /dev/video* device. Refresh the camera list and select a device."
            )

        with self._lifecycle_lock:
            with self._state_lock:
                if self._thread is not None and self._thread.is_alive():
                    if self._running:
                        raise RuntimeError("A camera is already running. Stop it before switching devices.")
                    raise RuntimeError("The camera worker is still shutting down. Try again shortly.")

            device_path = Path(device)
            if not os.access(device_path, os.R_OK | os.W_OK):
                raise RuntimeError(
                    f"Permission denied for {device}. Check device permissions and your Linux groups. "
                    "Your user may need video-group access followed by a full log out and log in."
                )

            capture = cv2.VideoCapture(device, cv2.CAP_V4L2)
            if not capture.isOpened():
                capture.release()
                capture = cv2.VideoCapture(device, cv2.CAP_ANY)
            if not capture.isOpened():
                capture.release()
                raise RuntimeError(
                    f"Could not open {device}. It may be busy, a metadata-only node, or unsupported."
                )

            capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self._stop_event.clear()

            with self._state_lock:
                self._capture = capture
                self._camera_device = device
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
                            self._error = f"OpenCV opened {self._camera_device}, but could not read a frame."
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
                "camera_device": self._camera_device,
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


VIDEO_SCAN_ROOT = Path(tempfile.gettempdir()) / "cubit-inspect-video-scans"
VIDEO_SCAN_ROOT.mkdir(parents=True, exist_ok=True)
VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm", ".mpg", ".mpeg"}
_video_jobs: dict[str, dict[str, Any]] = {}
_video_jobs_lock = Lock()


def _update_video_job(job_id: str, **changes: Any) -> None:
    with _video_jobs_lock:
        job = _video_jobs.get(job_id)
        if job is not None:
            job.update(changes)


def _video_job_snapshot(job: dict[str, Any]) -> dict[str, Any]:
    return {
        "job_id": job["job_id"],
        "filename": job["filename"],
        "status": job["status"],
        "progress": job["progress"],
        "current_frame": job["current_frame"],
        "total_frames": job["total_frames"],
        "error": job["error"],
        "summary": job["summary"],
        "result_url": f'/api/infer/video/{job["job_id"]}/result' if job["status"] == "completed" else None,
    }


def _process_video_scan(job_id: str, input_path: Path, output_path: Path, conf: float, imgsz: int) -> None:
    capture: cv2.VideoCapture | None = None
    writer: cv2.VideoWriter | None = None
    processed_frames = 0
    try:
        _update_video_job(job_id, status="loading_model", progress=0, error=None)
        model = get_model()
        _update_video_job(job_id, status="processing", progress=0, error=None)
        capture = cv2.VideoCapture(str(input_path))
        if not capture.isOpened():
            raise RuntimeError("OpenCV could not open this video. Try MP4, MOV, AVI, MKV or WebM.")

        total_frames = max(0, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if not fps or fps <= 0 or not np.isfinite(fps):
            fps = 25.0
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if width <= 0 or height <= 0:
            raise RuntimeError("The uploaded video has invalid frame dimensions.")

        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError("Could not create the annotated MP4. Check OpenCV video codec support.")

        class_counts: dict[str, int] = {"Crack": 0, "Spalling": 0, "Moisture": 0}
        confidence_sums: dict[str, float] = {"Crack": 0.0, "Spalling": 0.0, "Moisture": 0.0}
        total_detections = 0
        _update_video_job(job_id, total_frames=total_frames, fps=round(fps, 2), frame_width=width, frame_height=height)

        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            with _model_lock:
                result = model.predict(frame, imgsz=imgsz, conf=conf, device=DEVICE, verbose=False)[0]
            if result.boxes is not None:
                for box in result.boxes:
                    class_id = int(box.cls.item())
                    raw_class_name = str(result.names.get(class_id, f"Class {class_id}"))
                    known_classes = {"crack": "Crack", "spalling": "Spalling", "moisture": "Moisture"}
                    class_name = known_classes.get(raw_class_name.casefold(), raw_class_name)
                    class_counts.setdefault(class_name, 0)
                    confidence_sums.setdefault(class_name, 0.0)
                    class_counts[class_name] += 1
                    confidence_sums[class_name] += float(box.conf.item())
                    total_detections += 1
            writer.write(result.plot())
            processed_frames += 1
            progress = min(99, int(processed_frames * 100 / total_frames)) if total_frames > 0 else 0
            _update_video_job(job_id, current_frame=processed_frames, progress=progress)

        if processed_frames == 0:
            raise RuntimeError("The video contains no readable frames.")
        writer.release()
        writer = None
        capture.release()
        capture = None

        average_confidence = {
            name: round(confidence_sums[name] / count, 4) if count else 0.0
            for name, count in class_counts.items()
        }
        summary = {
            "total_detections": total_detections,
            "class_counts": class_counts,
            "average_confidence_by_class": average_confidence,
            "processed_frames": processed_frames,
            "total_frames": total_frames or processed_frames,
            "fps": round(fps, 2),
            "width": width,
            "height": height,
            "duration_seconds": round(processed_frames / fps, 2),
        }
        _update_video_job(
            job_id, status="completed", progress=100,
            current_frame=processed_frames, total_frames=total_frames or processed_frames,
            summary=summary, error=None,
        )
    except Exception as exc:
        _update_video_job(job_id, status="failed", error=f"{type(exc).__name__}: {exc}")
        output_path.unlink(missing_ok=True)
    finally:
        if capture is not None:
            capture.release()
        if writer is not None:
            writer.release()
        input_path.unlink(missing_ok=True)


@app.post("/api/infer/video", status_code=202)
async def start_video_inference(
    file: UploadFile = File(...),
    conf: float = Query(0.25, ge=0.0, le=1.0),
    imgsz: int = Query(1024, ge=32, le=2048),
) -> dict[str, Any]:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in VIDEO_EXTENSIONS:
        raise HTTPException(status_code=415, detail="Unsupported video format. Use MP4, MOV, M4V, AVI, MKV, MPG or WebM.")

    # Never load PyTorch weights or wait for camera shutdown on the ASGI event loop.
    camera_status = await run_in_threadpool(camera.status)
    if camera_status["running"]:
        stopped = await run_in_threadpool(camera.stop)
        if not stopped:
            raise HTTPException(status_code=409, detail="Stop the live camera before starting a video scan.")

    job_id = uuid4().hex
    job_dir = VIDEO_SCAN_ROOT / job_id
    job_dir.mkdir(parents=True, exist_ok=False)
    input_path = job_dir / f"source{suffix}"
    output_path = job_dir / "annotated.mp4"
    upload_size = 0

    try:
        with input_path.open("wb") as destination:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                upload_size += len(chunk)
                if upload_size > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"Video exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB upload limit.",
                    )
                await run_in_threadpool(destination.write, chunk)
        if upload_size == 0:
            raise HTTPException(status_code=400, detail="The uploaded video is empty.")
    except Exception:
        input_path.unlink(missing_ok=True)
        try:
            job_dir.rmdir()
        except OSError:
            pass
        raise
    finally:
        await file.close()

    job = {
        "job_id": job_id, "filename": Path(file.filename or "video").name,
        "status": "queued", "progress": 0, "current_frame": 0, "total_frames": 0,
        "error": None, "summary": None, "input_path": input_path, "output_path": output_path,
    }
    with _video_jobs_lock:
        _video_jobs[job_id] = job

    Thread(
        target=_process_video_scan,
        args=(job_id, input_path, output_path, conf, imgsz),
        name=f"cubit-video-{job_id[:8]}",
        daemon=True,
    ).start()
    return _video_job_snapshot(job)


@app.get("/api/infer/video/{job_id}")
def get_video_inference_status(job_id: str) -> dict[str, Any]:
    with _video_jobs_lock:
        job = _video_jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Video scan not found.")
        return _video_job_snapshot(job)


@app.get("/api/infer/video/{job_id}/result")
def get_video_inference_result(job_id: str) -> FileResponse:
    with _video_jobs_lock:
        job = _video_jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Video scan not found.")
        if job["status"] != "completed":
            raise HTTPException(status_code=409, detail="The annotated video is not ready yet.")
        output_path = job["output_path"]
    if not output_path.is_file():
        raise HTTPException(status_code=404, detail="The annotated video file is missing.")
    return FileResponse(output_path, media_type="video/mp4")


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
    device: str = Query(..., min_length=1),
    conf: float = Query(0.25, ge=0.0, le=1.0),
    imgsz: int = Query(1024, ge=32, le=2048),
) -> dict[str, Any]:
    get_model()
    try:
        camera.start(device, conf, imgsz)
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


def process_image_payload(payload: bytes, conf: float, imgsz: int) -> dict[str, Any]:
    """Decode and run CPU/GPU inference in a worker thread, not on the ASGI event loop."""
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

    # cv2 decoding, model initialization, prediction and annotation are blocking operations.
    return await run_in_threadpool(process_image_payload, payload, conf, imgsz)


@app.post("/api/infer/image")
async def infer_image(
    file: UploadFile = File(...),
    conf: float = Query(0.25, ge=0.0, le=1.0),
    imgsz: int = Query(1024, ge=32, le=2048),
) -> dict[str, Any]:
    return await run_image_inference(file, conf, imgsz)
