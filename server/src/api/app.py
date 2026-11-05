from __future__ import annotations

import base64
import grp
import os
import pwd
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


def linux_permission_diagnostics(
    nodes: list[Path],
    cameras: list[dict[str, Any]],
) -> dict[str, Any]:
    """Explain the current backend process' Linux camera access context."""
    uid = os.geteuid()
    gid = os.getegid()
    username = pwd.getpwuid(uid).pw_name
    effective_groups = set(os.getgroups()) | {gid}
    group_names = sorted(
        {
            grp.getgrgid(group_id).gr_name
            for group_id in effective_groups
            if _group_exists(group_id)
        }
    )
    try:
        video_gid = grp.getgrnam("video").gr_gid
        video_group_exists = True
    except KeyError:
        video_gid = None
        video_group_exists = False

    video_io = "Unknown"
    build_info = cv2.getBuildInformation()
    if "Video I/O:" in build_info:
        section = build_info.split("Video I/O:", 1)[1].split("\\n\\n", 1)[0]
        video_io = " ".join(line.strip() for line in section.splitlines() if line.strip())

    node_details: list[dict[str, Any]] = []
    for device, camera in zip(nodes, cameras):
        try:
            metadata = device.stat()
            owner = pwd.getpwuid(metadata.st_uid).pw_name
            group = grp.getgrgid(metadata.st_gid).gr_name
            mode = oct(metadata.st_mode & 0o777)
        except (OSError, KeyError):
            owner, group, mode = "unknown", "unknown", "unknown"
        node_details.append(
            {
                "device": str(device),
                "owner": owner,
                "group": group,
                "mode": mode,
                "readable": camera["readable"],
                "writable": camera["writable"],
                "openable": camera["openable"],
                "error": camera["error"],
            }
        )

    is_video_member = video_gid in effective_groups if video_gid is not None else False
    if not nodes:
        recommendation = (
            "The backend sees no /dev/video* nodes. This is not a Tauri permission prompt. "
            "Check that Linux detects the webcam and that the service is running on the host."
        )
    elif not any(camera["openable"] for camera in cameras):
        if not any(camera["readable"] and camera["writable"] for camera in cameras):
            recommendation = (
                "The backend cannot read/write the camera nodes. Check device ACLs and group "
                "membership. The setup script can add your user to the video group."
            )
        else:
            recommendation = (
                "The backend can access the device nodes but OpenCV could not capture a frame. "
                "Check whether these are metadata nodes, whether another app is using the camera, "
                "and whether the OpenCV build has V4L2 support."
            )
    else:
        recommendation = "At least one camera node is accessible and can return frames."

    return {
        "username": username,
        "uid": uid,
        "effective_groups": group_names,
        "video_group_exists": video_group_exists,
        "video_group_member": is_video_member,
        "opencv_video_io": video_io,
        "device_nodes": node_details,
        "recommendation": recommendation,
    }


def _group_exists(group_id: int) -> bool:
    try:
        grp.getgrgid(group_id)
        return True
    except KeyError:
        return False


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

    diagnostics = linux_permission_diagnostics(nodes, cameras)
    return {
        "cameras": cameras,
        "count": len(cameras),
        "usable_count": len(usable),
        "message": message,
        "diagnostics": diagnostics,
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
        # Include live OS diagnostics so the UI reports a real cause, not just "camera not found".
        discovery = list_cameras()
        raise HTTPException(
            status_code=409,
            detail={"message": str(exc), "diagnostics": discovery["diagnostics"]},
        ) from exc
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
