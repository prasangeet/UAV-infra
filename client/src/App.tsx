import { useEffect, useMemo, useRef, useState } from "react";
import "./App.css";

type Detection = {
  id: number;
  label: string;
  confidence: number;
  color: string;
};
type Health = {
  status: string;
  model_loaded: boolean;
  model_exists: boolean;
  device: string;
  model_name: string | null;
};
type InferenceResult = {
  image: string;
  detections: Array<{ class_name: string; confidence: number }>;
};
type CameraDevice = {
  device: string;
  name: string;
  readable: boolean;
  writable: boolean;
  openable: boolean;
  error: string | null;
  width?: number;
  height?: number;
};
type CameraDiscovery = {
  cameras: CameraDevice[];
  count: number;
  usable_count: number;
  message: string;
};

const API = import.meta.env.VITE_API_URL ?? "http://127.0.0.1:8000";

function colorFor(label: string): string {
  if (label.toLowerCase() === "crack") return "#f07843";
  if (label.toLowerCase() === "spalling") return "#e9b44c";
  return "#52a7d8";
}

function mapDetections(items: InferenceResult["detections"]): Detection[] {
  return items.map((item, index) => ({
    id: index + 1,
    label: item.class_name,
    confidence: item.confidence,
    color: colorFor(item.class_name),
  }));
}

function App() {
  const [page, setPage] = useState("Live inspection");
  const [health, setHealth] = useState<Health | null>(null);
  const [source, setSource] = useState<"camera" | "image" | "video">("camera");
  const [cameraOn, setCameraOn] = useState(false);
  const [cameraError, setCameraError] = useState("");
  const [detections, setDetections] = useState<Detection[]>([]);
  const [confidence, setConfidence] = useState(0.25);
  const [imgsz, setImgsz] = useState(1024);
  const [fps, setFps] = useState(0);
  const [cameraStreamKey, setCameraStreamKey] = useState(0);
  const [cameras, setCameras] = useState<CameraDevice[]>([]);
  const [selectedCamera, setSelectedCamera] = useState("");
  const [cameraScanMessage, setCameraScanMessage] = useState("Scanning Linux video devices…");
  const [scanningCameras, setScanningCameras] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState("");
  const [annotatedFrame, setAnnotatedFrame] = useState("");
  const [busy, setBusy] = useState(false);
  const cameraPollRef = useRef<number | null>(null);

  useEffect(() => {
    let active = true;

    fetch(`${API}/api/health`)
      .then((response) => {
        if (!response.ok) throw new Error("Inference service unavailable");
        return response.json();
      })
      .then((data: Health) => { if (active) setHealth(data); })
      .catch(() => { if (active) setHealth(null); });

    fetch(`${API}/api/cameras`)
      .then((response) => {
        if (!response.ok) throw new Error("Camera discovery endpoint unavailable.");
        return response.json() as Promise<CameraDiscovery>;
      })
      .then((data) => {
        if (!active) return;
        setCameras(data.cameras ?? []);
        setCameraScanMessage(data.message ?? "Camera scan complete.");
        const preferred = data.cameras?.find((camera) => camera.openable);
        setSelectedCamera((current) => (
          current && data.cameras?.some((camera) => camera.device === current && camera.openable)
            ? current
            : preferred?.device ?? ""
        ));
      })
      .catch((error) => {
        if (!active) return;
        setCameras([]);
        setCameraScanMessage(error instanceof Error ? error.message : "Could not discover cameras.");
      });

    return () => { active = false; };
  }, []);

  useEffect(() => {
    return () => {
      if (cameraPollRef.current !== null) window.clearTimeout(cameraPollRef.current);
      // Stop native capture when the desktop window is closed/unmounted.
      void fetch(`${API}/api/camera/stop`, { method: "POST" }).catch(() => undefined);
    };
  }, []);

  useEffect(() => {
    if (!previewUrl) return;
    return () => URL.revokeObjectURL(previewUrl);
  }, [previewUrl]);

  useEffect(() => {
    if (!cameraOn || source !== "camera") return;
    let cancelled = false;

    const pollStatus = async () => {
      try {
        const response = await fetch(`${API}/api/camera/status`);
        if (!response.ok) throw new Error("Could not read camera status.");
        const status = await response.json();
        if (cancelled) return;
        setDetections(mapDetections((status.detections ?? []).map((item: { class_name: string; confidence: number }) => ({
          class_name: item.class_name,
          confidence: item.confidence,
        }))));
        setFps(Math.round(status.fps ?? 0));
        if (status.error) setCameraError(status.error);
        if (!status.running) {
          setCameraOn(false);
          setFps(0);
          if (status.error) setCameraError(status.error);
          return;
        }
      } catch (error) {
        if (!cancelled) setCameraError(error instanceof Error ? error.message : "Camera status unavailable.");
      }
      if (!cancelled) cameraPollRef.current = window.setTimeout(pollStatus, 350);
    };

    void pollStatus();
    return () => {
      cancelled = true;
      if (cameraPollRef.current !== null) window.clearTimeout(cameraPollRef.current);
    };
  }, [cameraOn, source]);

  const counts = useMemo(() => ({
    crack: detections.filter((item) => item.label.toLowerCase() === "crack").length,
    spalling: detections.filter((item) => item.label.toLowerCase() === "spalling").length,
    moisture: detections.filter((item) => item.label.toLowerCase() === "moisture").length,
  }), [detections]);

  async function stopCamera() {
    try {
      await fetch(`${API}/api/camera/stop`, { method: "POST" });
    } catch {
      // The backend may already be stopped; the UI should still reset.
    }
    setCameraOn(false);
    setFps(0);
  }

  async function startCamera() {
    try {
      setCameraError("");
      if (!selectedCamera) {
        throw new Error(cameraScanMessage || "No usable camera device selected. Refresh the camera list.");
      }
      const response = await fetch(
        `${API}/api/camera/start?device=${encodeURIComponent(selectedCamera)}&conf=${confidence}&imgsz=${imgsz}`,
        { method: "POST" },
      );
      const result = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(result.detail ?? "Could not start camera.");
      setSource("camera");
      setSelectedFile(null);
      setAnnotatedFrame("");
      setDetections([]);
      setCameraStreamKey((key) => key + 1);
      setCameraOn(true);
    } catch (error) {
      setCameraOn(false);
      setCameraError(error instanceof Error ? error.message : "Unable to start native camera capture.");
    }
  }

  function chooseFile(file: File | undefined) {
    if (!file) return;
    stopCamera();
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setSelectedFile(file);
    setPreviewUrl(URL.createObjectURL(file));
    setSource(file.type.startsWith("video/") ? "video" : "image");
    setAnnotatedFrame("");
    setDetections([]);
    setCameraError("");
  }

  async function inspectImage() {
    if (!selectedFile || source !== "image") return;
    setBusy(true);
    setCameraError("");
    try {
      const form = new FormData();
      form.append("file", selectedFile);
      const response = await fetch(`${API}/api/infer/image?conf=${confidence}&imgsz=${imgsz}`, {
        method: "POST",
        body: form,
      });
      if (!response.ok) {
        const failure = await response.json().catch(() => ({}));
        throw new Error(failure.detail ?? "Image inference failed.");
      }
      const result: InferenceResult = await response.json();
      setAnnotatedFrame(`data:image/jpeg;base64,${result.image}`);
      setDetections(mapDetections(result.detections ?? []));
    } catch (error) {
      setCameraError(error instanceof Error ? error.message : "Image inference failed.");
    } finally {
      setBusy(false);
    }
  }

  async function refreshCameras() {
    setScanningCameras(true);
    setCameraError("");
    try {
      const response = await fetch(`${API}/api/cameras`);
      const data = await response.json().catch(() => ({})) as CameraDiscovery;
      if (!response.ok) throw new Error(data.message ?? "Camera discovery failed.");
      setCameras(data.cameras ?? []);
      setCameraScanMessage(data.message ?? "Camera scan complete.");
      const preferred = data.cameras?.find((camera) => camera.openable);
      setSelectedCamera((current) => (
        current && data.cameras?.some((camera) => camera.device === current && camera.openable)
          ? current
          : preferred?.device ?? ""
      ));
    } catch (error) {
      setCameraScanMessage(error instanceof Error ? error.message : "Could not discover cameras.");
    } finally {
      setScanningCameras(false);
    }
  }

  const usableCameras = cameras.filter((camera) => camera.openable);

  const engineLabel = health?.model_exists
    ? health.model_loaded ? "Model loaded" : "Checkpoint found"
    : "Checkpoint missing";

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark">C</div>
          <div><strong>CUBIT</strong><span>INSPECT</span></div>
        </div>
        <div className="workspace-label">WORKSPACE</div>
        <nav>
          {["Live inspection", "Image analysis", "Video analysis", "Inspection history"].map((item, index) => (
            <button
              key={item}
              className={page === item ? "nav-item active" : "nav-item"}
              onClick={() => {
                setPage(item);
                if (index === 1) setSource("image");
                if (index === 2) setSource("video");
                if (index > 0) stopCamera();
              }}
            >
              <span className="nav-icon">{["◉", "▧", "▷", "◷"][index]}</span>{item}
            </button>
          ))}
        </nav>
        <div className="sidebar-bottom">
          <div className="model-card">
            <div className="model-card-top"><span className={health?.model_exists ? "status-dot online" : "status-dot"} />AI ENGINE</div>
            <strong>{engineLabel}</strong>
            <small>{health?.device ?? "Waiting for inference service"}</small>
          </div>
          <div className="profile">
            <div className="avatar">U</div>
            <div><strong>Local workspace</strong><small>Infrastructure inspection</small></div>
            <span>···</span>
          </div>
        </div>
      </aside>

      <main className="main-content">
        <header className="topbar">
          <div className="breadcrumb">Workspace <span>/</span> <strong>{page}</strong></div>
          <div className="top-actions">
            <span className="connection"><i className={health ? "status-dot online" : "status-dot"} />{health ? "Local engine connected" : "Engine disconnected"}</span>
            <button className="icon-button" aria-label="Application menu">⌘</button>
          </div>
        </header>

        <section className="page-heading">
          <div>
            <div className="eyebrow">INFRASTRUCTURE INTELLIGENCE</div>
            <h1>{page}</h1>
            <p>Inspect structural surfaces and identify visible defects with computer vision.</p>
          </div>
          <div className="session-pill"><span className="live-pulse" /> LOCAL SESSION</div>
        </section>

        <section className="metrics-grid">
          <Metric label="DETECTIONS" value={String(detections.length).padStart(2, "0")} note="Current frame" icon="⌗" />
          <Metric label="CRACKS" value={String(counts.crack).padStart(2, "0")} note="Class 0 · Crack" icon="⌁" />
          <Metric label="SPALLING" value={String(counts.spalling).padStart(2, "0")} note="Class 1 · Spalling" icon="▱" />
          <Metric label="MOISTURE" value={String(counts.moisture).padStart(2, "0")} note="Class 2 · Moisture" icon="◌" />
        </section>

        <div className="content-grid">
          <section className="panel inspection-panel">
            <div className="panel-header">
              <div><h2>Inspection feed</h2><p>Live camera or local media input</p></div>
              <span className="feed-tag"><i className={cameraOn ? "status-dot online" : "status-dot"} /> {cameraOn ? "LIVE" : "STANDBY"}</span>
            </div>
            <div className="feed-toolbar">
              <div className="tabs">
                <button className={source === "camera" ? "tab active" : "tab"} onClick={() => { setSource("camera"); setAnnotatedFrame(""); }}>Camera</button>
                <button className={source === "image" ? "tab active" : "tab"} onClick={() => { stopCamera(); setSource("image"); }}>Image</button>
                <button className={source === "video" ? "tab active" : "tab"} onClick={() => { stopCamera(); setSource("video"); }}>Video</button>
              </div>
              {source === "camera" && (
                <div className="camera-controls">
                  <select
                    aria-label="Camera device"
                    value={selectedCamera}
                    onChange={(event) => setSelectedCamera(event.target.value)}
                    disabled={cameraOn || usableCameras.length === 0}
                  >
                    {usableCameras.length === 0 && <option value="">No usable camera found</option>}
                    {usableCameras.map((camera) => (
                      <option key={camera.device} value={camera.device}>
                        {camera.name} · {camera.device}
                      </option>
                    ))}
                  </select>
                  <button className="refresh-cameras" onClick={refreshCameras} disabled={scanningCameras || cameraOn}>
                    {scanningCameras ? "Scanning…" : "↻"}
                  </button>
                </div>
              )}
              {cameraOn && <span className="fps-counter">{fps || "—"} FPS</span>}
            </div>
            <div className="feed-stage">
              {cameraOn && source === "camera" && (
                <img
                  className="annotated-preview"
                  src={`${API}/api/camera/stream?session=${cameraStreamKey}`}
                  alt="Live camera stream with YOLO detections"
                  onError={() => setCameraError("Camera stream unavailable. Check that the inference server is running and camera capture is active.")}
                />
              )}
              {source !== "camera" && previewUrl && source === "image" && (
                <img className="media-preview" src={annotatedFrame || previewUrl} alt="Inspection result" />
              )}
              {source === "video" && previewUrl && <video className="media-preview" src={previewUrl} controls />}
              {source === "camera" && !cameraOn && (
                <div className="empty-feed">
                  <div className="camera-glyph">◉</div>
                  <strong>Camera feed is inactive</strong>
                  <span>{usableCameras.length ? `Selected device: ${selectedCamera}` : cameraScanMessage}</span>
                  {usableCameras.length > 0 ? (
                    <button className="primary-button" onClick={startCamera}>Start camera <span>→</span></button>
                  ) : (
                    <button className="secondary-button" onClick={refreshCameras} disabled={scanningCameras}>
                      {scanningCameras ? "Scanning devices…" : "Scan cameras again"}
                    </button>
                  )}
                </div>
              )}
              {source === "image" && !previewUrl && (
                <label className="empty-feed upload-zone">
                  <div className="camera-glyph">＋</div>
                  <strong>Choose an image to inspect</strong>
                  <span>JPG, PNG, BMP, TIFF or WebP</span>
                  <input type="file" accept="image/*" onChange={(event) => chooseFile(event.target.files?.[0])} />
                </label>
              )}
              {source === "video" && !previewUrl && (
                <label className="empty-feed upload-zone">
                  <div className="camera-glyph">▷</div>
                  <strong>Choose a video</strong>
                  <span>Video playback is available; video inference is a next increment.</span>
                  <input type="file" accept="video/*" onChange={(event) => chooseFile(event.target.files?.[0])} />
                </label>
              )}
              {cameraError && <div className="error-banner">{cameraError}</div>}
              <div className="feed-corner top-left">CUBIT / VISION</div>
              <div className="feed-corner bottom-right">YOLO11N · {imgsz} PX</div>
            </div>
            <div className="feed-footer">
              <div className="input-info"><span className={cameraOn ? "status-dot online" : "status-dot"} /> INPUT: {cameraOn ? selectedCamera : selectedFile?.name ?? (source === "camera" ? "NO CAMERA SELECTED" : "NO INPUT SELECTED")}</div>
              <div className="feed-actions">
                {source === "camera" && cameraOn && <button className="secondary-button" onClick={stopCamera}>Stop camera</button>}
                {source !== "camera" && (
                  <label className="secondary-button file-button">
                    Browse
                    <input type="file" accept={source === "video" ? "video/*" : "image/*"} onChange={(event) => chooseFile(event.target.files?.[0])} />
                  </label>
                )}
                {source === "image" && <button className="primary-button" disabled={!selectedFile || busy} onClick={inspectImage}>{busy ? "Analysing…" : "Run analysis →"}</button>}
              </div>
            </div>
          </section>

          <aside className="right-column">
            <section className="panel settings-panel">
              <div className="panel-header">
                <div><h2>Detection settings</h2><p>Configure inference parameters</p></div>
                <span className="settings-icon">⚙</span>
              </div>
              <div className="setting-row">
                <div><label>Confidence threshold</label><small>Minimum prediction score</small></div>
                <strong>{confidence.toFixed(2)}</strong>
              </div>
              <input className="range" type="range" min="0.05" max="0.9" step="0.05" value={confidence} onChange={(event) => setConfidence(Number(event.target.value))} />
              <div className="setting-row">
                <div><label>Input resolution</label><small>Inference image size</small></div>
                <select value={imgsz} onChange={(event) => setImgsz(Number(event.target.value))}>
                  <option value="640">640 px</option><option value="800">800 px</option><option value="1024">1024 px</option>
                </select>
              </div>
              <div className="divider" />
              <div className="model-info">
                <span>ACTIVE MODEL</span>
                <strong>{health?.model_name ?? "YOLO11n · Fine-tuned"}</strong>
                <small>{health?.device ?? "CUDA / CPU auto-detect"}</small>
              </div>
            </section>

            <section className="panel detections-panel">
              <div className="panel-header">
                <div><h2>Detected objects</h2><p>{detections.length} objects in current frame</p></div>
                <button className="more-button" aria-label="More options">···</button>
              </div>
              {detections.length ? (
                <div className="detection-list">
                  {detections.map((item) => (
                    <div className="detection-item" key={item.id}>
                      <span className="class-indicator" style={{ background: item.color }} />
                      <div className="detection-name"><strong>{item.label}</strong><small>Object #{String(item.id).padStart(3, "0")}</small></div>
                      <strong className="confidence-value">{(item.confidence * 100).toFixed(1)}%</strong>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="no-detections"><span>⌕</span><strong>No detections yet</strong><small>Detected defects will appear here.</small></div>
              )}
            </section>
          </aside>
        </div>
        <footer className="app-footer">
          <span>CUBIT INSPECT <b>·</b> LOCAL INFERENCE PLATFORM</span>
          <span>MODEL: YOLO11N <b>·</b> DATASET: CUBIT-DET</span>
        </footer>
      </main>
    </div>
  );
}

function Metric({ label, value, note, icon }: { label: string; value: string; note: string; icon: string }) {
  return (
    <div className="metric-card">
      <div className="metric-top"><span>{label}</span><i>{icon}</i></div>
      <div className="metric-value">{value}</div>
      <div className="metric-note">{note}</div>
    </div>
  );
}

export default App;
