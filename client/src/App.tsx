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
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState("");
  const [annotatedFrame, setAnnotatedFrame] = useState("");
  const [busy, setBusy] = useState(false);
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);

  useEffect(() => {
    let active = true;
    fetch(`${API}/api/health`)
      .then((response) => {
        if (!response.ok) throw new Error("Inference service unavailable");
        return response.json();
      })
      .then((data: Health) => { if (active) setHealth(data); })
      .catch(() => { if (active) setHealth(null); });
    return () => { active = false; };
  }, []);

  useEffect(() => () => {
    streamRef.current?.getTracks().forEach((track) => track.stop());
    if (previewUrl) URL.revokeObjectURL(previewUrl);
  }, [previewUrl]);

  useEffect(() => {
    if (!cameraOn || source !== "camera") return;
    let cancelled = false;
    let timer = 0;

    const processNextFrame = async () => {
      if (cancelled) return;
      const video = videoRef.current;
      if (video && video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA && video.videoWidth > 0) {
        try {
          const canvas = document.createElement("canvas");
          canvas.width = video.videoWidth;
          canvas.height = video.videoHeight;
          const context = canvas.getContext("2d");
          if (!context) throw new Error("Could not access camera frame.");
          context.drawImage(video, 0, 0, canvas.width, canvas.height);
          const blob = await new Promise<Blob | null>((resolve) => canvas.toBlob(resolve, "image/jpeg", 0.82));
          if (!blob) throw new Error("Could not encode camera frame.");

          const form = new FormData();
          form.append("file", blob, "camera-frame.jpg");
          const startedAt = performance.now();
          const response = await fetch(`${API}/api/infer/frame?conf=${confidence}&imgsz=${imgsz}`, {
            method: "POST",
            body: form,
          });
          if (!response.ok) {
            const failure = await response.json().catch(() => ({}));
            throw new Error(failure.detail ?? "Frame inference failed.");
          }
          const result: InferenceResult = await response.json();
          if (cancelled) return;
          setAnnotatedFrame(`data:image/jpeg;base64,${result.image}`);
          setDetections(mapDetections(result.detections ?? []));
          setFps(Math.max(1, Math.round(1000 / (performance.now() - startedAt))));
          setCameraError("");
        } catch (error) {
          if (!cancelled) setCameraError(error instanceof Error ? error.message : "Live inference failed.");
        }
      }
      if (!cancelled) timer = window.setTimeout(processNextFrame, 100);
    };

    void processNextFrame();
    return () => { cancelled = true; window.clearTimeout(timer); };
  }, [cameraOn, source, confidence, imgsz]);

  const counts = useMemo(() => ({
    crack: detections.filter((item) => item.label.toLowerCase() === "crack").length,
    spalling: detections.filter((item) => item.label.toLowerCase() === "spalling").length,
    moisture: detections.filter((item) => item.label.toLowerCase() === "moisture").length,
  }), [detections]);

  function stopCamera() {
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    setCameraOn(false);
    setFps(0);
  }

  async function startCamera() {
    try {
      setCameraError("");
      if (!navigator.mediaDevices?.getUserMedia) {
        throw new Error("Camera access is unavailable in this WebView.");
      }
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: false,
      });
      streamRef.current = stream;
      setSource("camera");
      setSelectedFile(null);
      setAnnotatedFrame("");
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
      setCameraOn(true);
    } catch (error) {
      setCameraError(error instanceof Error ? error.message : "Unable to access camera.");
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
              {cameraOn && <span className="fps-counter">{fps || "—"} FPS</span>}
            </div>
            <div className="feed-stage">
              <video ref={videoRef} className={cameraOn && source === "camera" && !annotatedFrame ? "camera-source visible" : "camera-source"} muted playsInline />
              {cameraOn && source === "camera" && annotatedFrame && (
                <img className="annotated-preview" src={annotatedFrame} alt="Live annotated detection frame" />
              )}
              {source !== "camera" && previewUrl && source === "image" && (
                <img className="media-preview" src={annotatedFrame || previewUrl} alt="Inspection result" />
              )}
              {source === "video" && previewUrl && <video className="media-preview" src={previewUrl} controls />}
              {source === "camera" && !cameraOn && (
                <div className="empty-feed">
                  <div className="camera-glyph">◉</div>
                  <strong>Camera feed is inactive</strong>
                  <span>Connect a camera to begin real-time inspection.</span>
                  <button className="primary-button" onClick={startCamera}>Start camera <span>→</span></button>
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
              <div className="input-info"><span className="status-dot online" /> INPUT: {cameraOn ? "CAMERA 01" : selectedFile?.name ?? "NO INPUT SELECTED"}</div>
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
