import { useEffect, useMemo, useRef, useState } from "react";
import "./App.css";
import "./VideoAnalysis.css";

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

type VideoScanSummary = {
  total_detections: number;
  class_counts: Record<string, number>;
  average_confidence_by_class: Record<string, number>;
  processed_frames: number;
  total_frames: number;
  fps: number;
  width: number;
  height: number;
  duration_seconds: number;
};
type VideoScanJob = {
  job_id: string;
  filename: string;
  status: "queued" | "processing" | "completed" | "failed";
  progress: number;
  current_frame: number;
  total_frames: number;
  error: string | null;
  summary: VideoScanSummary | null;
  result_url: string | null;
};

const API = import.meta.env.VITE_API_URL ?? "http://127.0.0.1:8001";

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
  const [videoPhase, setVideoPhase] = useState<"idle" | "review" | "uploading" | "processing" | "completed" | "failed">("idle");
  const [videoUploadProgress, setVideoUploadProgress] = useState(0);
  const [videoScan, setVideoScan] = useState<VideoScanJob | null>(null);
  const [videoError, setVideoError] = useState("");
  const [videoAcknowledged, setVideoAcknowledged] = useState(false);
  const [showVideoPreview, setShowVideoPreview] = useState(false);
  const videoTaskActiveRef = useRef(false);
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

  useEffect(() => () => { videoTaskActiveRef.current = false; }, []);

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
    const isVideo = file.type.startsWith("video/") || /\.(mp4|mov|m4v|avi|mkv|webm|mpg|mpeg)$/i.test(file.name);
    setSource(isVideo ? "video" : "image");
    setAnnotatedFrame("");
    setDetections([]);
    setCameraError("");
    setVideoScan(null);
    setVideoUploadProgress(0);
    setVideoError("");
    setVideoAcknowledged(false);
    setShowVideoPreview(false);
    setVideoPhase(isVideo ? "review" : "idle");
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


  async function startVideoScan() {
    if (!selectedFile || source !== "video") return;

    setVideoPhase("uploading");
    setVideoUploadProgress(0);
    setVideoScan(null);
    setVideoError("");
    setVideoAcknowledged(false);
    setCameraError("");
    videoTaskActiveRef.current = true;

    try {
      const form = new FormData();
      form.append("file", selectedFile);

      const job = await new Promise<{ job_id: string }>((resolve, reject) => {
        const xhr = new XMLHttpRequest();
        xhr.open("POST", API + "/api/infer/video?conf=" + confidence + "&imgsz=" + imgsz);
        xhr.upload.onprogress = (event) => {
          if (event.lengthComputable) {
            setVideoUploadProgress(Math.round((event.loaded / event.total) * 100));
          }
        };
        xhr.onload = () => {
          let payload: { job_id?: string; detail?: string } = {};
          try {
            payload = JSON.parse(xhr.responseText || "{}");
          } catch {
            reject(new Error("The video service returned an unreadable response."));
            return;
          }
          if (xhr.status < 200 || xhr.status >= 300 || !payload.job_id) {
            reject(new Error(payload.detail ?? "Could not upload the video."));
            return;
          }
          setVideoUploadProgress(100);
          resolve({ job_id: payload.job_id });
        };
        xhr.onerror = () => reject(new Error("Video upload failed. Check that the inference service is reachable."));
        xhr.onabort = () => reject(new Error("Video upload was cancelled."));
        xhr.send(form);
      });

      setVideoPhase("processing");
      while (videoTaskActiveRef.current) {
        const response = await fetch(API + "/api/infer/video/" + job.job_id);
        const status = await response.json().catch(() => ({})) as VideoScanJob & { detail?: string };
        if (!response.ok) throw new Error(status.detail ?? "Could not read video scan progress.");

        setVideoScan(status);
        if (status.status === "completed") {
          setShowVideoPreview(false);
          setVideoPhase("completed");
          return;
        }
        if (status.status === "failed") {
          throw new Error(status.error ?? "Video processing failed.");
        }
        await new Promise((resolve) => window.setTimeout(resolve, 600));
      }
    } catch (error) {
      setVideoError(error instanceof Error ? error.message : "Video scan failed.");
      setVideoPhase("failed");
    }
  }

  function resetVideoScan() {
    setVideoScan(null);
    setVideoUploadProgress(0);
    setVideoError("");
    setVideoAcknowledged(false);
    setShowVideoPreview(false);
    videoTaskActiveRef.current = false;
    setVideoPhase("review");
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

  const videoSummary = source === "video" && videoPhase === "completed" ? videoScan?.summary : null;
  const displayCounts = videoSummary ? {
    crack: videoSummary.class_counts.Crack ?? 0,
    spalling: videoSummary.class_counts.Spalling ?? 0,
    moisture: videoSummary.class_counts.Moisture ?? 0,
  } : counts;
  const videoProgress = videoPhase === "uploading" ? videoUploadProgress : (videoScan?.progress ?? 0);

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
          <Metric label="DETECTIONS" value={String(videoSummary ? videoSummary.total_detections : detections.length).padStart(2, "0")} note={videoSummary ? "Video observations" : "Current frame"} icon="⌗" />
          <Metric label="CRACKS" value={String(displayCounts.crack).padStart(2, "0")} note="Class 0 · Crack" icon="⌁" />
          <Metric label="SPALLING" value={String(displayCounts.spalling).padStart(2, "0")} note="Class 1 · Spalling" icon="▱" />
          <Metric label="MOISTURE" value={String(displayCounts.moisture).padStart(2, "0")} note="Class 2 · Moisture" icon="◌" />
        </section>

        <div className="content-grid">
          <section className="panel inspection-panel">
            <div className="panel-header">
              <div><h2>Inspection feed</h2><p>Live camera or local media input</p></div>
              <span className="feed-tag"><i className={cameraOn || videoPhase === "processing" ? "status-dot online" : "status-dot"} /> {cameraOn ? "LIVE" : videoPhase === "processing" || videoPhase === "uploading" ? "SCANNING" : videoPhase === "completed" ? "SCAN COMPLETE" : "STANDBY"}</span>
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
              {source === "video" && previewUrl && (
                <div className="video-preview-shell">
                  {showVideoPreview ? (
                    <video
                      className="media-preview"
                      src={videoPhase === "completed" && videoScan?.result_url ? API + videoScan.result_url : previewUrl}
                      controls
                      preload="metadata"
                      playsInline
                    />
                  ) : (
                    <div className="video-preview-prompt">
                      <span className="video-preview-symbol">▷</span>
                      <span className="video-kicker">
                        {videoPhase === "completed" ? "ANNOTATED VIDEO READY" : "VIDEO SELECTED"}
                      </span>
                      <strong>{selectedFile?.name ?? "Selected video"}</strong>
                      <p>
                        {videoPhase === "completed"
                          ? "The processed output is ready. Preview it when you need playback."
                          : "Playback is deferred to keep the desktop WebView responsive while selecting and processing footage."}
                      </p>
                      <button
                        className="secondary-button"
                        onClick={() => setShowVideoPreview(true)}
                      >
                        {videoPhase === "completed" ? "Preview annotated video" : "Preview source video"}
                      </button>
                    </div>
                  )}
                </div>
              )}
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
                  <span>Upload MP4, MOV, AVI, MKV or WebM for defect detection.</span>
                  <input type="file" accept="video/mp4,video/quicktime,video/x-msvideo,video/x-matroska,video/webm,.m4v,.mpg,.mpeg" onChange={(event) => chooseFile(event.target.files?.[0])} />
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
                    <input type="file" accept={source === "video" ? "video/*" : "image/*"} disabled={source === "video" && (videoPhase === "uploading" || videoPhase === "processing")} onChange={(event) => chooseFile(event.target.files?.[0])} />
                  </label>
                )}
                {source === "image" && <button className="primary-button" disabled={!selectedFile || busy} onClick={inspectImage}>{busy ? "Analysing…" : "Run analysis →"}</button>}
              </div>
            </div>

                {source === "video" && selectedFile && (
                  <section className="video-workflow" aria-live="polite">
                    {videoPhase === "review" && (
                      <div className="video-review">
                        <div className="video-review-copy">
                          <span className="video-kicker">READY FOR REVIEW</span>
                          <h3>Confirm video scan</h3>
                          <p>Check the selected footage and detection settings before running inference.</p>
                        </div>
                        <div className="video-file-facts">
                          <div><span>FILE</span><strong>{selectedFile.name}</strong></div>
                          <div><span>SIZE</span><strong>{(selectedFile.size / (1024 * 1024)).toFixed(1)} MB</strong></div>
                          <div><span>CONFIDENCE</span><strong>{confidence.toFixed(2)}</strong></div>
                          <div><span>RESOLUTION</span><strong>{imgsz} px</strong></div>
                        </div>
                        <div className="video-review-actions">
                          <span>Processing runs locally on this device.</span>
                          <button className="primary-button" onClick={startVideoScan}>Confirm & start scan <span>→</span></button>
                        </div>
                      </div>
                    )}

                    {(videoPhase === "uploading" || videoPhase === "processing") && (
                      <div className="video-progress-card">
                        <div className="video-progress-heading">
                          <div>
                            <span className="video-kicker">{videoPhase === "uploading" ? "STEP 1 OF 2" : "STEP 2 OF 2"}</span>
                            <h3>{videoPhase === "uploading" ? "Uploading video" : "Analysing video frames"}</h3>
                          </div>
                          <strong>{videoProgress}%</strong>
                        </div>
                        <div
                          className="video-progress-track"
                          role="progressbar"
                          aria-label={videoPhase === "uploading" ? "Video upload progress" : "Video processing progress"}
                          aria-valuemin={0}
                          aria-valuemax={100}
                          aria-valuenow={videoProgress}
                        >
                          <span style={{ width: (videoProgress) + "%" }} />
                        </div>
                        <div className="video-progress-meta">
                          <span>{videoPhase === "uploading" ? "Transferring file to the local inference service…" : videoScan?.status === "queued" ? "Waiting for the inference worker…" : "Running YOLO detection on each frame…"}</span>
                          {videoPhase === "processing" && (videoScan?.total_frames ?? 0) > 0 && (
                            <strong>{videoScan.current_frame.toLocaleString()} / {videoScan.total_frames.toLocaleString()} frames</strong>
                          )}
                        </div>
                        <p className="video-processing-note">Keep CUBIT Inspect open while the scan is running.</p>
                      </div>
                    )}

                    {videoPhase === "completed" && videoScan?.summary && (
                      <div className="video-complete-card">
                        <div className="video-complete-heading">
                          <span className="video-complete-icon">✓</span>
                          <div><span className="video-kicker">PROCESSING COMPLETE</span><h3>Video scan finished</h3></div>
                        </div>
                        <div className="video-result-stats">
                          <div><strong>{videoScan.summary.total_detections.toLocaleString()}</strong><span>Defect observations</span></div>
                          <div><strong>{videoScan.summary.processed_frames.toLocaleString()}</strong><span>Frames analysed</span></div>
                          <div><strong>{videoScan.summary.duration_seconds.toFixed(1)}s</strong><span>Video duration</span></div>
                        </div>
                        <p>Annotated video is ready. Review the results above, then confirm the scan.</p>
                        {videoAcknowledged ? (
                          <div className="video-acknowledged"><span>✓</span> Scan results confirmed for this session.</div>
                        ) : (
                          <div className="video-review-actions">
                            <span>Confirmation does not export or permanently save this scan.</span>
                            <button className="primary-button" onClick={() => setVideoAcknowledged(true)}>Confirm scan results <span>✓</span></button>
                          </div>
                        )}
                        <button className="video-secondary-action" onClick={resetVideoScan}>Run another scan</button>
                      </div>
                    )}

                    {videoPhase === "failed" && (
                      <div className="video-failed-card">
                        <span className="video-failed-icon">!</span>
                        <div><span className="video-kicker">SCAN FAILED</span><h3>Video could not be processed</h3><p>{videoError}</p></div>
                        <button className="secondary-button" onClick={resetVideoScan}>Try again</button>
                      </div>
                    )}
                  </section>
                )}

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
                <div><h2>Detected objects</h2><p>{videoSummary ? videoSummary.total_detections.toLocaleString() + " defect observations in video" : detections.length + " objects in current frame"}</p></div>
                <button className="more-button" aria-label="More options">···</button>
              </div>
              {source === "video" && videoSummary ? (
                <div className="detection-list video-detection-summary">
                  {["Crack", "Spalling", "Moisture"].map((label) => (
                    <div className="detection-item" key={label}>
                      <span className="class-indicator" style={{ background: colorFor(label) }} />
                      <div className="detection-name">
                        <strong>{label}</strong>
                        <small>{(videoSummary.average_confidence_by_class[label] ?? 0) > 0
                          ? "Average confidence · " + ((videoSummary.average_confidence_by_class[label] ?? 0) * 100).toFixed(1) + "%"
                          : "No observations"}</small>
                      </div>
                      <strong className="confidence-value">{(videoSummary.class_counts[label] ?? 0).toLocaleString()}</strong>
                    </div>
                  ))}
                </div>
              ) : detections.length ? (
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
