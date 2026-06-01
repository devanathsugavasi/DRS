import React, { useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  Activity,
  BarChart3,
  Camera,
  CheckCircle2,
  Download,
  FileJson,
  FileText,
  Gauge,
  RefreshCw,
  UploadCloud,
  Video,
} from "lucide-react";
import "./styles.css";

const API = "http://127.0.0.1:8766";

const optionDefs = [
  ["ball_detection", "Ball Detection"],
  ["ball_tracking", "Ball Tracking"],
  ["trajectory_prediction", "Trajectory Prediction"],
  ["lbw_analysis", "LBW Analysis"],
  ["edge_detection", "Edge Detection Analysis"],
  ["replay_generation", "Replay Generation"],
];

const defaultOptions = Object.fromEntries(optionDefs.map(([key]) => [key, true]));

function App() {
  const [files, setFiles] = useState([]);
  const [options, setOptions] = useState(defaultOptions);
  const [job, setJob] = useState(null);
  const [progress, setProgress] = useState(0);
  const [dragging, setDragging] = useState(false);
  const [backend, setBackend] = useState("checking");
  const pollRef = useRef(null);

  useEffect(() => {
    fetch(`${API}/api/testing/health`)
      .then((res) => res.ok ? setBackend("online") : setBackend("offline"))
      .catch(() => setBackend("offline"));
  }, []);

  useEffect(() => () => clearInterval(pollRef.current), []);

  const mode = files.length === 2 ? "Dual Camera Mode" : "Single Camera Mode";
  const result = job?.result;
  const summary = result?.summary;

  const upload = async () => {
    if (files.length < 1 || files.length > 2) return;
    setProgress(8);
    const form = new FormData();
    form.append("video_a", files[0]);
    if (files[1]) form.append("video_b", files[1]);
    form.append("options_json", JSON.stringify(options));

    const response = await fetch(`${API}/api/testing/jobs`, { method: "POST", body: form });
    if (!response.ok) throw new Error("Upload failed");
    const created = await response.json();
    setJob(created);
    setProgress(35);
    clearInterval(pollRef.current);
    pollRef.current = setInterval(() => pollJob(created.job_id), 1400);
  };

  const pollJob = async (jobId) => {
    const response = await fetch(`${API}/api/testing/jobs/${jobId}`);
    if (!response.ok) return;
    const updated = await response.json();
    setJob(updated);
    if (updated.status === "processing") setProgress((old) => Math.min(88, old + 8));
    if (updated.status === "complete" || updated.status === "failed") {
      setProgress(updated.status === "complete" ? 100 : 0);
      clearInterval(pollRef.current);
    }
  };

  const onDrop = (event) => {
    event.preventDefault();
    setDragging(false);
    const selected = Array.from(event.dataTransfer.files).filter((file) => file.type.startsWith("video/")).slice(0, 2);
    setFiles(selected);
  };

  const exportUrl = (name) => result ? `${API}/api/testing/jobs/${job.job_id}/exports/${name}` : "#";

  return (
    <main className="app-shell">
      <header className="topbar">
        <div className="brand">
          <div className="logo">DRS</div>
          <div>
            <h1>Cricket DRS Testing Platform</h1>
            <p>Offline delivery upload analysis for single and dual camera workflows</p>
          </div>
        </div>
        <div className={`backend ${backend}`}>
          <span />
          Backend {backend}
        </div>
      </header>

      <section className="workspace">
        <aside className="left-panel">
          <Panel title="Upload Section" icon={<UploadCloud size={18} />}>
            <div
              className={`dropzone ${dragging ? "dragging" : ""}`}
              onDragOver={(event) => { event.preventDefault(); setDragging(true); }}
              onDragLeave={() => setDragging(false)}
              onDrop={onDrop}
            >
              <UploadCloud size={34} />
              <strong>Upload 1 or 2 cricket delivery videos</strong>
              <span>Drag and drop MP4/MOV files, or choose from disk</span>
              <input
                type="file"
                accept="video/*"
                multiple
                onChange={(event) => setFiles(Array.from(event.target.files).slice(0, 2))}
              />
            </div>
            <div className="file-list">
              {files.length === 0 && <span>No videos selected</span>}
              {files.map((file, index) => (
                <div className="file-row" key={`${file.name}-${index}`}>
                  <Video size={16} />
                  <span>Camera {index + 1}</span>
                  <strong>{file.name}</strong>
                </div>
              ))}
            </div>
            <div className="progress">
              <div style={{ width: `${progress}%` }} />
            </div>
            <button className="primary" disabled={files.length === 0 || backend !== "online"} onClick={upload}>
              Process Delivery
            </button>
          </Panel>

          <Panel title="Analysis Options" icon={<Activity size={18} />}>
            <div className="option-grid">
              {optionDefs.map(([key, label]) => (
                <label className="check" key={key}>
                  <input
                    type="checkbox"
                    checked={options[key]}
                    onChange={(event) => setOptions({ ...options, [key]: event.target.checked })}
                  />
                  <span>{label}</span>
                </label>
              ))}
            </div>
          </Panel>

          <Panel title="Testing Mode" icon={<Camera size={18} />}>
            <div className="mode-card">
              <strong>{mode}</strong>
              <p>{files.length === 2 ? "Synchronizes two feeds and fuses tracking confidence." : "Analyzes one feed with approximate depth and trajectory uncertainty."}</p>
            </div>
          </Panel>
        </aside>

        <section className="center-panel">
          <Panel title="Output Dashboard" icon={<BarChart3 size={18} />} fill>
            <div className="video-grid">
              <VideoPane title="Original Video" file={files[0]} />
              <VideoPane title="Ball Tracking + Trajectory Overlay" src={result ? exportUrl("video") : null} />
              <VideoPane title="Clean DRS Animation" src={result ? exportUrl("animation") : null} />
            </div>
            <div className="evidence-grid">
              <Metric label="Bounce Point" value={formatPoint(summary?.pitching_location)} />
              <Metric label="Impact Point" value={formatPoint(summary?.impact_location)} />
              <Metric label="Stump Collision" value={summary?.predicted_wicket_impact || "waiting"} />
              <Metric label="Reliability" value={summary?.reliability || "waiting"} />
              <Metric label="Slow Motion Replay" value={result ? "generated" : "waiting"} />
              <Metric label="Frame Review" value={result ? `${result.cameras?.[0]?.frames_processed || 0} frames` : "waiting"} />
              <Metric label="Sync Error" value={formatMs(result?.sync_status?.sync_error_ms)} />
              <Metric label="Replay FPS" value={result?.sync_status?.replay_fps ? `${result.sync_status.replay_fps} fps` : "--"} />
            </div>
          </Panel>
        </section>

        <aside className="right-panel">
          <Panel title="DRS Result Panel" icon={<Gauge size={18} />}>
            <div className="decision">
              <span>LBW Recommendation</span>
              <strong className={summary?.lbw_recommendation === "OUT" ? "out" : ""}>{summary?.lbw_recommendation || "PENDING"}</strong>
              <div className="confidence">
                <div style={{ width: `${Math.round((summary?.confidence_score || 0) * 100)}%` }} />
              </div>
              <small>Confidence {summary ? `${Math.round(summary.confidence_score * 100)}%` : "--"}</small>
            </div>
            <Metric label="Ball Speed" value={summary ? `${summary.ball_speed_kmh} km/h` : "--"} />
            <Metric label="Pitching Location" value={formatPoint(summary?.pitching_location)} />
            <Metric label="Impact Location" value={formatPoint(summary?.impact_location)} />
            <Metric label="Uncertainty" value={summary ? `${Math.round(summary.uncertainty * 100)}%` : "--"} />
            <Metric label="Tracking Reliability" value={summary?.reliability || "--"} />
            <Metric label="Failed Gates" value={summary?.gate?.failed_gates?.length ? summary.gate.failed_gates.join(", ") : "none"} />
          </Panel>

          <Panel title="Export Features" icon={<Download size={18} />}>
            <div className="export-grid">
              <ExportButton href={exportUrl("video")} disabled={!result} icon={<Video size={16} />} label="Analyzed Video" />
              <ExportButton href={exportUrl("animation")} disabled={!result} icon={<Video size={16} />} label="Clean DRS Animation" />
              <ExportButton href={exportUrl("pdf")} disabled={!result} icon={<FileText size={16} />} label="DRS Report PDF" />
              <ExportButton href={exportUrl("json")} disabled={!result} icon={<FileJson size={16} />} label="Tracking JSON" />
              <ExportButton href={exportUrl("csv")} disabled={!result} icon={<FileText size={16} />} label="Tracking CSV" />
            </div>
          </Panel>

          <Panel title="Accuracy Gates" icon={<Activity size={18} />}>
            <div className="quality-list">
              {(summary?.tracking_quality || []).map((item, index) => (
                <div className="quality-card" key={index}>
                  <strong>Camera {index + 1}: {item.reliability || "unknown"}</strong>
                  <span>Coverage {Math.round((item.detection_coverage || 0) * 100)}% | Max gap {item.max_missing_gap || 0} | Smooth {Math.round((item.trajectory_smoothness || 0) * 100)}%</span>
                  <span>Avg conf {Math.round((item.mean_confidence || 0) * 100)}% | Jump rejects {item.jump_rejections || 0}</span>
                  {(item.warnings || []).map((warning) => <p key={warning}>{warning}</p>)}
                </div>
              ))}
              {!summary && <span className="muted">Quality gates appear after analysis.</span>}
            </div>
          </Panel>

          <Panel title="Readiness" icon={<CheckCircle2 size={18} />}>
            <Readiness title="Detector Model" item={result?.model_status} type="model" />
            <Readiness title="Camera Calibration" item={result?.calibration_status} type="calibration" />
            <Readiness title="Camera Sync / Replay" item={result?.sync_status} type="sync" />
          </Panel>
        </aside>
      </section>

      <footer className="bottom-bar">
        <div>
          <strong>Status</strong>
          <span>{job ? `${job.job_id} | ${job.status}` : "Waiting for delivery upload"}</span>
        </div>
        <div>
          <strong>Pipeline</strong>
          <span>detect ball, bat, pads, stumps, track, sync, predict, export</span>
        </div>
        <button onClick={() => job?.job_id && pollJob(job.job_id)}>
          <RefreshCw size={16} /> Refresh
        </button>
      </footer>
    </main>
  );
}

function Panel({ title, icon, children, fill }) {
  return (
    <section className={`panel ${fill ? "fill" : ""}`}>
      <header>
        <div>{icon}<span>{title}</span></div>
      </header>
      {children}
    </section>
  );
}

function VideoPane({ title, file, src }) {
  const url = useMemo(() => file ? URL.createObjectURL(file) : src, [file, src]);
  return (
    <div className="video-pane">
      <div>{title}</div>
      {url ? <video src={url} controls muted /> : <div className="empty-video">Video appears after upload/analysis</div>}
    </div>
  );
}

function Metric({ label, value }) {
  return (
    <div className="metric">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function ExportButton({ href, disabled, icon, label }) {
  return (
    <a className={`export ${disabled ? "disabled" : ""}`} href={disabled ? undefined : href} download>
      {icon}
      <span>{label}</span>
    </a>
  );
}

function Readiness({ title, item, type }) {
  const usable = Boolean(item?.usable);
  return (
    <div className="readiness">
      <strong>{title}</strong>
      <span className={usable ? "good" : "warn"}>
        {item ? (usable ? "Usable" : "Not ready") : "Waiting"}
      </span>
      <ReadinessMetrics item={item} type={type} />
      <p>{item?.reason || item?.message || "Run an analysis to see readiness status."}</p>
    </div>
  );
}

function ReadinessMetrics({ item, type }) {
  if (!item) return null;
  if (type === "model") {
    return <small>mAP50 {fmtPct(item.map50)} | Ball recall {fmtPct(item.ball_recall)} | Precision {fmtPct(item.precision)}</small>;
  }
  if (type === "calibration") {
    return <small>Reproj {fmtNum(item.reprojection_error_px)}px | Homography {fmtNum(item.homography_error_cm)}cm | Pitch {fmtNum(item.pitch_coordinate_error_cm)}cm</small>;
  }
  if (type === "sync") {
    return <small>Sync {formatMs(item.sync_error_ms)} | Replay {item.replay_fps || "--"} fps | Drops {item.dropped_frames ?? "--"}</small>;
  }
  return null;
}

function formatPoint(point) {
  if (!point) return "--";
  if (Array.isArray(point)) return `${Math.round(point[0])}, ${Math.round(point[1])} px`;
  return String(point);
}

function formatMs(value) {
  return value === null || value === undefined ? "--" : `${Number(value).toFixed(2)} ms`;
}

function fmtPct(value) {
  return value === null || value === undefined ? "--" : `${Math.round(Number(value) * 100)}%`;
}

function fmtNum(value) {
  return value === null || value === undefined ? "--" : Number(value).toFixed(2);
}

createRoot(document.getElementById("root")).render(<App />);
