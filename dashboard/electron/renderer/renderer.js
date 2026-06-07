const API_BASE = "http://localhost:8765";

const state = {
  decision: null,
  cameras: new Map(),
  activeAppeal: false,
};

const els = {
  engineState: document.getElementById("engine-state"),
  cameraPills: document.getElementById("camera-pills"),
  camera1: document.getElementById("camera-1"),
  camera2: document.getElementById("camera-2"),
  cameraPanel1: document.getElementById("camera-panel-1"),
  cameraPanel2: document.getElementById("camera-panel-2"),
  camera1Fps: document.getElementById("camera-1-fps"),
  camera2Fps: document.getElementById("camera-2-fps"),
  badge: document.getElementById("decision-badge"),
  title: document.getElementById("decision-title"),
  confidence: document.getElementById("ball-confidence"),
  impact: document.getElementById("impact-location"),
  wicket: document.getElementById("wicket-zone"),
  speed: document.getElementById("ball-speed"),
  trajectoryStatus: document.getElementById("trajectory-status"),
  canvas: document.getElementById("trajectory-canvas"),
  log: document.getElementById("system-log"),
  requestReview: document.getElementById("request-review"),
  confirmOut: document.getElementById("confirm-out"),
  confirmNotOut: document.getElementById("confirm-not-out"),
};

async function jsonFetch(route, options = {}) {
  const response = await fetch(`${API_BASE}${route}`, options);
  if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
  return response.json();
}

function log(message) {
  const stamp = new Date().toLocaleTimeString();
  els.log.textContent = `${stamp} | ${message}\n${els.log.textContent}`.slice(0, 1200);
}

async function refreshHealth() {
  try {
    const health = window.drs?.getHealth ? await window.drs.getHealth() : await jsonFetch("/api/health");
    els.engineState.textContent = `Engine ${health.status || "ok"} · ${health.active_model_name || health.model_name || "model unknown"}`;
  } catch (error) {
    els.engineState.textContent = "Engine offline";
    log(`Health failed: ${error.message}`);
  }
}

async function refreshCameraStatus() {
  try {
    const data = await jsonFetch("/api/cameras/fps");
    state.cameras.clear();
    data.cameras.forEach((camera) => state.cameras.set(Number(camera.id), camera));
    renderCameraPills(data.cameras);
  } catch (error) {
    renderCameraPills([
      { id: 1, fps: 0, status: "offline" },
      { id: 2, fps: 0, status: "offline" },
    ]);
  }
}

function renderCameraPills(cameras) {
  els.cameraPills.innerHTML = cameras.map((camera) => {
    const status = camera.status || "offline";
    return `<span class="camera-pill ${status}">Cam ${camera.id} · ${Number(camera.fps || 0).toFixed(1)} fps</span>`;
  }).join("");
  updatePanelStatus(1, els.cameraPanel1, els.camera1Fps);
  updatePanelStatus(2, els.cameraPanel2, els.camera2Fps);
}

function updatePanelStatus(cameraId, panel, fpsEl) {
  const camera = state.cameras.get(cameraId) || { fps: 0, status: "offline" };
  panel.classList.remove("live", "offline", "synthetic");
  panel.classList.add(camera.status || "offline");
  fpsEl.textContent = `${Number(camera.fps || 0).toFixed(1)} fps`;
}

function refreshCameraFrames() {
  els.camera1.src = `${API_BASE}/api/live/1.jpg?t=${Date.now()}`;
  els.camera2.src = `${API_BASE}/api/live/2.jpg?t=${Date.now()}`;
}

async function refreshDecision() {
  try {
    const decision = await jsonFetch("/api/decision/current");
    state.decision = decision;
    renderDecision(decision);
    drawTrajectory(decision);
  } catch (error) {
    log(`Decision poll failed: ${error.message}`);
  }
}

function renderDecision(decision) {
  const status = decision.status || "WAITING";
  state.activeAppeal = status !== "WAITING";
  els.badge.className = `badge ${statusClass(status)}`;
  els.badge.textContent = status.replace("_", " ");
  els.title.textContent = decision.outcome || statusText(status);
  els.confidence.textContent = formatPercent(decision.ball_confidence);
  els.impact.textContent = formatPoint(decision.impact_point);
  els.wicket.textContent = decision.wicket_zone_status || "--";
  els.speed.textContent = decision.ball_speed_kmh ? `${Number(decision.ball_speed_kmh).toFixed(1)} km/h` : "--";
  els.trajectoryStatus.textContent = decision.trajectory?.length ? `${decision.trajectory.length} points` : "Waiting";
  [els.confirmOut, els.confirmNotOut].forEach((button) => {
    button.disabled = !state.activeAppeal;
  });
}

function statusClass(status) {
  if (status === "OUT") return "out";
  if (status === "NOT_OUT") return "not-out";
  if (status === "PROCESSING") return "processing";
  return "waiting";
}

function statusText(status) {
  if (status === "PROCESSING") return "Processing review";
  return "Waiting for appeal";
}

function formatPercent(value) {
  return value === null || value === undefined ? "--" : `${Math.round(Number(value) * 100)}%`;
}

function formatPoint(point) {
  if (!point) return "--";
  return `${Math.round(point.x)}, ${Math.round(point.y)}`;
}

function drawTrajectory(decision) {
  const canvas = els.canvas;
  const ctx = canvas.getContext("2d");
  const width = canvas.width;
  const height = canvas.height;
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#101510";
  ctx.fillRect(0, 0, width, height);

  const pitch = { x: 50, y: 48, w: 400, h: 74 };
  ctx.fillStyle = "#245234";
  ctx.fillRect(pitch.x, pitch.y, pitch.w, pitch.h);
  ctx.strokeStyle = "#d8d0aa";
  ctx.lineWidth = 2;
  ctx.strokeRect(pitch.x, pitch.y, pitch.w, pitch.h);

  const wicket = decision.wicket_zone || { x: 412, y: 64, w: 18, h: 42 };
  ctx.fillStyle = "rgba(239, 159, 39, 0.35)";
  ctx.fillRect(wicket.x, wicket.y, wicket.w, wicket.h);
  label(ctx, "WICKET", wicket.x - 10, wicket.y - 8, "#ef9f27");

  const points = decision.trajectory || [];
  if (points.length > 1) {
    ctx.strokeStyle = "#f4f4f0";
    ctx.lineWidth = 3;
    ctx.beginPath();
    points.forEach((point, index) => {
      if (index === 0) ctx.moveTo(point.x, point.y);
      else ctx.lineTo(point.x, point.y);
    });
    ctx.stroke();
  }

  if (decision.bounce_point) {
    ctx.fillStyle = "#ffd45c";
    ctx.beginPath();
    ctx.arc(decision.bounce_point.x, decision.bounce_point.y, 5, 0, Math.PI * 2);
    ctx.fill();
    label(ctx, "BOUNCE", decision.bounce_point.x + 8, decision.bounce_point.y - 8, "#ffd45c");
  }

  if (decision.impact_point) {
    const p = decision.impact_point;
    ctx.strokeStyle = "#e24b4a";
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(p.x - 7, p.y - 7);
    ctx.lineTo(p.x + 7, p.y + 7);
    ctx.moveTo(p.x + 7, p.y - 7);
    ctx.lineTo(p.x - 7, p.y + 7);
    ctx.stroke();
    label(ctx, "IMPACT", p.x + 10, p.y + 14, "#e24b4a");
  }

  if (decision.predicted_extension?.length > 1) {
    ctx.strokeStyle = "#37b7d8";
    ctx.setLineDash([8, 6]);
    ctx.lineWidth = 2;
    ctx.beginPath();
    decision.predicted_extension.forEach((point, index) => {
      if (index === 0) ctx.moveTo(point.x, point.y);
      else ctx.lineTo(point.x, point.y);
    });
    ctx.stroke();
    ctx.setLineDash([]);
  }
}

function label(ctx, text, x, y, color) {
  ctx.fillStyle = color;
  ctx.font = "10px Consolas, monospace";
  ctx.fillText(text, x, y);
}

async function requestReview() {
  try {
    const response = window.drs?.requestReview
      ? await window.drs.requestReview({ camera_ids: [1, 2] })
      : await jsonFetch("/api/appeal/request", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ camera_ids: [1, 2] }),
        });
    renderDecision(response.decision || response);
    drawTrajectory(response.decision || response);
    log("Review requested.");
  } catch (error) {
    log(`Review request failed: ${error.message}`);
  }
}

async function confirmDecision(outcome) {
  try {
    const decision = await jsonFetch("/api/decision/confirm", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ outcome }),
    });
    renderDecision(decision);
    drawTrajectory(decision);
    log(`Decision confirmed: ${outcome}`);
  } catch (error) {
    log(`Confirm failed: ${error.message}`);
  }
}

els.requestReview.addEventListener("click", requestReview);
els.confirmOut.addEventListener("click", () => confirmDecision("OUT"));
els.confirmNotOut.addEventListener("click", () => confirmDecision("NOT_OUT"));

window.drs?.onDecision((decision) => {
  renderDecision(decision);
  drawTrajectory(decision);
});

refreshHealth();
refreshCameraStatus();
refreshDecision();
refreshCameraFrames();
setInterval(refreshCameraFrames, 80);
setInterval(refreshDecision, 300);
setInterval(refreshCameraStatus, 2000);
setInterval(refreshHealth, 5000);
