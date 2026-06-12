import * as THREE from "../node_modules/three/build/three.module.js";
import { OrbitControls } from "../node_modules/three/examples/jsm/controls/OrbitControls.js";
import { CalibrationModal } from "./components/CalibrationModal.js";
import { DRSAnimationSequencer } from "./components/DRSAnimationSequencer.js";
import { ResultsPanel } from "./components/ResultsPanel.js";
import { StatusPanel } from "./components/StatusPanel.js";
import { TestingPanel } from "./components/TestingPanel.js";

const API_BASE = "http://localhost:8765";
const WS_BASE = "ws://localhost:8765";
const MAX_CAMERAS = 6;

const state = {
  decision: null,
  cameras: [],
  mode: { id: "visible", label: "Mode A - visible-spectrum approximation" },
  activeAppeal: false,
  replayFrame: 0,
  scene: null,
  replayTimer: null,
  panelMode: "live",
  testingPanel: null,
  statusPanel: null,
  resultsPanel: null,
  animationSequencer: null,
  calibrationModal: null,
  activeVideoInfo: null,
};

const timers = {};

const els = {
  engineState: document.getElementById("engine-state"),
  liveIndicator: document.getElementById("live-indicator"),
  modeBanner: document.getElementById("mode-banner"),
  modeToggle: document.getElementById("mode-toggle"),
  leftPanelTitle: document.getElementById("left-panel-title"),
  cameraPills: document.getElementById("camera-pills"),
  cameraCount: document.getElementById("camera-count"),
  cameraGrid: document.getElementById("camera-grid"),
  badge: document.getElementById("decision-badge"),
  title: document.getElementById("decision-title"),
  overlay: document.getElementById("broadcast-overlay"),
  overall: document.getElementById("overall-confidence"),
  impact: document.getElementById("impact-location"),
  wicket: document.getElementById("wicket-zone"),
  speed: document.getElementById("ball-speed"),
  explanation: document.getElementById("decision-explanation"),
  trajectoryStatus: document.getElementById("trajectory-status"),
  sceneHost: document.getElementById("trajectory-scene"),
  timeline: document.getElementById("decision-timeline"),
  confidenceBreakdown: document.getElementById("confidence-breakdown"),
  hotspotMode: document.getElementById("hotspot-mode"),
  hotspotView: document.getElementById("hotspot-view"),
  ultraedge: document.getElementById("ultraedge-canvas"),
  healthGrid: document.getElementById("health-grid"),
  history: document.getElementById("review-history"),
  frameTimeline: document.getElementById("frame-timeline"),
  frameLabel: document.getElementById("frame-label"),
  requestReview: document.getElementById("request-review"),
  confirmOut: document.getElementById("confirm-out"),
  confirmNotOut: document.getElementById("confirm-not-out"),
  calibrationButton: document.getElementById("calibration-button"),
  replayTrajectory: document.getElementById("replay-trajectory"),
  resetCamera: document.getElementById("reset-camera"),
  replayBack: document.getElementById("replay-back"),
  replayForward: document.getElementById("replay-forward"),
  replayPlay: document.getElementById("replay-play"),
  replayPause: document.getElementById("replay-pause"),
  replaySpeed: document.getElementById("replay-speed"),
  replayExport: document.getElementById("replay-export"),
  testingButton: document.getElementById("testing-platform-button"),
  testingDialog: document.getElementById("testing-platform-dialog"),
  testingFrame: document.getElementById("testing-platform-frame"),
  closeTesting: document.getElementById("close-testing-platform"),
  calibrationModalRoot: document.getElementById("calibration-modal-root"),
};

async function jsonFetch(route, options = {}) {
  const response = await fetch(`${API_BASE}${route}`, options);
  if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
  return response.json();
}

function setEngineOnline(online) {
  els.liveIndicator.classList.toggle("offline", !online);
  els.liveIndicator.querySelector("span").textContent = online ? "Live" : "Offline";
}

async function refreshHealth() {
  try {
    const health = window.drs?.getHealth ? await window.drs.getHealth() : await jsonFetch("/api/health");
    setEngineOnline(true);
    els.engineState.textContent = `Engine ${health.status || "ok"} | ${health.active_model_name || "model"} | ${formatDuration(health.uptime_seconds)}`;
  } catch {
    setEngineOnline(false);
    els.engineState.textContent = "Engine offline";
  }
}

async function refreshSystemHealth() {
  try {
    const health = window.drs?.getSystemHealth ? await window.drs.getSystemHealth() : await jsonFetch("/api/system/health");
    els.healthGrid.innerHTML = [
      ["CPU", pct(health.cpu_percent)],
      ["RAM", pct(health.ram_percent)],
      ["GPU", health.gpu?.available ? pct(health.gpu.percent) : "Telemetry n/a"],
      ["FPS", Object.values(health.camera_fps || {}).map((value) => Number(value).toFixed(1)).join(" / ") || "--"],
      ["Drops", sumValues(health.frame_drops)],
      ["Latency", `${health.latency_ms} ms`],
      ["Calibration", health.calibration?.readiness || "missing"],
      ["Tracking", trackingHealthLabel(health.camera_health)],
      ["Storage", `${health.storage?.free_gb ?? "--"} GB free`],
      ["Network", health.network?.status || "--"],
    ].map(([label, value]) => `<div><span>${label}</span><strong>${value}</strong></div>`).join("");
  } catch {
    els.healthGrid.innerHTML = `<div><span>Health</span><strong>Offline</strong></div>`;
  }
}

async function refreshCameraStatus() {
  try {
    const payload = await jsonFetch("/api/cameras/fps");
    state.cameras = payload.cameras || [];
    state.mode = payload.mode || state.mode;
    renderMode();
    renderCameraGrid();
    refreshCameraFrames();
  } catch {
    state.cameras = [];
    renderCameraGrid();
  }
}

function renderMode() {
  els.modeBanner.textContent = state.mode.label;
  els.modeBanner.classList.toggle("thermal", state.mode.id === "thermal_demo");
  els.hotspotMode.textContent = state.mode.id === "thermal_demo" ? "Demo overlay - simulated" : "Visible-spectrum approximation";
  els.hotspotView.textContent = state.mode.id === "thermal_demo"
    ? "Presentation heat colors are simulated and explicitly not real thermal data."
    : "Mode A uses frame differencing and motion-energy approximation.";
  els.hotspotView.classList.toggle("thermal", state.mode.id === "thermal_demo");
}

function renderCameraGrid() {
  if (state.panelMode !== "live") return;
  const connected = state.cameras.filter((camera) => camera.connected);
  els.cameraCount.textContent = `${connected.length} / ${MAX_CAMERAS} connected`;
  els.cameraPills.innerHTML = state.cameras.map((camera) => (
    `<span class="camera-pill ${camera.status}">Cam ${camera.id} | ${Number(camera.fps || 0).toFixed(1)} fps</span>`
  )).join("");

  const panels = [];
  const slots = [...state.cameras];
  while (slots.length < MAX_CAMERAS) slots.push(null);
  slots.slice(0, MAX_CAMERAS).forEach((camera, index) => {
    const cameraId = camera?.id ?? index;
    if (camera?.connected) {
      panels.push(`
        <article class="camera-panel ${camera.status}" data-camera-id="${cameraId}">
          <img id="camera-${cameraId}" alt="Camera ${cameraId} feed" />
          <div class="camera-placeholder">Waiting for feed</div>
          <div class="camera-label">Camera ${cameraId}</div>
          <div class="camera-fps">${Number(camera.fps || 0).toFixed(1)} fps</div>
        </article>
      `);
    } else {
      panels.push(renderAnalysisPanel(index + 1, connected.length));
    }
  });
  els.cameraGrid.className = `camera-grid count-${Math.max(1, connected.length)}`;
  els.cameraGrid.innerHTML = panels.join("");
  els.cameraGrid.querySelectorAll(".camera-panel img").forEach((img) => {
    img.addEventListener("load", () => {
      img.nextElementSibling.hidden = true;
      img.style.opacity = "1";
    });
    img.addEventListener("error", () => {
      img.nextElementSibling.hidden = false;
      img.style.opacity = "0";
    });
  });
}

function renderAnalysisPanel(slot, connectedCount) {
  const labels = ["Motion energy", "Calibration quality", "Sync quality", "Prediction volume", "Model confidence", "Replay buffer"];
  const label = labels[(slot - 1) % labels.length];
  const value = connectedCount <= 2 ? `${68 + slot * 3}%` : "Standby";
  return `
    <article class="analysis-tile">
      <span>Analysis panel</span>
      <strong>${label}</strong>
      <div class="mini-meter"><i style="width:${connectedCount <= 2 ? 42 + slot * 8 : 18}%"></i></div>
      <small>${value}</small>
    </article>
  `;
}

function refreshCameraFrames() {
  if (state.panelMode !== "live") return;
  const stamp = Date.now();
  state.cameras.filter((camera) => camera.connected).forEach((camera) => {
    const img = document.getElementById(`camera-${camera.id}`);
    if (img) img.src = `${API_BASE}/api/live/${camera.id}.jpg?t=${stamp}`;
  });
}

function renderLiveFrames(frames) {
  Object.entries(frames || {}).forEach(([cameraId, frame]) => {
    const img = document.getElementById(`camera-${cameraId}`);
    if (img && frame.jpeg_base64) {
      img.src = `data:image/jpeg;base64,${frame.jpeg_base64}`;
    }
  });
}

async function refreshDecision() {
  try {
    const decision = await jsonFetch("/api/decision/current");
    renderDecision(decision);
  } catch {}
}

function renderDecision(decision) {
  state.decision = decision;
  const status = decision.status || "WAITING";
  state.activeAppeal = status !== "WAITING";
  els.badge.className = `badge ${statusClass(status)}`;
  els.badge.textContent = displayStatus(status);
  els.title.textContent = decision.outcome || statusText(status);
  els.overlay.className = `broadcast-overlay ${statusClass(status)}`;
  els.overlay.textContent = broadcastText(status, decision);
  els.overall.textContent = pct(decision.overall_confidence ?? decision.ball_confidence);
  els.impact.textContent = formatPoint(decision.impact_marker || decision.impact_point);
  els.wicket.textContent = decision.wicket_zone_status || "--";
  els.speed.textContent = decision.ball_speed_kmh ? `${Number(decision.ball_speed_kmh).toFixed(1)} km/h` : "--";
  els.explanation.textContent = decision.explanation || "Awaiting appeal sequence.";
  els.trajectoryStatus.textContent = decision.trajectory?.length ? `${decision.trajectory.length} tracked 3D points` : "Waiting for review data";
  els.confirmOut.disabled = !state.activeAppeal;
  els.confirmNotOut.disabled = !state.activeAppeal;
  renderTimeline(decision.timeline || []);
  renderConfidence(decision);
  updateTrajectory(decision);
  drawUltraEdge(decision);
  renderHotspot(decision);
}

function renderTimeline(items) {
  const fallback = ["Appeal", "Ball Detected", "Bounce Detected", "Impact Detected", "Wicket Predicted", "Decision Generated", "Umpire Call"];
  const source = items.length ? items : fallback.map((label, index) => ({ label, status: index === 0 ? "active" : "pending" }));
  els.timeline.innerHTML = source.map((item) => `
    <div class="timeline-row ${item.status}">
      <i></i><span>${item.label}</span>
    </div>
  `).join("");
}

function renderConfidence(decision) {
  const rows = [
    ["Ball detection", decision.ball_confidence],
    ["Tracking", decision.tracking_confidence],
    ["Calibration", decision.calibration_confidence],
    ["Prediction", decision.prediction_confidence],
    ["Model", decision.model_confidence],
  ];
  els.confidenceBreakdown.innerHTML = rows.map(([label, value]) => `
    <div class="confidence-row">
      <span>${label}</span><strong>${pct(value)}</strong>
      <div><i style="width:${Math.round(Number(value || 0) * 100)}%"></i></div>
    </div>
  `).join("");
}

function initDashboardModules() {
  state.statusPanel = new StatusPanel(els.leftPanelTitle, els.cameraCount, els.cameraGrid);
  state.resultsPanel = new ResultsPanel(els);
  state.animationSequencer = new DRSAnimationSequencer({
    overlay: els.overlay,
    title: els.title,
    frameLabel: els.frameLabel,
    replayTimeline: els.frameTimeline,
  });
  state.calibrationModal = new CalibrationModal(els.calibrationModalRoot);
}

async function showTestingPanel() {
  state.panelMode = "testing";
  els.leftPanelTitle.textContent = "Testing";
  els.cameraCount.textContent = "Quick Test";
  state.testingPanel = new TestingPanel(els.cameraGrid, {
    onVideoInfo: (videoInfo) => {
      state.activeVideoInfo = videoInfo;
      state.statusPanel.testing(videoInfo, { step: "Waiting for analysis", percent: 0 });
    },
    onProgress: (progress) => {
      state.statusPanel.testing(state.activeVideoInfo, progress);
    },
    onResults: (results) => {
      renderAnalysisResults(results);
    },
  });
  await state.testingPanel.render();
}

function renderAnalysisResults(results) {
  state.panelMode = "results";
  state.statusPanel.summary(results);
  state.resultsPanel.render(results);
  const decision = dashboardDecisionFromResults(results);
  renderDecision(decision);
  state.animationSequencer.play(results);
}

function dashboardDecisionFromResults(results) {
  const verdict = results.decision?.verdict || "UMPIRES_CALL";
  const trajectory = results.trajectory?.points || [];
  return {
    status: verdict === "NOT_OUT" ? "NOT_OUT" : verdict === "OUT" ? "OUT" : "PROCESSING",
    outcome: verdict.replace("_", " "),
    decision: verdict.replace("_", " "),
    overall_confidence: results.decision?.confidence,
    ball_confidence: results.ball_tracking?.avg_confidence,
    tracking_confidence: results.ball_tracking?.detection_rate,
    calibration_confidence: results.lbw_gates?.pitching?.confidence,
    prediction_confidence: results.lbw_gates?.wickets?.confidence,
    model_confidence: results.ball_tracking?.avg_confidence,
    impact_point: results.trajectory?.impact_point,
    impact_marker: results.trajectory?.impact_point,
    bounce_point: results.trajectory?.bounce_point,
    wicket_zone_status: results.lbw_gates?.wickets?.result || "--",
    ball_speed_kmh: results.summary?.ball_speed_kmh,
    trajectory,
    predicted_extension: trajectory.slice(-3),
    wicket_prediction: results.trajectory?.predicted_stumps,
    timeline: [
      { label: "Release", status: "complete" },
      { label: "Bounce", status: results.trajectory?.bounce_point ? "complete" : "pending" },
      { label: "Pad Impact", status: results.trajectory?.impact_point ? "complete" : "pending" },
      { label: "Stump Line", status: "complete" },
    ],
    explanation: results.decision?.explanation,
  };
}

async function requestReview() {
  const cameraIds = state.cameras.filter((camera) => camera.connected).map((camera) => camera.id);
  const response = window.drs?.requestReview
    ? await window.drs.requestReview({ camera_ids: cameraIds })
    : await jsonFetch("/api/appeal/request", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ camera_ids: cameraIds }),
      });
  renderDecision(response.decision || response);
}

async function confirmDecision(outcome) {
  const decision = await jsonFetch("/api/decision/confirm", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ outcome }),
  });
  renderDecision(decision);
  refreshReviews();
}

async function refreshReviews() {
  try {
    const payload = window.drs?.getReviews ? await window.drs.getReviews() : await jsonFetch("/api/reviews");
    els.history.innerHTML = (payload.reviews || []).map((review) => `
      <button type="button" class="review-row" data-review-id="${review.id}">
        <span>${new Date(review.time).toLocaleTimeString()} | ${review.over}</span>
        <strong>${review.decision}</strong>
        <small>${pct(review.confidence)} | replay ready</small>
      </button>
    `).join("") || `<span class="muted">No stored reviews yet.</span>`;
    els.history.querySelectorAll(".review-row").forEach((button) => {
      button.addEventListener("click", async () => {
        const review = await jsonFetch(`/api/reviews/${button.dataset.reviewId}`);
        renderDecision({ ...state.decision, ...review, status: review.decision?.replaceAll(" ", "_") || "PROCESSING" });
      });
    });
  } catch {
    els.history.innerHTML = `<span class="muted">Review history unavailable.</span>`;
  }
}

function initThree() {
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x07100d);
  const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 100);
  camera.position.set(8, 7, 12);
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  els.sceneHost.appendChild(renderer.domElement);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.target.set(0, 0, 0.55);

  scene.add(new THREE.HemisphereLight(0xffffff, 0x163126, 1.4));
  const key = new THREE.DirectionalLight(0xffffff, 1.8);
  key.position.set(-4, 8, 6);
  scene.add(key);
  buildPitch(scene);

  state.scene = { scene, camera, renderer, controls, dynamic: new THREE.Group() };
  scene.add(state.scene.dynamic);
  resizeThree();
  renderer.setAnimationLoop(() => {
    controls.update();
    renderer.render(scene, camera);
  });
}

function buildPitch(scene) {
  const pitch = new THREE.Mesh(
    new THREE.BoxGeometry(20.12, 3.05, 0.04),
    new THREE.MeshStandardMaterial({ color: 0x8f7d55, roughness: 0.8 })
  );
  pitch.position.z = -0.02;
  scene.add(pitch);
  const turf = new THREE.GridHelper(24, 24, 0x2a6b49, 0x204532);
  turf.rotation.x = Math.PI / 2;
  turf.position.z = -0.04;
  scene.add(turf);
  const stumpMaterial = new THREE.MeshStandardMaterial({ color: 0xf2e6bd });
  [-0.23, 0, 0.23].forEach((y) => {
    const stump = new THREE.Mesh(new THREE.CylinderGeometry(0.035, 0.035, 0.72, 16), stumpMaterial);
    stump.rotation.x = Math.PI / 2;
    stump.position.set(7.1, y, 0.36);
    scene.add(stump);
  });
}

function updateTrajectory(decision) {
  if (!state.scene) return;
  const group = state.scene.dynamic;
  group.clear();
  const points = normalizeTrajectory(decision.trajectory || []);
  if (points.length > 1) {
    const geometry = new THREE.BufferGeometry().setFromPoints(points);
    group.add(new THREE.Line(geometry, new THREE.LineBasicMaterial({ color: 0xf8f7ef, linewidth: 4 })));
    addTube(group, points, 0x42d895, 0.035);
    addConfidenceVolumes(group, points, decision);
  }
  addMarker(group, normalizePoint(decision.bounce_point), 0xffd45c, "bounce");
  addMarker(group, normalizePoint(decision.impact_marker || decision.impact_point), 0xe24b4a, "impact");
  const predicted = normalizeTrajectory(decision.predicted_extension || []);
  if (predicted.length > 1) addTube(group, predicted, 0x37b7d8, 0.025);
  addWicketPrediction(group, decision.wicket_prediction);
}

function addTube(group, points, color, radius) {
  const curve = new THREE.CatmullRomCurve3(points);
  const tube = new THREE.Mesh(
    new THREE.TubeGeometry(curve, 64, radius, 12, false),
    new THREE.MeshStandardMaterial({ color, emissive: color, emissiveIntensity: 0.22 })
  );
  group.add(tube);
}

function addConfidenceVolumes(group, points) {
  points.forEach((point, index) => {
    const mesh = new THREE.Mesh(
      new THREE.SphereGeometry(0.18 + index * 0.018, 24, 12),
      new THREE.MeshStandardMaterial({ color: 0x37b7d8, transparent: true, opacity: 0.12, depthWrite: false })
    );
    mesh.position.copy(point);
    group.add(mesh);
  });
}

function addMarker(group, point, color) {
  if (!point) return;
  const marker = new THREE.Mesh(
    new THREE.SphereGeometry(0.14, 24, 16),
    new THREE.MeshStandardMaterial({ color, emissive: color, emissiveIntensity: 0.35 })
  );
  marker.position.copy(point);
  group.add(marker);
}

function addWicketPrediction(group, prediction) {
  const zone = new THREE.Mesh(
    new THREE.BoxGeometry(0.62, 0.72, 0.72),
    new THREE.MeshStandardMaterial({ color: 0xef9f27, transparent: true, opacity: 0.16 })
  );
  zone.position.set(7.1, 0, 0.36);
  group.add(zone);
  if (prediction?.collision) addMarker(group, normalizePoint(prediction.collision), 0xef9f27);
}

function normalizeTrajectory(points) {
  return points.map(normalizePoint).filter(Boolean);
}

function normalizePoint(point) {
  if (!point) return null;
  const x = Number(point.x);
  const y = Number(point.y);
  const z = Number(point.z ?? 0.2);
  if ([x, y, z].some((value) => Number.isNaN(value))) return null;
  return new THREE.Vector3(x, y, z);
}

function resizeThree() {
  if (!state.scene) return;
  const rect = els.sceneHost.getBoundingClientRect();
  state.scene.camera.aspect = rect.width / Math.max(1, rect.height);
  state.scene.camera.updateProjectionMatrix();
  state.scene.renderer.setSize(rect.width, rect.height, false);
}

function resetThreeCamera() {
  state.scene.camera.position.set(8, 7, 12);
  state.scene.controls.target.set(0, 0, 0.55);
  state.scene.controls.update();
}

function replayTrajectory() {
  clearInterval(state.replayTimer);
  state.replayFrame = 0;
  state.replayTimer = setInterval(() => {
    state.replayFrame = Math.min(100, state.replayFrame + 4);
    els.frameTimeline.value = String(state.replayFrame);
    els.frameLabel.textContent = `Frame ${state.replayFrame}`;
    if (state.replayFrame >= 100) clearInterval(state.replayTimer);
  }, 90);
}

async function replayControl(action, extra = {}) {
  const payload = await jsonFetch("/api/replay/control", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ action, ...extra }),
  });
  renderReplayState(payload);
}

function renderReplayState(payload) {
  if (!payload) return;
  const total = Math.max(1, Number(payload.total_frames || 100) - 1);
  els.frameTimeline.max = String(total);
  els.frameTimeline.value = String(Math.min(total, Number(payload.cursor || 0)));
  els.frameLabel.textContent = `Frame ${els.frameTimeline.value} | ${payload.playing ? "Playing" : "Paused"} | ${Number(payload.speed || 1)}x`;
}

async function exportReplay() {
  els.frameLabel.textContent = "Exporting replay...";
  const payload = await jsonFetch("/api/replay/export", { method: "POST" });
  els.frameLabel.textContent = `Exported: ${payload.path}`;
}

function drawUltraEdge(decision) {
  const ctx = els.ultraedge.getContext("2d");
  const { width, height } = els.ultraedge;
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#080d0f";
  ctx.fillRect(0, 0, width, height);
  const events = decision?.edge_analysis?.events || [];
  const peak = Number(decision?.edge_analysis?.edge_probability || 0);
  ctx.strokeStyle = "#37b7d8";
  ctx.beginPath();
  for (let x = 0; x < width; x += 1) {
    const phase = x / width;
    const spike = events.length && phase > 0.52 && phase < 0.66 ? peak * 40 : 0;
    const y = height / 2 + Math.sin(x * 0.08) * 10 + Math.sin(x * 0.21) * 4 - spike;
    if (x === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();
  if (peak > 0.2) {
    ctx.fillStyle = "#ef9f27";
    ctx.fillRect(width * 0.58, 18, 2, height - 36);
  }
}

function renderHotspot(decision) {
  const hotspot = decision?.hotspot_analysis || {};
  if (hotspot.contact_detected) {
    els.hotspotView.textContent = `Contact detected (${pct(hotspot.confidence)}) · ${hotspot.reason || "Optical-flow proxy"}`;
    els.hotspotView.classList.add("active");
  } else {
    els.hotspotView.textContent = hotspot.reason || "No contact heatmap yet";
    els.hotspotView.classList.remove("active");
  }
}

function connectChannel(channel) {
  const socket = new WebSocket(`${WS_BASE}/ws/${channel}`);
  socket.addEventListener("message", (event) => {
    try {
      const payload = JSON.parse(event.data);
      if (channel === "decision" && payload.decision) renderDecision(payload.decision);
      if (channel === "system" && payload.health) renderSystemPayload(payload.health);
      if (channel === "live" && payload.cameras) {
        state.cameras = payload.cameras;
        renderCameraGrid();
        renderLiveFrames(payload.frames);
      }
      if (channel === "trajectory" && payload.trajectory) {
        renderDecision({ ...state.decision, trajectory: payload.trajectory });
      }
      if (channel === "replay" && payload.replay) {
        renderReplayState(payload.replay);
      }
    } catch {}
  });
  socket.addEventListener("close", () => {
    setTimeout(() => connectChannel(channel), 2000);
  });
}

function connectWebSockets() {
  ["system", "decision", "trajectory", "replay", "live"].forEach(connectChannel);
}

function renderSystemPayload(health) {
  els.healthGrid.innerHTML = [
    ["CPU", pct(health.cpu_percent)],
    ["RAM", pct(health.ram_percent)],
    ["GPU", health.gpu?.available ? pct(health.gpu.percent) : "Telemetry n/a"],
    ["FPS", Object.values(health.camera_fps || {}).map((value) => Number(value).toFixed(1)).join(" / ") || "--"],
    ["Drops", sumValues(health.frame_drops)],
    ["Latency", `${health.latency_ms} ms`],
    ["Calibration", health.calibration?.readiness || "missing"],
    ["Tracking", trackingHealthLabel(health.camera_health)],
    ["Storage", `${health.storage?.free_gb ?? "--"} GB free`],
    ["Network", health.network?.status || "--"],
  ].map(([label, value]) => `<div><span>${label}</span><strong>${value}</strong></div>`).join("");
}

async function toggleMode() {
  const next = state.mode.id === "thermal_demo" ? "visible" : "thermal_demo";
  state.mode = window.drs?.setAnalysisMode
    ? await window.drs.setAnalysisMode({ mode: next })
    : await jsonFetch("/api/analysis-mode", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ mode: next }),
      });
  renderMode();
  refreshCameraFrames();
}

function statusClass(status) {
  if (status === "OUT") return "out";
  if (status === "NOT_OUT") return "not-out";
  if (status === "PROCESSING") return "processing";
  return "waiting";
}

function displayStatus(status) {
  return String(status).replaceAll("_", " ");
}

function statusText(status) {
  if (status === "OUT") return "OUT";
  if (status === "NOT_OUT") return "NOT OUT";
  return status === "PROCESSING" ? "Processing review" : "Waiting for appeal";
}

function broadcastText(status, decision) {
  if (status === "OUT") return "OUT";
  if (status === "NOT_OUT") return "NOT OUT";
  if (decision?.wicket_prediction?.umpire_call) return "UMPIRE'S CALL";
  return "WAITING";
}

function pct(value) {
  return value === null || value === undefined ? "--" : `${Math.round(Number(value) * 100)}%`;
}

function formatPoint(point) {
  if (!point) return "--";
  if (Array.isArray(point)) return point.map((value) => Number(value).toFixed(1)).join(", ");
  return `${Number(point.x).toFixed(1)}, ${Number(point.y).toFixed(1)}, ${Number(point.z ?? 0).toFixed(1)}`;
}

function formatDuration(seconds) {
  if (seconds === null || seconds === undefined) return "--";
  const mins = Math.floor(seconds / 60);
  const secs = seconds % 60;
  return mins ? `${mins}m ${secs}s` : `${secs}s`;
}

function sumValues(values) {
  return Object.values(values || {}).reduce((total, value) => total + Number(value || 0), 0);
}

function trackingHealthLabel(cameras) {
  if (!Array.isArray(cameras) || !cameras.length) return "--";
  const score = cameras.reduce((total, camera) => total + Number(camera.health_score || 0), 0) / cameras.length;
  if (score >= 0.8) return "high";
  if (score >= 0.55) return "medium";
  return "low";
}

els.requestReview.addEventListener("click", requestReview);
els.confirmOut.addEventListener("click", () => confirmDecision("OUT"));
els.confirmNotOut.addEventListener("click", () => confirmDecision("NOT_OUT"));
els.calibrationButton.addEventListener("click", () => { state.calibrationModal.open(); });
els.modeToggle.addEventListener("click", toggleMode);
els.replayTrajectory.addEventListener("click", replayTrajectory);
els.resetCamera.addEventListener("click", resetThreeCamera);
els.replayPlay.addEventListener("click", () => replayControl("play", { speed: Number(els.replaySpeed.value) }));
els.replayPause.addEventListener("click", () => replayControl("pause"));
els.replayBack.addEventListener("click", () => replayControl("step_back"));
els.replayForward.addEventListener("click", () => replayControl("step_forward"));
els.replaySpeed.addEventListener("change", () => replayControl("speed", { speed: Number(els.replaySpeed.value) }));
els.replayExport.addEventListener("click", () => {
  if (state.animationSequencer?.results) state.animationSequencer.exportMP4();
  else exportReplay();
});
els.frameTimeline.addEventListener("input", () => replayControl("seek", { frame_index: Number(els.frameTimeline.value) }));
els.testingButton.addEventListener("click", showTestingPanel);
els.closeTesting?.addEventListener("click", () => els.testingDialog.close());

window.drs?.onDecision((decision) => renderDecision(decision));
window.addEventListener("resize", resizeThree);
window.addEventListener("keydown", (event) => {
  if (event.target.matches("input, textarea")) return;
  const key = event.key.toLowerCase();
  if (key === "r") requestReview();
  if (key === "o" && state.activeAppeal) confirmDecision("OUT");
  if (key === "n" && state.activeAppeal) confirmDecision("NOT_OUT");
});

window.drs?.onStartupStatus?.((status) => {
  if (status?.testingPlatform?.status === "unavailable") {
    els.explanation.textContent = status.testingPlatform.message;
  }
});

initDashboardModules();
initThree();
connectWebSockets();
refreshHealth();
refreshSystemHealth();
refreshCameraStatus();
refreshDecision();
refreshReviews();
timers.health = setInterval(refreshHealth, 15000);
timers.system = setInterval(refreshSystemHealth, 15000);
timers.cameras = setInterval(refreshCameraStatus, 10000);
timers.frames = setInterval(refreshCameraFrames, 1000);
timers.decision = setInterval(refreshDecision, 5000);
