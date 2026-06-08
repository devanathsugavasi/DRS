const API_BASE = "http://localhost:8765";

const MARKER_ORDER = [
  ["off_stump", "Off stump"],
  ["middle_stump", "Middle stump"],
  ["leg_stump", "Leg stump"],
  ["bowling_crease", "Bowling crease"],
  ["popping_crease", "Popping crease"],
];

const MARKER_COLORS = {
  off_stump: "#ffd45c",
  middle_stump: "#f4f4f0",
  leg_stump: "#37b7d8",
  bowling_crease: "#ef9f27",
  popping_crease: "#e24b4a",
};

const state = {
  cameraId: 1,
  activeMarker: "off_stump",
  markers: {},
  imageSize: [0, 0],
  profile: null,
  history: [],
};

const els = {
  markerList: document.getElementById("marker-list"),
  canvas: document.getElementById("marker-canvas"),
  canvasWrap: document.getElementById("canvas-wrap"),
  canvasEmpty: document.getElementById("canvas-empty"),
  image: document.getElementById("calibration-image"),
  activeLabel: document.getElementById("active-marker-label"),
  progress: document.getElementById("marker-progress"),
  saveStatus: document.getElementById("save-status"),
  iccDimensions: document.getElementById("icc-dimensions"),
  savedProfiles: document.getElementById("saved-profiles"),
  captureFrame: document.getElementById("capture-frame"),
  uploadSnapshot: document.getElementById("upload-snapshot"),
  undoMarker: document.getElementById("undo-marker"),
  clearMarkers: document.getElementById("clear-markers"),
  saveCalibration: document.getElementById("save-calibration"),
  backDashboard: document.getElementById("back-dashboard"),
};

async function jsonFetch(route, options = {}) {
  const response = await fetch(`${API_BASE}${route}`, options);
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || `${response.status} ${response.statusText}`);
  }
  return response.json();
}

function renderMarkerList() {
  els.markerList.innerHTML = MARKER_ORDER.map(([key, label]) => {
    const point = state.markers[key];
    const done = Boolean(point);
    const active = state.activeMarker === key;
    const coords = done ? `${Math.round(point.x)}, ${Math.round(point.y)}` : "Not marked";
    return `<button type="button" class="marker-item ${active ? "active" : ""} ${done ? "done" : ""}" data-marker="${key}">
      <strong>${label}</strong>
      <span>${coords}</span>
    </button>`;
  }).join("");
  els.markerList.querySelectorAll("[data-marker]").forEach((button) => {
    button.addEventListener("click", () => {
      state.activeMarker = button.dataset.marker;
      renderMarkerList();
      updateToolbar();
    });
  });
}

function updateToolbar() {
  const label = MARKER_ORDER.find(([key]) => key === state.activeMarker)?.[1] || state.activeMarker;
  const done = Object.keys(state.markers).length;
  els.activeLabel.textContent = `Active: ${label}`;
  els.progress.textContent = `${done} / ${MARKER_ORDER.length} marked`;
}

function resizeCanvas() {
  const rect = els.canvasWrap.getBoundingClientRect();
  els.canvas.width = Math.max(320, Math.floor(rect.width));
  els.canvas.height = Math.max(280, Math.floor(rect.height));
  drawCanvas();
}

function drawCanvas() {
  const ctx = els.canvas.getContext("2d");
  const width = els.canvas.width;
  const height = els.canvas.height;
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#101510";
  ctx.fillRect(0, 0, width, height);

  if (els.image.src && !els.image.hidden) {
    const scale = Math.min(width / els.image.naturalWidth, height / els.image.naturalHeight);
    const drawW = els.image.naturalWidth * scale;
    const drawH = els.image.naturalHeight * scale;
    const offsetX = (width - drawW) / 2;
    const offsetY = (height - drawH) / 2;
    ctx.drawImage(els.image, offsetX, offsetY, drawW, drawH);
    state.drawTransform = { scale, offsetX, offsetY };
  } else {
    state.drawTransform = null;
  }

  MARKER_ORDER.forEach(([key, label]) => {
    const point = state.markers[key];
    if (!point || !state.drawTransform) return;
    const x = state.drawTransform.offsetX + point.x * state.drawTransform.scale;
    const y = state.drawTransform.offsetY + point.y * state.drawTransform.scale;
    ctx.fillStyle = MARKER_COLORS[key];
    ctx.beginPath();
    ctx.arc(x, y, 7, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = "#f4f4f0";
    ctx.font = "12px Consolas, monospace";
    ctx.fillText(label, x + 10, y - 8);
  });

  els.canvasEmpty.hidden = Boolean(els.image.src && !els.image.hidden);
}

function imageToMarkerCoords(clientX, clientY) {
  if (!state.drawTransform || !els.image.naturalWidth) return null;
  const rect = els.canvas.getBoundingClientRect();
  const x = clientX - rect.left;
  const y = clientY - rect.top;
  const rawX = (x - state.drawTransform.offsetX) / state.drawTransform.scale;
  const rawY = (y - state.drawTransform.offsetY) / state.drawTransform.scale;
  if (rawX < 0 || rawY < 0 || rawX > els.image.naturalWidth || rawY > els.image.naturalHeight) {
    return null;
  }
  return { x: rawX, y: rawY };
}

function setMarker(point) {
  state.history.push({ key: state.activeMarker, previous: state.markers[state.activeMarker] || null });
  state.markers[state.activeMarker] = { x: point.x, y: point.y };
  const currentIndex = MARKER_ORDER.findIndex(([key]) => key === state.activeMarker);
  if (currentIndex >= 0 && currentIndex < MARKER_ORDER.length - 1) {
    state.activeMarker = MARKER_ORDER[currentIndex + 1][0];
  }
  renderMarkerList();
  updateToolbar();
  drawCanvas();
}

async function loadDefaultProfile() {
  const profile = await jsonFetch("/api/calibration/default-profile");
  els.iccDimensions.innerHTML = Object.entries(profile.world_dimensions).map(([key, value]) => {
    const label = key.replaceAll("_", " ");
    return `<div><dt>${label}</dt><dd>${value}</dd></div>`;
  }).join("");
}

async function refreshSavedProfiles() {
  const status = await jsonFetch("/api/calibration/status");
  if (!status.camera_ids?.length) {
    els.savedProfiles.innerHTML = "<li>No saved camera profiles</li>";
    return;
  }
  els.savedProfiles.innerHTML = status.camera_ids.map((id) => `<li class="good">Camera ${id} calibrated</li>`).join("");
}

async function loadCameraProfile(cameraId) {
  state.cameraId = cameraId;
  state.markers = {};
  state.history = [];
  try {
    const profile = await jsonFetch(`/api/calibration/cameras/${cameraId}`);
    state.profile = profile;
    state.markers = { ...profile.markers };
    state.imageSize = profile.image_size;
    if (profile.snapshot_available) {
      await loadSnapshot(`${API_BASE}${profile.snapshot_url}?t=${Date.now()}`);
    }
    els.saveStatus.textContent = `Loaded saved profile for camera ${cameraId}`;
    els.saveStatus.className = "save-status good";
  } catch (_error) {
    state.profile = null;
    els.image.removeAttribute("src");
    els.image.hidden = true;
    els.saveStatus.textContent = `Camera ${cameraId} not calibrated yet`;
    els.saveStatus.className = "save-status warn";
    drawCanvas();
  }
  state.activeMarker = "off_stump";
  renderMarkerList();
  updateToolbar();
}

async function loadSnapshot(url) {
  return new Promise((resolve, reject) => {
    els.image.onload = () => {
      els.image.hidden = false;
      state.imageSize = [els.image.naturalWidth, els.image.naturalHeight];
      resizeCanvas();
      resolve();
    };
    els.image.onerror = reject;
    els.image.src = url;
  });
}

async function captureFrame() {
  const result = await jsonFetch(`/api/calibration/cameras/${state.cameraId}/capture`, { method: "POST" });
  await loadSnapshot(`${API_BASE}${result.snapshot_url}?t=${Date.now()}`);
  els.saveStatus.textContent = `Captured frame for camera ${state.cameraId}`;
  els.saveStatus.className = "save-status";
}

async function uploadSnapshot(file) {
  const form = new FormData();
  form.append("file", file);
  const response = await fetch(`${API_BASE}/api/calibration/cameras/${state.cameraId}/snapshot`, {
    method: "POST",
    body: form,
  });
  if (!response.ok) throw new Error("Snapshot upload failed");
  await loadSnapshot(`${API_BASE}/api/calibration/cameras/${state.cameraId}/snapshot?t=${Date.now()}`);
}

async function saveCalibration() {
  const missing = MARKER_ORDER.map(([key]) => key).filter((key) => !state.markers[key]);
  if (missing.length) {
    els.saveStatus.textContent = `Missing markers: ${missing.join(", ")}`;
    els.saveStatus.className = "save-status warn";
    return;
  }
  if (!state.imageSize[0] || !state.imageSize[1]) {
    els.saveStatus.textContent = "Capture or upload a frame before saving";
    els.saveStatus.className = "save-status warn";
    return;
  }
  const result = await jsonFetch(`/api/calibration/cameras/${state.cameraId}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ markers: state.markers, image_size: state.imageSize }),
  });
  state.profile = result.profile;
  els.saveStatus.textContent = `Saved camera ${state.cameraId} · homography error ${result.profile.homography_error_cm} cm`;
  els.saveStatus.className = "save-status good";
  await refreshSavedProfiles();
}

els.canvas.addEventListener("click", (event) => {
  const point = imageToMarkerCoords(event.clientX, event.clientY);
  if (!point) return;
  setMarker(point);
});

els.captureFrame.addEventListener("click", () => captureFrame().catch((error) => {
  els.saveStatus.textContent = error.message;
  els.saveStatus.className = "save-status warn";
}));

els.uploadSnapshot.addEventListener("change", (event) => {
  const file = event.target.files?.[0];
  if (!file) return;
  uploadSnapshot(file).catch((error) => {
    els.saveStatus.textContent = error.message;
    els.saveStatus.className = "save-status warn";
  });
  event.target.value = "";
});

els.undoMarker.addEventListener("click", () => {
  const last = state.history.pop();
  if (!last) return;
  if (last.previous) state.markers[last.key] = last.previous;
  else delete state.markers[last.key];
  state.activeMarker = last.key;
  renderMarkerList();
  updateToolbar();
  drawCanvas();
});

els.clearMarkers.addEventListener("click", () => {
  state.markers = {};
  state.history = [];
  state.activeMarker = "off_stump";
  renderMarkerList();
  updateToolbar();
  drawCanvas();
});

els.saveCalibration.addEventListener("click", () => saveCalibration().catch((error) => {
  els.saveStatus.textContent = error.message;
  els.saveStatus.className = "save-status warn";
}));

els.backDashboard.addEventListener("click", () => {
  window.location.href = "index.html";
});

document.querySelectorAll(".camera-btn").forEach((button) => {
  button.addEventListener("click", () => {
    document.querySelectorAll(".camera-btn").forEach((item) => item.classList.remove("active"));
    button.classList.add("active");
    loadCameraProfile(Number(button.dataset.camera));
  });
});

window.addEventListener("resize", resizeCanvas);

renderMarkerList();
updateToolbar();
loadDefaultProfile();
refreshSavedProfiles();
loadCameraProfile(1);
