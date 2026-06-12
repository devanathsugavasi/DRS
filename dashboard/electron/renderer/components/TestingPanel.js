import { watchAnalysisJob, pollAnalysisJob } from "../hooks/useAnalysisJob.js";
import { loadCalibrationProfiles } from "../hooks/useCalibrationProfiles.js";

const API_BASE = "http://localhost:8765";
const STEPS = ["Extracting frames...", "Detecting ball...", "Tracking...", "Predicting trajectory...", "Running LBW analysis...", "Complete"];

export class TestingPanel {
  constructor(root, { onVideoInfo, onProgress, onResults }) {
    this.root = root;
    this.onVideoInfo = onVideoInfo;
    this.onProgress = onProgress;
    this.onResults = onResults;
    this.mode = "quick";
    this.videoFile = null;
    this.videoInfo = null;
    this.profileId = "";
    this.jobWatcher = null;
    this.pollStop = null;
  }

  async render() {
    this.root.className = "testing-panel fade-panel";
    this.root.innerHTML = this.template();
    this.bind();
    await this.loadProfiles();
  }

  template() {
    return `
      <div class="testing-tabs">
        <button type="button" class="active" data-tab="quick">Quick Test</button>
        <button type="button" data-tab="calibrated">Calibrated DRS</button>
      </div>
      <section class="testing-tab" id="quick-tab">
        <div class="warning-banner">Running without calibration. Results are approximate (+/-15cm accuracy).</div>
        ${this.uploadMarkup("quick-video", "Upload Video")}
        <button id="quick-analyze" type="button" disabled>Analyze</button>
        <div id="quick-results" class="inline-results"></div>
      </section>
      <section class="testing-tab" id="calibrated-tab" hidden>
        <label>Calibration Profile
          <select id="calibration-profile"></select>
        </label>
        <button id="open-calibration-from-testing" type="button">No profile - create new</button>
        ${this.uploadMarkup("calibrated-video", "Upload Video")}
        <button id="calibrated-analyze" type="button" disabled>Analyze with Calibration</button>
        <canvas id="inline-pitch-map" width="420" height="260"></canvas>
        <div id="calibrated-results" class="inline-results"></div>
      </section>
    `;
  }

  uploadMarkup(inputId, label) {
    return `
      <label class="drop-zone">
        <input id="${inputId}" type="file" accept=".mp4,.avi,.mov,.MTS,.mts" />
        <strong>${label}</strong>
        <span>Drop or select .mp4, .avi, .mov, .MTS</span>
      </label>
      <section class="uploaded-video" hidden>
        <video muted playsinline></video>
        <div>
          <strong class="video-name"></strong>
          <span class="video-meta"></span>
        </div>
      </section>
      <div class="analysis-progress"><i style="width:0%"></i></div>
      <span class="analysis-status">Waiting for upload</span>
    `;
  }

  bind() {
    this.root.querySelectorAll("[data-tab]").forEach((button) => {
      button.addEventListener("click", () => this.switchTab(button.dataset.tab));
    });
    this.root.querySelector("#quick-video").addEventListener("change", (event) => this.loadVideo(event));
    this.root.querySelector("#calibrated-video").addEventListener("change", (event) => this.loadVideo(event));
    this.root.querySelector("#quick-analyze").addEventListener("click", () => this.analyze(false));
    this.root.querySelector("#calibrated-analyze").addEventListener("click", () => this.analyze(true));
    this.root.querySelector("#calibration-profile").addEventListener("change", (event) => {
      this.profileId = event.target.value;
      this.updateButtons();
    });
    this.root.querySelector("#open-calibration-from-testing").addEventListener("click", () => {
      document.getElementById("calibration-button")?.click();
    });
  }

  switchTab(mode) {
    this.mode = mode;
    this.root.querySelectorAll("[data-tab]").forEach((button) => button.classList.toggle("active", button.dataset.tab === mode));
    this.root.querySelector("#quick-tab").hidden = mode !== "quick";
    this.root.querySelector("#calibrated-tab").hidden = mode !== "calibrated";
    this.updateButtons();
  }

  async loadProfiles() {
    const select = this.root.querySelector("#calibration-profile");
    try {
      const profiles = await loadCalibrationProfiles();
      select.innerHTML = `<option value="">No profile - create new</option>${profiles.map((profile) => (
        `<option value="${profile.id}">${profile.name} | ${Number(profile.rms_error_px || 0).toFixed(1)}px</option>`
      )).join("")}`;
    } catch {
      select.innerHTML = `<option value="">Backend offline</option>`;
    }
  }

  loadVideo(event) {
    const file = event.target.files?.[0];
    if (!file) return;
    this.videoFile = file;
    const url = URL.createObjectURL(file);
    const tab = event.target.closest(".testing-tab");
    const video = tab.querySelector("video");
    const holder = tab.querySelector(".uploaded-video");
    video.src = url;
    video.addEventListener("loadedmetadata", () => {
      this.videoInfo = {
        name: file.name,
        duration: video.duration,
        resolution: `${video.videoWidth || "--"}x${video.videoHeight || "--"}`,
        fps: "--",
        totalFrames: video.duration ? Math.round(video.duration * 30) : "--",
      };
      tab.querySelector(".video-name").textContent = file.name;
      tab.querySelector(".video-meta").textContent = `${formatDuration(video.duration)} | ${this.videoInfo.resolution}`;
      holder.hidden = false;
      this.onVideoInfo?.(this.videoInfo);
      this.updateButtons();
    }, { once: true });
  }

  updateButtons() {
    this.root.querySelector("#quick-analyze").disabled = !this.videoFile;
    this.root.querySelector("#calibrated-analyze").disabled = !this.videoFile || !this.profileId;
  }

  async analyze(calibrated) {
    if (!this.videoFile) return;
    this.setProgress({ step: STEPS[0], percent: 5 });
    const form = new FormData();
    form.append("video", this.videoFile);
    form.append("options_json", JSON.stringify({ edge_detection: true, replay_generation: true }));
    let route = "/api/analyze";
    if (calibrated) {
      route = "/api/analyze/calibrated";
      form.append("calibration_profile_id", this.profileId);
    }
    const response = await fetch(`${API_BASE}${route}`, { method: "POST", body: form });
    if (!response.ok) {
      this.setProgress({ step: `Analysis failed: ${response.status}`, percent: 0 });
      return;
    }
    const payload = await response.json();
    this.watchJob(payload.job_id);
  }

  async watchJob(jobId) {
    this.jobWatcher?.close();
    this.pollStop?.();
    this.jobWatcher = watchAnalysisJob(jobId, {
      onProgress: (payload) => this.setProgress(payload),
      onDecision: () => this.loadResults(jobId),
      onAnimation: () => this.loadResults(jobId),
      onError: (message) => this.setProgress({ step: message, percent: 0 }),
    });
    this.pollStop = await pollAnalysisJob(jobId, {
      onProgress: (payload) => this.setProgress(payload),
      onComplete: () => this.loadResults(jobId),
      onError: (message) => this.setProgress({ step: message, percent: 0 }),
    });
  }

  async loadResults(jobId) {
    const response = await fetch(`${API_BASE}/api/analyze/${jobId}/results`);
    if (!response.ok) return;
    const results = await response.json();
    this.renderInlineResults(results);
    this.onResults?.(results);
  }

  setProgress(progress) {
    const percent = Number(progress.percent ?? progress.progress ?? 0);
    this.root.querySelectorAll(".analysis-progress i").forEach((bar) => { bar.style.width = `${percent}%`; });
    this.root.querySelectorAll(".analysis-status").forEach((status) => { status.textContent = progress.step || progress.current_step || "Processing"; });
    this.onProgress?.(progress);
  }

  renderInlineResults(results) {
    const verdict = results.decision?.verdict || "UMPIRES_CALL";
    const host = this.root.querySelector(this.mode === "quick" ? "#quick-results" : "#calibrated-results");
    const gates = results.lbw_gates || {};
    host.innerHTML = `
      <section class="result-card ${verdict.toLowerCase()}">
        <strong>DECISION: ${verdict.replace("_", " ")}</strong>
        <span>Confidence: ${Math.round(Number(results.decision?.confidence || 0) * 100)}%</span>
        <span>Ball Speed: ${results.summary?.ball_speed_kmh || "--"} km/h</span>
        <span>Bounce Point: ${formatPoint(results.trajectory?.bounce_point)}</span>
        <span>Impact Height: ${gates.impact?.height_m ?? "--"}m</span>
        <hr />
        <span>Pitching: ${gates.pitching?.result || "--"} (${Math.round(Number(gates.pitching?.confidence || 0) * 100)}%)</span>
        <span>Impact: ${gates.impact?.result || "--"} (${Math.round(Number(gates.impact?.confidence || 0) * 100)}%)</span>
        <span>Wickets: ${gates.wickets?.result || "--"} (${Math.round(Number(gates.wickets?.confidence || 0) * 100)}%)</span>
      </section>
    `;
  }
}

function formatDuration(seconds) {
  if (!seconds || Number.isNaN(seconds)) return "--";
  const mins = Math.floor(seconds / 60);
  const secs = Math.round(seconds % 60).toString().padStart(2, "0");
  return `${mins}:${secs}`;
}

function formatPoint(point) {
  if (!point) return "--";
  return `${Number(point.x || 0).toFixed(1)}, ${Number(point.y || 0).toFixed(1)}, ${Number(point.z || 0).toFixed(1)}`;
}
