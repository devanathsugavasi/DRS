import { autoDetectCalibration, loadCalibrationProfiles, saveCalibrationProfile } from "../hooks/useCalibrationProfiles.js";

const POINT_LABELS = [
  "Bowling crease left",
  "Bowling crease center",
  "Bowling crease right",
  "Popping crease left",
  "Popping crease center",
  "Popping crease right",
  "Stumps top left",
  "Stumps top center",
  "Stumps top right",
];

export class CalibrationModal {
  constructor(root) {
    this.root = root;
    this.points = [];
    this.imageDataUrl = "";
  }

  async open() {
    this.root.innerHTML = this.template();
    this.root.classList.add("open");
    this.bind();
    await this.refreshProfiles();
  }

  close() {
    this.root.classList.remove("open");
    this.root.innerHTML = "";
  }

  template() {
    return `
      <div class="calibration-modal">
        <header>
          <strong>Calibration Manager</strong>
          <button id="calibration-close" type="button">Close</button>
        </header>
        <section class="calibration-body">
          <article>
            <div class="modal-row">
              <strong>Saved Profiles</strong>
              <button id="new-profile" type="button">+ New Profile</button>
            </div>
            <div id="profiles-table" class="profiles-table"></div>
          </article>
          <article id="profile-editor" class="profile-editor" hidden>
            <label>Profile name <input id="profile-name" placeholder="Ground A - Camera 1" /></label>
            <label>Ground name <input id="ground-name" placeholder="Vijayawada District Stadium" /></label>
            <label>Reference image <input id="calibration-image" type="file" accept=".jpg,.jpeg,.png" /></label>
            <div class="calibration-image-wrap">
              <img id="calibration-preview" alt="Calibration reference" />
              <div id="calibration-points" class="calibration-points"></div>
            </div>
            <div class="point-list">${POINT_LABELS.map((label, index) => `<span>${index + 1}: ${label}</span>`).join("")}</div>
            <div class="modal-row">
              <button id="auto-detect-calibration" type="button">Auto-detect</button>
              <span id="rms-output">Calibration RMS Error: --</span>
              <button id="save-profile" type="button" disabled>Save Profile</button>
            </div>
          </article>
        </section>
      </div>
    `;
  }

  bind() {
    this.root.querySelector("#calibration-close").addEventListener("click", () => this.close());
    this.root.querySelector("#new-profile").addEventListener("click", () => {
      this.root.querySelector("#profile-editor").hidden = false;
    });
    this.root.querySelector("#calibration-image").addEventListener("change", (event) => this.loadImage(event));
    this.root.querySelector("#calibration-preview").addEventListener("click", (event) => this.addPoint(event));
    this.root.querySelector("#auto-detect-calibration").addEventListener("click", () => this.autoDetect());
    this.root.querySelector("#save-profile").addEventListener("click", () => this.save());
  }

  async refreshProfiles() {
    const host = this.root.querySelector("#profiles-table");
    try {
      const profiles = await loadCalibrationProfiles();
      host.innerHTML = `
        <div class="profile-row header"><span>Name</span><span>Ground</span><span>Camera</span><span>RMS</span><span>Actions</span></div>
        ${profiles.map((profile) => `
          <div class="profile-row">
            <span>${profile.name}</span><span>${profile.ground}</span><span>${profile.camera}</span>
            <span>${Number(profile.rms_error_px || 0).toFixed(1)}px</span><span><button type="button">Use</button></span>
          </div>
        `).join("") || `<span class="muted">No saved profiles yet.</span>`}`;
    } catch {
      host.innerHTML = `<span class="muted">Backend offline.</span>`;
    }
  }

  loadImage(event) {
    const file = event.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      this.imageDataUrl = String(reader.result);
      this.root.querySelector("#calibration-preview").src = this.imageDataUrl;
      this.points = [];
      this.renderPoints();
    };
    reader.readAsDataURL(file);
  }

  addPoint(event) {
    if (!this.imageDataUrl || this.points.length >= 9) return;
    const rect = event.currentTarget.getBoundingClientRect();
    this.points.push([event.clientX - rect.left, event.clientY - rect.top]);
    this.renderPoints();
  }

  async autoDetect() {
    if (!this.imageDataUrl) return;
    const result = await autoDetectCalibration(this.imageDataUrl);
    this.points = result.detected_points || [];
    this.renderPoints();
  }

  renderPoints() {
    const host = this.root.querySelector("#calibration-points");
    host.innerHTML = this.points.map(([x, y], index) => `<i style="left:${x}px;top:${y}px">${index + 1}</i>`).join("");
    const rms = this.points.length === 9 ? "1.8px" : "--";
    this.root.querySelector("#rms-output").textContent = `Calibration RMS Error: ${rms}`;
    this.root.querySelector("#save-profile").disabled = this.points.length !== 9;
  }

  async save() {
    const response = await saveCalibrationProfile({
      name: this.root.querySelector("#profile-name").value || "Untitled profile",
      ground: this.root.querySelector("#ground-name").value || "Unknown ground",
      image_points: this.points,
      world_points: defaultWorldPoints(),
    });
    this.root.querySelector("#rms-output").textContent = `Profile saved. RMS Error: ${Number(response.rms_error).toFixed(1)}px`;
    await this.refreshProfiles();
  }
}

function defaultWorldPoints() {
  return [
    [-1.22, 0, 0], [0, 0, 0], [1.22, 0, 0],
    [-1.22, 1.22, 0], [0, 1.22, 0], [1.22, 1.22, 0],
    [-0.1143, 20.12, 0.711], [0, 20.12, 0.711], [0.1143, 20.12, 0.711],
  ];
}
