import { drawPitchMap } from "./PitchMap2D.js";

export class ResultsPanel {
  constructor(elements) {
    this.elements = elements;
  }

  render(results) {
    if (!results) return;
    const decision = results.decision || {};
    const gates = results.lbw_gates || {};
    this.renderTimeline(results);
    this.renderExplanation(decision.explanation || "Analysis complete.", decision.verdict);
    this.renderConfidence(gates, decision.confidence);
    this.renderHotspot(results);
    this.renderUltraEdge(results);
    this.renderPitchMap(results);
    this.renderExports(results);
  }

  renderTimeline(results) {
    const items = [
      ["Release", "complete"],
      ["Bounce", results.trajectory?.bounce_point ? "complete" : "pending"],
      ["Pad Impact", results.trajectory?.impact_point ? "complete" : "pending"],
      ["Stump Line", "complete"],
    ];
    this.elements.timeline.innerHTML = `<div class="event-timeline">${items.map(([label, status], index) => `
      <button type="button" class="event-dot ${status}" data-frame="${index * 30}">
        <i></i><span>${label}</span>
      </button>
    `).join("")}</div>`;
    this.elements.timeline.querySelectorAll(".event-dot").forEach((button) => {
      button.addEventListener("click", () => {
        this.elements.frameTimeline.value = button.dataset.frame || "0";
        this.elements.frameLabel.textContent = `Frame ${button.dataset.frame || 0}`;
      });
    });
  }

  renderExplanation(text, verdict) {
    const cls = verdict === "OUT" ? "sentence-out" : verdict === "NOT_OUT" ? "sentence-not-out" : "sentence-call";
    const sentences = String(text).split(/(?<=\.)\s+/).filter(Boolean);
    this.elements.explanation.innerHTML = sentences.map((sentence) => `<span class="${cls}">${sentence}</span>`).join(" ");
  }

  renderConfidence(gates, overall) {
    const rows = [
      ["Pitching", gates.pitching?.confidence],
      ["Impact", gates.impact?.confidence],
      ["Wickets", gates.wickets?.confidence],
      ["Overall", overall ?? gates.overall?.confidence],
    ];
    this.elements.confidenceBreakdown.innerHTML = rows.map(([label, value]) => {
      const pct = Math.round(Number(value || 0) * 100);
      return `<div class="confidence-row">
        <span>${label}</span><strong>${pct}%</strong>
        <div><i style="width:${pct}%"></i></div>
      </div>`;
    }).join("");
  }

  renderHotspot(results) {
    const impact = results.trajectory?.impact_point;
    if (!impact) {
      this.elements.hotspotView.textContent = "HotSpot data unavailable in Quick Test mode";
      return;
    }
    this.elements.hotspotView.innerHTML = `<strong>Estimated impact zone</strong><span>${formatPoint(impact)}</span>`;
    this.elements.hotspotView.classList.add("active");
  }

  renderUltraEdge(results) {
    const ctx = this.elements.ultraedge.getContext("2d");
    const { width, height } = this.elements.ultraedge;
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = "#080d0f";
    ctx.fillRect(0, 0, width, height);
    ctx.strokeStyle = "#37b7d8";
    ctx.beginPath();
    for (let x = 0; x < width; x += 1) {
      const y = height / 2 + Math.sin(x * 0.08) * 3;
      if (x === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
    this.elements.hotspotMode.textContent = results.edge_analysis?.edge_detected ? "Edge detected" : "No edge detected";
  }

  renderPitchMap(results) {
    const canvas = document.getElementById("pitch-map-canvas");
    drawPitchMap(canvas, results);
  }

  renderExports(results) {
    const host = document.getElementById("export-panel");
    if (!host) return;
    const jobId = results.job_id;
    host.innerHTML = `
      <button type="button" data-export="json">Export JSON</button>
      <button type="button" data-export="pdf">Export PDF</button>
      <button type="button" data-export="animation">Export MP4</button>
    `;
    host.querySelectorAll("button").forEach((button) => {
      button.addEventListener("click", () => {
        const kind = button.dataset.export;
        window.location.href = `http://localhost:8765/api/testing/jobs/${jobId}/exports/${kind}`;
      });
    });
  }
}

function formatPoint(point) {
  if (!point) return "--";
  return `${Number(point.x || 0).toFixed(1)}, ${Number(point.y || 0).toFixed(1)}, ${Number(point.z || 0).toFixed(1)}`;
}
