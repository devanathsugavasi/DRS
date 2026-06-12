export class DRSAnimationSequencer {
  constructor({ overlay, title, frameLabel, replayTimeline }) {
    this.overlay = overlay;
    this.title = title;
    this.frameLabel = frameLabel;
    this.replayTimeline = replayTimeline;
    this.timer = null;
    this.step = 0;
    this.results = null;
  }

  play(results) {
    this.reset();
    this.results = results;
    const steps = [
      () => this.show("REVIEWING...", "Reviewing delivery", 0),
      () => this.show("TRACKING BALL", "Slow motion replay", 24),
      () => this.show("BALL TRACKING", "3D trajectory drawing", 55),
      () => this.show("IMPACT", "Impact zone", 78),
      () => this.show(this.verdictText(), "Final decision", 100),
    ];
    steps[0]();
    this.timer = setInterval(() => {
      this.step += 1;
      if (this.step >= steps.length) {
        clearInterval(this.timer);
        this.timer = null;
        return;
      }
      steps[this.step]();
    }, 1000);
  }

  pause() {
    if (this.timer) clearInterval(this.timer);
    this.timer = null;
  }

  reset() {
    this.pause();
    this.step = 0;
  }

  exportMP4() {
    const jobId = this.results?.job_id;
    if (jobId) window.location.href = `http://localhost:8765/api/testing/jobs/${jobId}/exports/animation`;
  }

  show(text, title, frame) {
    this.overlay.textContent = text;
    this.overlay.className = `broadcast-overlay ${this.className()}`;
    this.title.textContent = title;
    this.replayTimeline.value = String(frame);
    this.frameLabel.textContent = `Frame ${frame}`;
  }

  verdictText() {
    const verdict = this.results?.decision?.verdict || "UMPIRES_CALL";
    return verdict.replace("_", " ");
  }

  className() {
    const verdict = this.results?.decision?.verdict || "";
    if (verdict === "OUT") return "out";
    if (verdict === "NOT_OUT") return "not-out";
    if (verdict.includes("UMPIRE")) return "processing";
    return "waiting";
  }
}
