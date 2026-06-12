export class StatusPanel {
  constructor(headerTitle, headerMeta, body) {
    this.headerTitle = headerTitle;
    this.headerMeta = headerMeta;
    this.body = body;
  }

  live(cameras, renderCameraGrid) {
    this.headerTitle.textContent = "Camera matrix";
    const connected = cameras.filter((camera) => camera.connected).length;
    this.headerMeta.textContent = `${connected} / 6 connected`;
    renderCameraGrid();
  }

  testing(videoInfo, progress) {
    this.headerTitle.textContent = "Video Info";
    this.headerMeta.textContent = progress?.status || "Testing";
    const pct = Number(progress?.percent || progress?.progress || 0);
    this.body.className = "status-panel";
    this.body.innerHTML = `
      <section class="video-info-card">
        <strong>Video Info</strong>
        <span>File: ${videoInfo?.name || "--"}</span>
        <span>Duration: ${formatDuration(videoInfo?.duration)}</span>
        <span>Resolution: ${videoInfo?.resolution || "--"}</span>
        <span>FPS: ${videoInfo?.fps || "--"}</span>
        <span>Total frames: ${videoInfo?.totalFrames || "--"}</span>
      </section>
      <section class="video-info-card">
        <strong>Analysis Progress</strong>
        <div class="analysis-progress"><i style="width:${pct}%"></i></div>
        <span>Status: ${progress?.step || progress?.current_step || "Waiting"}</span>
        <span>Frames processed: ${progress?.frames_done || progress?.frames_processed || 0}/${videoInfo?.totalFrames || "--"}</span>
        <span>Ball detected: ${progress?.ball_detected || 0} frames</span>
        <span>Detections: ${videoInfo?.totalFrames ? Math.round(((progress?.ball_detected || 0) / videoInfo.totalFrames) * 1000) / 10 : "--"}%</span>
      </section>
    `;
  }

  summary(results) {
    this.headerTitle.textContent = "Delivery Summary";
    this.headerMeta.textContent = results?.decision?.verdict || "Complete";
    const tracking = results?.ball_tracking || {};
    const video = results?.video_info || {};
    this.body.className = "status-panel";
    this.body.innerHTML = `
      <section class="video-info-card">
        <strong>Delivery Summary</strong>
        <span>Ball speed: ${results?.summary?.ball_speed_kmh || "--"} km/h</span>
        <span>Swing: --</span>
        <span>Bounce height: ${formatPoint(results?.trajectory?.bounce_point)}</span>
        <span>Impact height: ${results?.lbw_gates?.impact?.height_m ?? "--"}m</span>
        <span>Frames tracked: ${tracking.frames_tracked || 0}/${video.total_frames || "--"}</span>
        <span>Track confidence: ${Math.round(Number(tracking.avg_confidence || 0) * 100)}%</span>
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
  return `${Number(point.z || 0).toFixed(2)}m`;
}
