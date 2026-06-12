export function watchAnalysisJob(jobId, handlers = {}) {
  const socket = new WebSocket(`ws://localhost:8765/ws/job/${jobId}`);
  socket.addEventListener("message", (event) => {
    try {
      const payload = JSON.parse(event.data);
      if (payload.type === "progress") handlers.onProgress?.(payload);
      if (payload.type === "tracking_update") handlers.onTracking?.(payload);
      if (payload.type === "decision_ready") handlers.onDecision?.(payload);
      if (payload.type === "animation_ready") handlers.onAnimation?.(payload);
    } catch {}
  });
  socket.addEventListener("error", () => handlers.onError?.("WebSocket unavailable; polling status."));
  return {
    close: () => socket.close(),
  };
}

export async function pollAnalysisJob(jobId, handlers = {}) {
  const timer = setInterval(async () => {
    try {
      const response = await fetch(`http://localhost:8765/api/analyze/${jobId}/status`);
      if (!response.ok) return;
      const status = await response.json();
      handlers.onProgress?.({ type: "progress", step: status.current_step, percent: status.progress, frames_done: status.frames_processed });
      if (status.status === "complete") {
        clearInterval(timer);
        handlers.onComplete?.();
      }
      if (status.status === "error") {
        clearInterval(timer);
        handlers.onError?.(status.current_step);
      }
    } catch {}
  }, 1200);
  return () => clearInterval(timer);
}
