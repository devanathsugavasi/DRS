const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("drs", {
  onDecision: (cb) => ipcRenderer.on("decision-update", (_event, decision) => cb(decision)),
  onStartupStatus: (cb) => ipcRenderer.on("startup-status", (_event, status) => cb(status)),
  requestReview: (data) => ipcRenderer.invoke("request-review", data),
  getHealth: () => ipcRenderer.invoke("get-health"),
  getStartupStatus: () => ipcRenderer.invoke("get-startup-status"),
  getSystemHealth: () => ipcRenderer.invoke("get-system-health"),
  getReviews: () => ipcRenderer.invoke("get-reviews"),
  getTestingPlatformUrl: () => ipcRenderer.invoke("get-testing-platform-url"),
  setAnalysisMode: (data) => ipcRenderer.invoke("set-analysis-mode", data),
  command: (name) => ipcRenderer.invoke("operator-command", name),
});
