const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("drs", {
  onDecision: (cb) => ipcRenderer.on("decision-update", (_event, decision) => cb(decision)),
  requestReview: (data) => ipcRenderer.invoke("request-review", data),
  getHealth: () => ipcRenderer.invoke("get-health"),
  command: (name) => ipcRenderer.invoke("operator-command", name)
});
