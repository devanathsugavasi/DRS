const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("drs", {
  command: (name) => ipcRenderer.invoke("operator-command", name)
});
