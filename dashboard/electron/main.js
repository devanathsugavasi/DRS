const { app, BrowserWindow, dialog, ipcMain } = require("electron");
const { spawn } = require("child_process");
const http = require("http");
const path = require("path");

const ENGINE_PORT = 8765;
const ENGINE_URL = `http://localhost:${ENGINE_PORT}`;
const REPO_ROOT = path.join(__dirname, "..", "..");
const PYTHON_EXE = path.join(REPO_ROOT, ".venv", "Scripts", "python.exe");

let engineProcess = null;
let mainWindow = null;

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1440,
    height: 960,
    minWidth: 1100,
    minHeight: 720,
    backgroundColor: "#090b10",
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      contextIsolation: true,
      nodeIntegration: false
    }
  });

  mainWindow.loadURL(loadingScreen());
}

function loadingScreen() {
  return `data:text/html;charset=utf-8,${encodeURIComponent(`
    <body style="margin:0;background:#090b10;color:#f8fafc;font-family:Inter,Segoe UI,sans-serif;display:grid;place-items:center;height:100vh">
      <main style="text-align:center">
        <h1 style="font-size:28px;margin:0 0 10px">Starting DRS engine...</h1>
        <p style="color:#94a3b8;margin:0">FastAPI is loading the cricket analysis backend.</p>
      </main>
    </body>
  `)}`;
}

function startEngine() {
  engineProcess = spawn(PYTHON_EXE, ["-m", "uvicorn", "core.testing_api:app", "--port", String(ENGINE_PORT)], {
    cwd: REPO_ROOT,
    stdio: "ignore",
    windowsHide: true
  });
}

function waitForEngine(timeoutMs = 30000) {
  const started = Date.now();
  return new Promise((resolve, reject) => {
    const timer = setInterval(() => {
      healthCheck((ready) => {
        if (ready) {
          clearInterval(timer);
          resolve();
        } else if (Date.now() - started > timeoutMs) {
          clearInterval(timer);
          reject(new Error("Timed out waiting for the DRS engine."));
        }
      });
    }, 500);
  });
}

function healthCheck(callback) {
  const request = http.get(`${ENGINE_URL}/api/health`, (response) => {
    response.resume();
    callback(response.statusCode === 200);
  });
  request.on("error", () => callback(false));
  request.setTimeout(450, () => {
    request.destroy();
    callback(false);
  });
}

async function loadDashboardUi() {
  if (!mainWindow) return;
  await mainWindow.loadFile(path.join(__dirname, "renderer", "index.html"));
}

app.whenReady().then(async () => {
  createWindow();
  startEngine();
  try {
    await waitForEngine();
    await loadDashboardUi();
  } catch (error) {
    dialog.showErrorBox("DRS engine failed to start", error.message);
    app.quit();
  }
});

app.on("will-quit", () => {
  if (engineProcess && !engineProcess.killed) {
    engineProcess.kill();
  }
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") app.quit();
});

app.on("activate", () => {
  if (BrowserWindow.getAllWindows().length === 0) createWindow();
});

ipcMain.handle("operator-command", async (_event, command) => {
  return { ok: true, command, timestamp: Date.now() };
});

ipcMain.handle("get-health", async () => getJson("/api/health"));

ipcMain.handle("request-review", async (_event, data) => postJson("/api/appeal/request", data));

function getJson(route) {
  return new Promise((resolve, reject) => {
    const request = http.get(`${ENGINE_URL}${route}`, (response) => {
      let body = "";
      response.setEncoding("utf8");
      response.on("data", (chunk) => {
        body += chunk;
      });
      response.on("end", () => {
        try {
          resolve(JSON.parse(body));
        } catch (error) {
          reject(error);
        }
      });
    });
    request.on("error", reject);
  });
}

function postJson(route, payload) {
  return new Promise((resolve, reject) => {
    const body = JSON.stringify(payload || {});
    const request = http.request(`${ENGINE_URL}${route}`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Content-Length": Buffer.byteLength(body)
      }
    }, (response) => {
      let data = "";
      response.setEncoding("utf8");
      response.on("data", (chunk) => {
        data += chunk;
      });
      response.on("end", () => {
        try {
          const parsed = data ? JSON.parse(data) : {};
          if (mainWindow) mainWindow.webContents.send("decision-update", parsed.decision || parsed);
          resolve(parsed);
        } catch (error) {
          reject(error);
        }
      });
    });
    request.on("error", reject);
    request.write(body);
    request.end();
  });
}
