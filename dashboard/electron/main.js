const { app, BrowserWindow, dialog, ipcMain } = require("electron");
const { spawn } = require("child_process");
const fs = require("fs");
const http = require("http");
const path = require("path");

const ENGINE_PORT = 8765;
const TESTING_PLATFORM_PORT = 5173;
const ENGINE_URL = `http://localhost:${ENGINE_PORT}`;
const TESTING_PLATFORM_URL = `http://127.0.0.1:${TESTING_PLATFORM_PORT}`;
const REPO_ROOT = path.resolve(__dirname, "..", "..");
const PYTHON_EXE = path.join(REPO_ROOT, ".venv", "Scripts", "python.exe");
const TESTING_PLATFORM_ROOT = path.join(REPO_ROOT, "dashboard", "testing-platform");
const LOG_PREFIX = "[DRS Electron]";

let engineProcess = null;
let testingPlatformProcess = null;
let mainWindow = null;

const startupState = {
  engine: { status: "pending", message: "Starting backend..." },
  testingPlatform: { status: "skipped", message: "Optional — not started yet" },
};

function log(message, ...args) {
  console.log(LOG_PREFIX, message, ...args);
}

function logError(message, ...args) {
  console.error(LOG_PREFIX, message, ...args);
}

process.on("unhandledRejection", (reason) => {
  logError("Unhandled promise rejection:", reason);
});

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1440,
    height: 960,
    minWidth: 1100,
    minHeight: 720,
    backgroundColor: "#090b10",
    show: true,
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });

  mainWindow.loadURL(loadingScreen(startupState.engine.message));
  mainWindow.on("closed", () => {
    mainWindow = null;
  });
}

function loadingScreen(detail) {
  const text = detail || "Starting DRS engine...";
  return `data:text/html;charset=utf-8,${encodeURIComponent(`
    <body style="margin:0;background:#090b10;color:#f8fafc;font-family:Inter,Segoe UI,sans-serif;display:grid;place-items:center;height:100vh">
      <main style="text-align:center;max-width:520px;padding:24px">
        <h1 style="font-size:28px;margin:0 0 10px">Starting DRS dashboard</h1>
        <p style="color:#94a3b8;margin:0">${text}</p>
      </main>
    </body>
  `)}`;
}

function attachProcessLogging(label, child) {
  if (!child) return;
  child.stdout?.on("data", (chunk) => log(`${label} stdout:`, String(chunk).trim()));
  child.stderr?.on("data", (chunk) => logError(`${label} stderr:`, String(chunk).trim()));
  child.on("error", (error) => {
    logError(`${label} spawn error:`, error.message);
  });
  child.on("exit", (code, signal) => {
    log(`${label} exited`, { code, signal });
  });
}

function spawnOptions(cwd) {
  const options = {
    cwd,
    env: { ...process.env },
    windowsHide: true,
    stdio: ["ignore", "pipe", "pipe"],
  };
  if (process.platform === "win32") {
    options.shell = true;
  }
  return options;
}

function healthCheckAsync() {
  return new Promise((resolve) => {
    healthCheck(resolve);
  });
}

async function startEngine() {
  startupState.engine = { status: "starting", message: "Launching FastAPI backend..." };

  if (await healthCheckAsync()) {
    startupState.engine = { status: "online", message: "Using existing backend on port 8765" };
    log("Backend already running — skipping spawn");
    return true;
  }

  if (!fs.existsSync(PYTHON_EXE)) {
    const message = `Python venv not found at ${PYTHON_EXE}`;
    startupState.engine = { status: "failed", message };
    logError(message);
    return false;
  }

  try {
    const command = process.platform === "win32"
      ? `"${PYTHON_EXE}" -m uvicorn core.testing_api:app --host 127.0.0.1 --port ${ENGINE_PORT}`
      : null;

    engineProcess = command
      ? spawn(command, [], spawnOptions(REPO_ROOT))
      : spawn(
          PYTHON_EXE,
          ["-m", "uvicorn", "core.testing_api:app", "--host", "127.0.0.1", "--port", String(ENGINE_PORT)],
          { ...spawnOptions(REPO_ROOT), shell: false }
        );

    attachProcessLogging("engine", engineProcess);
    startupState.engine = { status: "started", message: "Backend process launched" };
    log("Engine spawn ok", { python: PYTHON_EXE, cwd: REPO_ROOT });
    return true;
  } catch (error) {
    startupState.engine = { status: "failed", message: error.message };
    logError("Engine spawn failed:", error.message);
    return false;
  }
}

function testingPlatformReady() {
  return fs.existsSync(path.join(TESTING_PLATFORM_ROOT, "package.json"))
    && fs.existsSync(path.join(TESTING_PLATFORM_ROOT, "node_modules"));
}

function startTestingPlatform() {
  if (!testingPlatformReady()) {
    startupState.testingPlatform = {
      status: "unavailable",
      message: "React testing platform not installed (optional). Run: cd dashboard/testing-platform && npm install",
    };
    log("Skipping testing platform — package.json or node_modules missing");
    return false;
  }

  startupState.testingPlatform = { status: "starting", message: "Launching React dev server (optional)..." };

  try {
    const npmScript = "npm run dev";
    testingPlatformProcess = spawn(npmScript, [], spawnOptions(TESTING_PLATFORM_ROOT));
    attachProcessLogging("testing-platform", testingPlatformProcess);
    startupState.testingPlatform = {
      status: "started",
      message: `Optional testing UI starting at ${TESTING_PLATFORM_URL}`,
      url: TESTING_PLATFORM_URL,
    };
    log("Testing platform spawn ok", { cwd: TESTING_PLATFORM_ROOT });
    return true;
  } catch (error) {
    startupState.testingPlatform = {
      status: "failed",
      message: `Testing platform failed to start: ${error.message}`,
      url: null,
    };
    logError("Testing platform spawn failed:", error.message);
    return false;
  }
}

function waitForEngine(timeoutMs = 45000) {
  const started = Date.now();
  return new Promise((resolve, reject) => {
    const timer = setInterval(() => {
      healthCheck((ready) => {
        if (ready) {
          clearInterval(timer);
          startupState.engine = { status: "online", message: "Backend connected" };
          resolve();
        } else if (Date.now() - started > timeoutMs) {
          clearInterval(timer);
          startupState.engine = {
            status: "offline",
            message: `Backend not reachable at ${ENGINE_URL} after ${timeoutMs / 1000}s`,
          };
          reject(new Error(startupState.engine.message));
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
  const indexPath = path.join(__dirname, "renderer", "index.html");
  if (!fs.existsSync(indexPath)) {
    throw new Error(`Dashboard UI missing: ${indexPath}`);
  }
  await mainWindow.loadFile(indexPath);
}

async function bootstrap() {
  createWindow();

  const engineSpawned = await startEngine();
  startTestingPlatform();

  try {
    if (engineSpawned) {
      if (mainWindow) mainWindow.loadURL(loadingScreen("Waiting for FastAPI on port 8765..."));
      await waitForEngine();
    } else {
      await dialog.showMessageBox(mainWindow, {
        type: "warning",
        title: "DRS backend unavailable",
        message: "The Python backend could not be started.",
        detail: startupState.engine.message,
      });
    }
  } catch (error) {
    logError("Backend health check failed:", error.message);
    await dialog.showMessageBox(mainWindow, {
      type: "warning",
      title: "DRS backend offline",
      message: "The dashboard will open in offline mode.",
      detail: `${error.message}\n\nYou can still use calibration UI. Upload testing requires the backend.`,
    });
  }

  try {
    await loadDashboardUi();
    if (mainWindow) {
      mainWindow.webContents.send("startup-status", startupState);
    }
    log("Dashboard loaded", startupState);
  } catch (error) {
    logError("Dashboard load failed:", error.message);
    dialog.showErrorBox("Dashboard failed to load", error.message);
  }
}

app.whenReady().then(bootstrap).catch((error) => {
  logError("Bootstrap failed:", error.message);
  dialog.showErrorBox("DRS failed to start", error.message);
});

function killChild(child, label) {
  if (!child || child.killed) return;
  try {
    if (process.platform === "win32") {
      spawn(`taskkill /pid ${child.pid} /T /F`, [], { shell: true, windowsHide: true });
    } else {
      child.kill("SIGTERM");
    }
    log(`Stopped ${label}`);
  } catch (error) {
    logError(`Failed to stop ${label}:`, error.message);
  }
}

app.on("will-quit", () => {
  killChild(engineProcess, "engine");
  killChild(testingPlatformProcess, "testing-platform");
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") app.quit();
});

app.on("activate", () => {
  if (BrowserWindow.getAllWindows().length === 0) {
    bootstrap().catch((error) => logError("Re-activate bootstrap failed:", error.message));
  }
});

ipcMain.handle("operator-command", async (_event, command) => {
  return { ok: true, command, timestamp: Date.now() };
});

ipcMain.handle("get-startup-status", async () => startupState);

ipcMain.handle("get-health", async () => getJson("/api/health"));
ipcMain.handle("get-system-health", async () => getJson("/api/system/health"));
ipcMain.handle("get-reviews", async () => getJson("/api/reviews"));
ipcMain.handle("get-testing-platform-url", async () => ({
  url: TESTING_PLATFORM_URL,
  status: startupState.testingPlatform.status,
  message: startupState.testingPlatform.message,
  available: startupState.testingPlatform.status === "started",
}));

ipcMain.handle("request-review", async (_event, data) => postJson("/api/appeal/request", data, true));
ipcMain.handle("set-analysis-mode", async (_event, data) => postJson("/api/analysis-mode", data));

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
          if (response.statusCode && response.statusCode >= 400) {
            reject(new Error(`HTTP ${response.statusCode} for ${route}`));
            return;
          }
          resolve(JSON.parse(body));
        } catch (error) {
          reject(error);
        }
      });
    });
    request.on("error", reject);
    request.setTimeout(5000, () => {
      request.destroy();
      reject(new Error(`Timeout requesting ${route}`));
    });
  });
}

function postJson(route, payload, emitDecision = false) {
  return new Promise((resolve, reject) => {
    const body = JSON.stringify(payload || {});
    const request = http.request(`${ENGINE_URL}${route}`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Content-Length": Buffer.byteLength(body),
      },
    }, (response) => {
      let data = "";
      response.setEncoding("utf8");
      response.on("data", (chunk) => {
        data += chunk;
      });
      response.on("end", () => {
        try {
          const parsed = data ? JSON.parse(data) : {};
          if (response.statusCode && response.statusCode >= 400) {
            reject(new Error(parsed.detail || `HTTP ${response.statusCode}`));
            return;
          }
          if (emitDecision && mainWindow) {
            mainWindow.webContents.send("decision-update", parsed.decision || parsed);
          }
          resolve(parsed);
        } catch (error) {
          reject(error);
        }
      });
    });
    request.on("error", reject);
    request.setTimeout(10000, () => {
      request.destroy();
      reject(new Error(`Timeout posting ${route}`));
    });
    request.write(body);
    request.end();
  });
}
