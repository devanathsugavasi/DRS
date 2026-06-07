# Cricket DRS Electron and MVP Audit

## Executive summary

The Electron app was displaying the Cricket DRS Testing Platform because `dashboard/electron/main.js` explicitly loaded the React Vite dev server at `http://localhost:5173` in development and the React build from `dashboard/testing-platform/dist` in production. That made Electron a wrapper around the upload testing platform, not a dedicated broadcast/umpire dashboard.

This has been corrected. Electron now loads `dashboard/electron/renderer/index.html`, a dedicated vanilla desktop renderer, and still starts FastAPI on port `8765`.

Startup MVP readiness score: 56/100.

The project has meaningful pieces: YOLO model loading, class filtering, upload analysis, tracking, basic LBW gates, export generation, calibration visibility, Electron packaging, and local test coverage. It is not yet a production DRS product because real-camera calibration, validated model metrics, true 3D ball reconstruction, robust live decision orchestration, and real match-grade accuracy tests are incomplete.

## Root cause analysis

Before the fix:

- `dashboard/electron/package.json` had `main: "main.js"` and `npm start` ran `electron .`.
- `dashboard/electron/main.js` created a `BrowserWindow`.
- It spawned `uvicorn core.testing_api:app --port 8765`.
- It polled `GET http://localhost:8765/api/health`.
- When ready, development mode loaded `http://localhost:5173`.
- Production mode loaded `../testing-platform/dist/index.html`.

Responsible files:

- `dashboard/electron/main.js`
- `dashboard/electron/package.json`
- `dashboard/testing-platform/package.json`
- `dashboard/testing-platform/src/App.jsx`

Result: Electron was not a desktop DRS dashboard. It was an Electron shell around the React upload testing tool.

After the fix:

- `dashboard/electron/main.js` starts `.venv/Scripts/python.exe -m uvicorn core.testing_api:app --port 8765`.
- It waits for `/api/health`.
- It loads `dashboard/electron/renderer/index.html`.
- `dashboard/electron/preload.js` exposes secure IPC helpers.
- `dashboard/electron/renderer/renderer.js` polls live frames, camera FPS, and decisions.

## Current frontend applications

| App | Location | Purpose | Current status |
|---|---|---|---|
| React testing platform | `dashboard/testing-platform/` | Developer upload/review/testing UI | Working internal tool |
| Electron broadcast dashboard | `dashboard/electron/renderer/` | Desktop umpire/broadcast dashboard | Newly implemented MVP shell |
| Old Electron static files | `dashboard/electron/index.html`, `renderer.js`, `styles.css` | Older command-center UI | Not loaded by current Electron |
| Tkinter dashboard | `ui/dashboard.py` | Legacy local dashboard | Separate from Electron |
| Training UI | `ui/training_app.py` | Model training helper | Separate tool |

## Backend services

| Service | File | Purpose | Issue |
|---|---|---|---|
| Live camera backend | `core/api_server.py` | Physical camera capture, replay, `/api/live`, `/ws/status` | Not used by Electron after the testing-platform switch |
| Testing backend | `core/testing_api.py` | Upload-based testing pipeline | Used by Electron; now extended with dashboard endpoints |
| Pipeline | `core/testing_pipeline.py` | Offline delivery processing | Heuristic geometry and simplified animation |
| Camera manager | `core/camera_manager.py` | Capture/replay buffer | Needs reconnection, stronger sync metrics, live FPS endpoint integration |

## Gap analysis

| Feature | Status | Existing files | Missing components | Priority |
|---|---|---|---|---|
| Desktop application | Partial | `dashboard/electron/main.js`, `dashboard/electron/renderer/*` | Production-grade IPC, app icon, installer metadata | P0 |
| Video upload | Present | `core/testing_api.py`, `dashboard/testing-platform/src/App.jsx` | Multi-format normalization, file size policy | P1 |
| Ball detection | Partial | `core/ball_detector.py`, `models/cricket_ball_yolov8.pt` | Validated mAP50 >= 0.88 report | P0 |
| Ball tracking | Partial | `core/ball_association.py`, `core/tracking_quality.py` | Real-world robustness tests | P0 |
| Trajectory prediction | Partial | `core/trajectory.py`, `core/testing_pipeline.py` | Calibration-backed world coordinates | P0 |
| Pitching point detection | Heuristic | `core/testing_pipeline.py` | Real bounce detection validation | P0 |
| Impact detection | Heuristic | `core/testing_pipeline.py` | Pad/bat/player detector or calibrated geometry | P0 |
| Wicket intersection prediction | Partial | `core/lbw.py`, `core/lbw_engine.py` | Validated stump-zone projection | P0 |
| LBW decision engine | Partial | `core/lbw_engine.py`, `core/readiness.py` | One unified engine contract | P0 |
| Confidence score | Present | `core/testing_pipeline.py`, `core/readiness.py` | Calibrated confidence model | P1 |
| Ball speed | Approximate | `core/testing_pipeline.py` | Meter-scale calibration | P1 |
| Frame review | Present | React testing platform | Broadcast renderer frame stepping | P1 |
| Hawk-Eye style animation | Basic | `core/testing_pipeline.py` | True 3D trajectory and pitch model | P0 |
| Decision summary | Present | Testing exports and dashboard | Live decision lifecycle | P1 |
| Export reports | Partial | `core/testing_pipeline.py` | Validated 3-page PDF, richer CSV schema | P1 |
| Real camera sync | Partial | `core/camera_manager.py`, `core/synchronization.py` | Benchmark report and pass/fail gate | P0 |
| Calibration | Partial | `core/calibration.py`, `scripts/calibrate.py` | Real camera calibration data in `data/calibration/` | P0 |

## Classification of current dashboards

- React testing platform: Developer testing tool and internal debug interface. Not production desktop app.
- Old Electron command center: Mock-heavy dashboard prototype. Not currently primary production UI.
- New Electron renderer: Dedicated desktop MVP shell. Not production-ready until live camera, model validation, calibration, and decision tests pass.

## Code issues

- There are two LBW engines: `core/lbw.py` and `core/lbw_engine.py`. This risks inconsistent decisions.
- `core/testing_pipeline.py` still uses heuristic stumps/pads/bat boxes.
- Model metrics in `models/model_evaluation.json` are missing/null, so production readiness gates should fail.
- The PDF export is minimal and not a broadcast-grade evidence report.
- CSV export is tracking-oriented, not decision-report oriented.
- `core.testing_api` constructs the pipeline globally at import time, which can make CI and startup heavier than needed.

## Electron issues found

- Wrong frontend loaded: fixed.
- Electron was using Testing Platform React dev server: fixed.
- Production build included Testing Platform dist instead of dashboard renderer: fixed.
- Preload exposed only a generic command API: fixed with `onDecision`, `requestReview`, and `getHealth`.
- Missing dedicated renderer folder: fixed.
- Backend endpoints needed by broadcast renderer were missing from `testing_api`: fixed with synthetic camera/decision endpoints.

## Computer vision issues

- The cricket model is installed, but validated accuracy gates are not proven.
- Local logs show repeated warnings: no valid ball-class detections after class filtering.
- The external video folder exists at `E:\PRIVATE\AVCHD\BDMV\STREAM`.
- The pitch image folder exists at `E:\PRIVATE\AVCHD\BDMV\PITCH img`, not `PITCH`.
- Real calibration files are not present under `data/calibration/`.

## DRS logic issues

- The system correctly avoids confident decisions when gates fail, but the gate model is not yet the requested nine-gate contract.
- Impact and pitching are heuristics unless calibration is imported.
- REVIEW INCONCLUSIVE is the correct safe output for missing evidence.

## Animation issues

- Current animation is a clean 2D replay, not Hawk-Eye-grade 3D.
- No 3D trajectory HTML/PNG export yet.
- No edge waveform visualization in exported report yet.

## Exact `npm start` trace

Current corrected behavior from `dashboard/electron`:

1. `npm start`
2. `electron .`
3. Electron reads `dashboard/electron/package.json`
4. Electron starts `dashboard/electron/main.js`
5. `main.js` creates `BrowserWindow`
6. `main.js` spawns `.venv/Scripts/python.exe -m uvicorn core.testing_api:app --port 8765`
7. `main.js` polls `GET http://localhost:8765/api/health`
8. When ready, Electron loads `dashboard/electron/renderer/index.html`
9. Renderer polls:
   - `GET /api/live/1.jpg`
   - `GET /api/live/2.jpg`
   - `GET /api/cameras/fps`
   - `GET /api/decision/current`
10. Appeal buttons call:
   - `POST /api/appeal/request`
   - `POST /api/decision/confirm`

## Correct architecture

```text
Cricket DRS Desktop Application
|-- Electron shell
|   |-- Main process starts/stops FastAPI
|   |-- Secure preload IPC
|   `-- Broadcast renderer
|-- FastAPI engine
|   |-- Match upload API
|   |-- Live camera API
|   |-- Decision API
|   |-- Export API
|   `-- Health/diagnostics API
|-- Analysis engine
|   |-- Ball detection
|   |-- Ball tracking
|   |-- Calibration
|   |-- Trajectory prediction
|   |-- LBW gates
|   `-- Report exports
`-- Artifacts
    |-- Videos
    |-- JSON/CSV/PDF
    |-- Calibration files
    `-- Validation reports
```

## Development roadmap

Phase 1 - Fix Electron/Desktop Architecture

- Done: load dedicated Electron renderer.
- Done: preload IPC bridge.
- Done: package renderer directly.
- Next: remove or archive old unused Electron root HTML/JS after confirming no dependency.

Phase 2 - Complete Computer Vision Pipeline

- Validate cricket model against a labeled dataset.
- Import local `.MTS` clips into repeatable fixtures.
- Convert pitch images into calibration assets.
- Replace heuristic pads/bat/stumps with calibrated or detected geometry.

Phase 3 - Implement DRS Decision Engine

- Unify `core/lbw.py` and `core/lbw_engine.py`.
- Implement the full nine-gate decision contract.
- Add OUT / NOT_OUT / REVIEW_INCONCLUSIVE scenario fixtures.

Phase 4 - Implement Hawk-Eye Style Animation

- Add 3D trajectory generation.
- Add wicket-zone intersection visualization.
- Add edge waveform visualization.
- Embed animation artifacts into PDF exports.

Phase 5 - Production Packaging

- Add app icon, metadata, signing strategy.
- Add Docker backend deployment.
- Add GitHub Actions for Python, Electron build, and artifact upload.
- Add camera benchmark reports to release checklist.
