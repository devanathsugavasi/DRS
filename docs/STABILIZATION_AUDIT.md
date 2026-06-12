# DRS Stabilization Audit

Date: 2026-06-12

## Executive Status

This audit is a stabilization checkpoint, not a redesign.

Completed verification:
- `python drs_app.py --testing-api` starts successfully on `127.0.0.1:8765`.
- `/api/health` returns `ok`.
- `/api/calibration/profiles` returns successfully.
- `/api/analyze/test` returns successfully.
- `/api/analyze/test/results` returns successfully after fixing a helper bug.
- Upload -> analyze -> results works through `/api/analyze` using `tests/fixtures/scenarios/lbw_out_in_line.mp4`.
- Exports returned HTTP 200 for JSON, PDF, analyzed video, and animation MP4.
- `npm start` no longer immediately crashes with Electron GPU fatal errors after disabling GPU acceleration and moving Electron user data/cache under the project data folder.
- `pip check` passed.
- Focused tests passed: `tests/test_api.py`, `tests/test_ws_hub.py`, `tests/test_testing_platform.py`.

Not fully verified:
- Manual Electron renderer console inspection was not completed.
- Dashboard upload was verified through the same backend endpoint used by the dashboard, but not by manually clicking the Electron UI.
- Calibrated dashboard upload was not end-to-end verified because no saved calibration profile exists yet.

## Files Modified During Stabilization

- `core/testing_api.py`
- `core/ws_hub.py`
- `dashboard/electron/main.js`
- `dashboard/electron/preload.js`
- `dashboard/electron/renderer/index.html`
- `dashboard/electron/renderer/renderer.js`
- `dashboard/electron/renderer/styles.css`

Related earlier training-pipeline changes currently in the worktree:
- `scripts/prepare_training_data.py`
- `scripts/train_yolo_drs.py`
- `training/data.yaml`

## Testing Workflow Verification

### Backend

Status: Working.

Verified:
- Health endpoint works.
- Calibration profiles endpoint works.
- Analysis status/results endpoints work.
- File upload analysis works.
- Export endpoints work.

Observed result from fixture upload:
- Analysis completed.
- Exports were generated.
- Decision was `REVIEW INCONCLUSIVE`.
- Ball detections were zero for the chosen fixture, so the model/data path works but the specific clip did not produce usable tracking.

Blocker:
- The current detector/model did not detect the ball in the small fixture clip. Real `.MTS`-derived labeled data is still required for reliable DRS decisions.

### Electron Dashboard

Status: Starts and remains running after stabilization fixes.

Fixes made:
- Disabled Electron hardware acceleration and GPU compositing.
- Added Chromium flags to avoid GPU process fatal crash.
- Moved dev `userData` and cache paths into `data/electron-user-data` and `data/electron-cache`.
- Switched dashboard file loading to explicit `file://` URL handling.
- Stopped auto-starting the old React testing platform from Electron because integrated testing now exists in the main dashboard and the old spawn path was failing.

Remaining verification:
- Need renderer console inspection.
- Need manual click-through: Testing -> upload -> Analyze -> Results.

## Duplicate Systems Audit

### LBW

Authoritative:
- `core/lbw_engine.py`
- `core/drs_decision.py`

Review before delete:
- `core/lbw.py`

Reason:
- `core/drs_decision.py` and tests use `core/lbw_engine.py`.
- `core/lbw.py` appears to be an older visualization/decision helper and is not part of the main testing workflow.

### Synchronization

Authoritative for live workflow:
- `core/synchronization.py`

Review before delete:
- `core/sync.py`

Reason:
- `core/integration.py` and `core/api_server.py` import `SyncVerifier` from `core/synchronization.py`.
- `core/sync.py` is referenced by `tests/test_sync.py`, so it may be a legacy unit-tested implementation. Do not delete until test coverage is migrated or the test is retired.

### Calibration

Authoritative for dashboard/testing pitch profiles:
- `core/pitch_calibration.py`

Authoritative for checkerboard/multi-camera calibration:
- `core/calibration.py`

Keep both for now.

Reason:
- They serve different workflows. `pitch_calibration.py` is actively used by testing/dashboard/readiness. `calibration.py` is used by `scripts/calibrate.py` and supports checkerboard calibration.

### Trajectory

Authoritative:
- `core/trajectory.py`

No duplicate trajectory module found.

### Tracking

Authoritative for testing/offline association:
- `core/ball_association.py`

Authoritative for live pipeline:
- `core/ball_tracker.py`

Review before delete:
- `core/tracker.py`

Reason:
- `core/tracker.py` is only referenced by `tests/test_tracker.py` in the import scan.
- It may be an older tracker implementation.

### API Routes

Duplicate but intentional:
- `core/api_server.py`: live/multi-camera backend.
- `core/testing_api.py`: offline/dashboard testing backend.

Risk:
- Both expose similar routes such as `/api/health`, `/api/cameras/fps`, `/api/decision/current`, `/api/reviews`, `/api/live/{camera_id}.jpg`.

Recommendation:
- Keep both for now.
- Name them explicitly in docs as Live API and Testing API.
- Do not merge until the dashboard testing workflow is stable.

## Dead Code / Cleanup Candidates

### Safe To Delete

Do not delete automatically; these are candidates only.

- `data/training_smoke/`
  - Generated smoke-test data.
- `data/electron-cache/`
  - Generated Electron runtime cache.
- `data/electron-user-data/`
  - Generated Electron runtime state. Safe to clear when Electron is closed.
- `logs/testing_api_stdout.log`
- `logs/testing_api_stderr.log`
- `logs/electron_stdout.log`
- `logs/electron_stderr.log`
  - Generated logs. Keep if debugging.

### Review Before Delete

- `dashboard/electron/index.html`
- `dashboard/electron/renderer.js`
- `dashboard/electron/styles.css`
  - Older root-level Electron UI files. Current Electron loads `dashboard/electron/renderer/index.html`.

- `dashboard/testing-platform/`
  - Old separate React testing platform. It is superseded by integrated dashboard testing, but still useful as a fallback/dev tool.

- `scripts/run_working_demo.ps1`
- `scripts/start_testing_platform.ps1`
- `scripts/create_testing_platform_shortcut.ps1`
  - Old testing platform/demo launch scripts. Review before removal because docs still reference them.

- `ui/dashboard.py`
  - Tkinter dashboard path used by `drs_app.py` default mode. Keep until live Electron is the only supported UI.

- `main.js` at repo root
  - Empty placeholder. Review packaging references before deleting.

- `core/lbw.py`
- `core/sync.py`
- `core/tracker.py`
  - Legacy/parallel implementations. Review tests and behavior before removing.

- `training/images/`, `training/labels/`, `training/review/`
  - Generated training data. DRS-related, but large. Keep if actively labeling/training; move to external storage or gitignore if not meant for source control.

- `12345/`
  - Local `.MTS` footage copy. DRS-related but huge. Keep outside git/source control if possible.

### Required

- `drs_app.py`
- `core/testing_api.py`
- `core/testing_pipeline.py`
- `core/testing_database.py`
- `core/ws_hub.py`
- `core/ball_detector.py`
- `core/ball_association.py`
- `core/drs_decision.py`
- `core/lbw_engine.py`
- `core/readiness.py`
- `core/trajectory.py`
- `core/pitch_calibration.py`
- `dashboard/electron/main.js`
- `dashboard/electron/preload.js`
- `dashboard/electron/renderer/index.html`
- `dashboard/electron/renderer/renderer.js`
- `dashboard/electron/renderer/styles.css`
- `dashboard/electron/renderer/components/*`
- `dashboard/electron/renderer/hooks/*`
- `models/cricket_ball_yolov8.pt`
- `models/model_evaluation.json`
- `config/settings.py`
- `config/calibration_profiles.json`

## Dashboard Cleanup Audit

Target sections requested:
- Home
- Live DRS
- Testing
- Calibration
- Results
- Settings
- Logs

Current state:
- Main Electron dashboard has integrated Testing and Calibration modal.
- It still has older labels and controls for multi-camera status, HotSpot, UltraEdge, Review Center, and Testing Platform dialog markup.
- It does not yet have explicit Home/Live DRS/Settings/Logs route-level navigation.

Recommendation:
- Do not remove panels until the integrated Testing click-through is manually verified.
- Then remove old embedded testing-platform dialog markup and IPC.
- Keep HotSpot/UltraEdge panels only if they display meaningful idle states or real analysis output.

## Calibration Logic Audit

Testing:
- Calibration is optional.
- Quick Test works without calibration.
- UI warning exists for approximate results.

Live:
- Calibration-required guard is not fully enforced yet.
- Add a live-mode startup guard before enabling physical live DRS.

## Dependency Audit

Required Python packages:
- `opencv-python`
- `numpy`
- `scipy`
- `ultralytics`
- `pillow`
- `sounddevice`
- `fastapi`
- `uvicorn[standard]`
- `websockets`
- `python-multipart`
- `reportlab`
- `aiofiles`
- `sqlalchemy`
- `alembic`
- `pydantic`
- `pydantic-settings`
- `python-dotenv`
- `rich`
- `loguru`
- `pytest`
- `pytest-asyncio`
- `httpx`

Optional/missing from requirements but used defensively:
- `psutil` is imported optionally in API health functions.
- `torch` is imported optionally by the training script and normally arrives with Ultralytics.

Required Electron packages:
- `electron`
- `electron-builder`
- `three`

Old React testing platform packages:
- `vite`
- `react`
- `react-dom`
- `lucide-react`
- `tailwindcss`
- `@vitejs/plugin-react`

Broken imports:
- None found by compile/check/test pass.

## Architecture Usage

### Actively Used By Testing Workflow

- `drs_app.py`
- `core/testing_api.py`
- `core/testing_pipeline.py`
- `core/testing_database.py`
- `core/ws_hub.py`
- `core/ball_detector.py`
- `core/ball_association.py`
- `core/drs_decision.py`
- `core/lbw_engine.py`
- `core/readiness.py`
- `core/tracking_quality.py`
- `core/trajectory.py`
- `core/hotspot.py`
- `core/audio_edge.py`
- `core/pitch_calibration.py`
- `dashboard/electron/*`

### Actively Used By Live Workflow

- `drs_app.py`
- `core/api_server.py`
- `core/integration.py`
- `core/camera_manager.py`
- `core/ball_detector.py`
- `core/ball_tracker.py`
- `core/synchronization.py`
- `core/ws_hub.py`
- `core/pitch_calibration.py`

### Currently Unused Or Legacy-Looking

- `core/lbw.py`
- `core/sync.py`
- `core/tracker.py`
- `dashboard/electron/index.html`
- `dashboard/electron/renderer.js`
- `dashboard/electron/styles.css`
- `main.js`
- `scripts/create_testing_platform_shortcut.ps1`
- `scripts/start_testing_platform.ps1`

## Completion Estimates

Current project completion: 72%

Current dashboard completion: 70%

Current testing workflow completion: 82%

Current live DRS completion: 45%

## Remaining Blockers Before Real Live DRS

- Cricket-ball model needs real labeled `.MTS` training data and validation.
- Live mode needs calibration-required guard.
- Live single-camera pipeline still needs explicit product hardening.
- Calibrated analysis needs a real saved calibration profile.
- Electron renderer console needs direct inspection.
- Dashboard needs final cleanup of old testing-platform dialog remnants.
- Generated training data and large local `.MTS` clips need source-control hygiene.

## Do Not Delete

Do not delete unrelated Desktop files.

Do not delete zip archives without explicit approval.

Do not delete generated training folders until labeling/training status is confirmed.
