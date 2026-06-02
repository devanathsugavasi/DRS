# Cricket DRS Prototype

This workspace contains a modular Python/OpenCV cricket DRS prototype covering:

- synchronized 2-6 camera capture with timestamped frames
- instant replay buffers, playback controls, slow motion, and frame stepping
- efficient OpenCV `VideoWriter` recording
- YOLOv8 cricket ball detection with JSON/CSV export
- Kalman-filter ball tracking, trajectory drawing, velocity, and direction
- checkerboard camera calibration and pitch-plane homography support
- software synchronization verification, dropped-frame reporting, and flash sync hooks
- projectile trajectory prediction and LBW decision suggestions
- audio FFT edge detection with video timestamp alignment
- Tkinter umpire dashboard for live review

## Install

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

For RTX/CUDA GPU installs:

```powershell
pip install -r requirements-gpu.txt
```

Place a trained cricket ball model at:

```text
models/cricket_ball_yolov8.pt
```

If the custom model is absent, the detector falls back to YOLOv8n's COCO `sports ball` class.

## Run Dashboard

```powershell
python drs_app.py --cameras 0,1 --record
```

## Electron Broadcast Dashboard

```powershell
cd dashboard\electron
npm install
npm start
```

The Electron shell connects to the Python backend at `http://127.0.0.1:8765`.
It now provides an enterprise DRS command-center layout with live camera review, appeal presets, decision analytics, system monitoring, pitch map, 3D trajectory canvas, trend charts, replay controls, logs, and WebSocket backend health.

Start the backend first:

```powershell
python drs_app.py --api --cameras 0,1,2,3,4,5 --record
```

Then start Electron in a second terminal:

```powershell
cd dashboard\electron
npm start
```

The backend exposes:

- `GET /api/cameras` for camera IDs and health
- `GET /api/presets/NO_BALL`, `/api/presets/LBW`, `/api/presets/EDGE`
- `GET /api/live/{camera_id}.jpg` for latest live frame
- `POST /api/replay/create` to snapshot the replay buffer
- `POST /api/replay/request` with `{ "camera_ids": [0, 2], "frame_index": 123 }` for synced multi-camera replay metadata
- `GET /api/replay/{camera_id}.jpg?frame_index=123`
- `WS /ws/status` for sync and camera health

Electron uses the WebSocket to show backend connection/sync status and uses `/api/replay/request` so all selected cameras are aligned to the same replay timestamp for appeal review.

## Offline DRS Testing Platform

Use this to upload one delivery video or two synchronized camera-angle videos and generate a DRS-style test analysis without live cameras.

Start the testing backend:

```powershell
.\.venv\Scripts\python.exe drs_app.py --testing-api --host 127.0.0.1 --port 8766
```

Start the React/Tailwind frontend:

```powershell
cd dashboard\testing-platform
npm install
npm run dev
```

Open `http://127.0.0.1:5174`.

Full architecture, API design, database design, source layout, and deployment notes are in `docs/DRS_TESTING_PLATFORM.md`.

Create a Windows desktop launcher:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\create_testing_platform_shortcut.ps1
```

Windows blocks fully automatic taskbar pinning for normal apps, so after the shortcut is created, right-click `Cricket DRS Testing Platform` on the desktop and choose **Pin to taskbar**.

## Accuracy Workflow

For reliable ball tracking, train a cricket-specific model on your own footage:

```powershell
.\.venv\Scripts\python.exe scripts\train_yolo_drs.py --base-model yolo11l.pt --epochs 120 --imgsz 1280 --device 0
.\.venv\Scripts\python.exe scripts\evaluate_yolo_drs.py --model models\training_runs\drs_yolov8\weights\best.pt
```

The detector selector prefers local `models/yolo11x.pt`, then `models/yolo11l.pt`, then `models/yolov8x.pt`, then `models/cricket_ball_yolov8.pt`. It does not claim tournament readiness unless model mAP, ball recall, calibration, sync, tracking, replay FPS, and decision confidence pass the readiness gates.

See `docs/ACCURACY_PLAYBOOK.md` for camera calibration, training data, reliability gates, and clean DRS animation guidance.

## Run Headless OpenCV Mode

```powershell
python drs_app.py --cameras 0,1 --headless --seconds 120
```

Recordings, detections, tracking exports, audio events, and calibration files are written under `data/`.

## Production Notes

- Use global shutter/high-FPS cameras where possible.
- Prefer hardware trigger or genlock in production; this project includes software timestamp alignment and optional flash detection for lower-cost deployments.
- Train YOLOv8 on red and white balls under real match lighting, including motion blur, shadows, pitch wear, and crowd backgrounds.
- For LBW-grade accuracy, feed the trajectory engine with calibrated multi-camera 3D reconstruction rather than the approximate pixel-to-world helper.
