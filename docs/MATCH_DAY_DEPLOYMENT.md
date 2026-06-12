# Match-Day DRS Deployment Guide

Use this checklist before operating Cricket DRS at an academy session, pilot match, or investor demo.

## 1. Pre-Match Setup (30–45 minutes)

### Environment
```powershell
cd C:\path\to\DRS
.\.venv\Scripts\Activate.ps1
pip install -e .
```

### Models
- Confirm `models/cricket_ball_yolov8.pt` or `models/best.pt` exists
- Run smoke validation:
```powershell
python scripts/validate_detector.py --synthetic --device auto
```

### Calibration (required for LBW accuracy)
Per camera (repeat for each feed):
1. Open Electron → **Calibration**
2. Capture frame or upload snapshot
3. Mark: off stump, middle stump, leg stump, bowling crease, popping crease
4. Save profile

Or CLI wizard:
```powershell
python scripts/calibration_wizard.py --camera 0
python scripts/calibration_wizard.py --camera 1
```

Verify:
```powershell
python -c "from core.pitch_calibration import calibration_status_payload; print(calibration_status_payload())"
```

## 2. Start the System

### Primary (Electron broadcast dashboard)
```powershell
cd dashboard\electron
npm start
```

Electron will:
- Start FastAPI on port **8765** (or connect to existing backend)
- Optionally start React testing UI on **5173**

### Offline analysis only
```powershell
python drs_app.py --testing-api --host 127.0.0.1 --port 8765
cd dashboard\testing-platform
npm run dev
```

## 3. Offline DRS Workflow (Upload Review)

1. Upload 1–2 delivery videos in Testing platform
2. Enable: ball detection, tracking, LBW, replay
3. Wait for job completion
4. Verify exports: JSON, PDF, analyzed MP4, animation MP4
5. In Electron dashboard → **Request Review** to load latest decision

Quick validation script:
```powershell
python scripts/validate_full_pipeline.py --video data\test_videos\your_clip.mp4
```

## 4. Live DRS Workflow (Cameras)

```powershell
python drs_app.py --api --cameras 0,1 --record
```

- Camera 0 = bowler end, Camera 1 = square leg (adjust to your setup)
- If a camera fails, system falls back to synthetic feed and auto-reconnects
- Restart a camera via API: `POST /api/cameras/{id}/reconnect`

## 5. Appeal Workflow

1. Operator triggers **Request Review** (Electron) or `POST /api/appeal/request`
2. System loads latest completed analysis job
3. LBW engine produces OUT / NOT OUT / UMPIRE'S CALL / REVIEW INCONCLUSIVE
4. Readiness gates may block final OUT/NOT OUT until model + calibration metrics pass
5. Operator confirms OUT or NOT OUT

## 6. GPU Notes

If CUDA fails (`torch.AcceleratorError`):
- System auto-falls back to CPU inference
- Set `INFERENCE_DEVICE=cpu` in `.env` for stable operation
- Reinstall matching CUDA PyTorch build from `requirements-gpu.txt` when ready

## 7. Health Checks

| Check | Command / URL |
|-------|----------------|
| API health | `http://127.0.0.1:8765/api/health` |
| Calibration | `http://127.0.0.1:8765/api/calibration/status` |
| System health | `http://127.0.0.1:8765/api/system/health` |
| Tests | `pytest tests/ -q` |

## 8. Known Limitations (Pilot / MVP)

- Dual-camera 3D triangulation not tournament-validated
- GPU acceleration may require CUDA driver fix on some RTX systems
- Live Electron feeds use synthetic fallback when cameras unavailable
- Model metrics from smoke tests are not substitute for held-out match validation

## 9. Emergency Fallback

If Electron fails to start:
```powershell
python drs_app.py --testing-api
# Open http://127.0.0.1:5173 manually after npm run dev in testing-platform
```

If detection is poor:
- Lower confidence threshold in upload options
- Re-run `validate_detector.py` on your footage
- Add labels and retrain with `scripts/train_yolo_drs.py`
