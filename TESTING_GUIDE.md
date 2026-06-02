# Cricket DRS - Testing Guide & Troubleshooting

## Quick Testing with Short Clips (10-30 seconds)

### Prerequisites
1. Python 3.11+ with all dependencies installed
2. `pip install -e .` (install package in editable mode)
3. Cricket ball detection model loaded

### Testing via Testing Platform (Recommended for Offline Testing)

**Start the testing backend:**
```bash
python drs_app.py --testing-api --host 127.0.0.1 --port 8766
```

**Expected output:**
```
INFO: Started server process [XXXX]
INFO: Uvicorn running on http://127.0.0.1:8766
```

**Note:** The backend may take 30-60 seconds to start on first run due to model loading. Wait for "Started server process" message.

**Start the React testing platform:**
```bash
cd dashboard/testing-platform
npm run dev
```

**Troubleshooting "Backend offline":**
1. Check that `python drs_app.py --testing-api` is still running
2. Check port 8766 is accessible (no firewall blocks)
3. Check browser console for CORS errors
4. Restart both backend and frontend

### Testing with 2-3 Camera Inputs (Multi-hour recordings)

**For live capture with FastAPI:**
```bash
python drs_app.py --api --cameras 0,1,2 --record
```

**Starting the Electron dashboard:**
```bash
cd dashboard/electron
npm start
```

**For headless recording to file:**
```bash
python drs_app.py --cameras 0,1,2 --record --headless --seconds 7200
```

### Known Limitations with Short Clips (< 30 seconds)

1. **Model warmup time:** First inference adds 0.5-2s overhead per camera
2. **Tracking initialization:** ByteTrack needs 3-5 frames to establish reliable tracks
3. **Trajectory prediction:** Requires 10+ frames of tracked ball for accuracy
4. **LBW readiness gates:** May report "INCONCLUSIVE" on short clips due to insufficient data
5. **Sync detection:** Multi-camera sync verification needs at least 50 frames per camera

### Expected Output for 10-second Clip

**Should complete within:**
- Single camera: 15-30 seconds
- Dual camera: 20-40 seconds
- Analysis → JSON/CSV → Video generation

**All metrics should populate:**
- ✅ Ball detection count
- ✅ Tracking points
- ✅ Ball speed (km/h)
- ✅ Bounce/impact point coordinates
- ✅ Tracking quality score
- ✅ LBW decision (OUT/NOT OUT/INCONCLUSIVE)

### Performance Tips for Long Recordings (2-3 hours)

1. **Reduce FPS if possible:** Lower frame rate saves processing
   ```bash
   # Process at 15 FPS instead of 30
   options_json='{"max_frames": 5400}'  # 3 hours @ 30 FPS = 324,000 frames
   ```

2. **Use GPU acceleration:** If RTX GPU available
   ```bash
   pip install -r requirements-gpu.txt
   ```

3. **Monitor memory:** Long recordings use ~500MB-1GB per camera stream
   ```bash
   # Check memory usage during processing
   python -m psutil
   ```

4. **Background processing:** Run analysis in background
   ```bash
   # Upload multiple clips and let system queue them
   # Check /data/testing/outputs for results
   ```

### Debug Logging

**Enable verbose logging:**
```bash
# For backend (set in core modules)
# logs/drs_pipeline.log
# logs/ball_detector.log
# logs/camera_manager.log
```

**Check job status:**
```bash
sqlite3 data/testing/drs_testing.sqlite3 "SELECT job_id, status, created_at FROM jobs ORDER BY created_at DESC LIMIT 5;"
```

**If job fails:**
```bash
sqlite3 data/testing/drs_testing.sqlite3 "SELECT job_id, status, error FROM jobs WHERE status='failed' LIMIT 1;"
```

### Sample Test Files

Create a test video if needed:
```bash
# Using OpenCV Python script to generate test video
python scripts/generate_test_video.py --output test_delivery.mp4 --duration 10 --fps 30
```

## Next Steps

1. **Test with 10-second clip first** - Verify full pipeline works
2. **Upload 30-second clip** - Check performance scaling
3. **Test multi-camera** - 2-3 cameras for 2-3 hours

---

Generated: Cricket DRS Testing Platform v1.0
