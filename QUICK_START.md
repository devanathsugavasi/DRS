# Quick Start: Testing Cricket DRS with Video

## Step 1: Generate a Test Video (1-2 minutes)

```powershell
python scripts/generate_test_video.py --output test_delivery.mp4 --duration 10 --fps 30
```

**Output:** `test_delivery.mp4` (10-second synthetic cricket delivery video)

---

## Step 2: Start the Testing Backend (30-60 seconds startup)

Open a **new PowerShell terminal** in your project directory:

```powershell
python drs_app.py --testing-api --host 127.0.0.1 --port 8765
```

**Expected output:**
```
10:33:50 | INFO | core.ball_detector - Loaded YOLO detector from models/...
INFO: Uvicorn running on http://127.0.0.1:8765
```

**Wait for "Uvicorn running" message before proceeding to Step 3**

---

## Step 3: Start the Testing Frontend (10 seconds)

Open **another new PowerShell terminal**:

```powershell
cd dashboard/testing-platform
npm run dev
```

**Expected output:**
```
  Local:   http://localhost:5173/
```

---

## Step 4: Upload and Test Your Video

1. **Open browser:** http://localhost:5173/
2. **Click "Upload Delivery"** button
3. **Select `test_delivery.mp4`** file
4. **Click "Process Delivery"** button
5. **Wait 20-30 seconds** for analysis to complete

---

## Step 5: View Results

After processing completes, you should see:

[OK] **Ball Tracking + Trajectory Overlay**
- Ball position in each frame
- Predicted trajectory path
- Speed indicator

[OK] **Clean DRS Animation**
- Slow-motion replay of delivery
- Marked bounce and impact points

[OK] **LBW Recommendation**
- Decision: OUT / NOT OUT / UMPIRE'S CALL
- Confidence score

[OK] **Metrics Panel**
- Ball Speed: ~XXX km/h
- Bounce Point: (X, Y)
- Impact Point: (X, Y)
- Tracking Reliability: MEDIUM/HIGH
- Pitch Location: Line/Length
- Impact Location: Leg/Middle/Off side

---

## Troubleshooting

### "Backend offline" in the web interface?

**Fix:**
1. Check backend terminal - look for "Uvicorn running" message
2. Verify port 8765 is not blocked by firewall
3. Restart both backend and frontend
   ```powershell
   # Stop both terminals (Ctrl+C)
   # Restart backend first, wait 30s
   python drs_app.py --testing-api --host 127.0.0.1 --port 8765
   # Then restart frontend
   cd dashboard/testing-platform && npm run dev
   ```

### Analysis takes too long?

- **10-second clip:** Should complete in 20-40 seconds
- **First run:** May take 60+ seconds due to model warmup
- **If > 2 minutes:** Job likely failed - check browser console for errors

### No metrics showing up?

1. Check browser **Console** (F12 -> Console tab) for errors
2. Check backend terminal for exceptions
3. Try a different video format (MP4, H.264 codec)

### Video won't upload?

- **Max file size:** 500MB per video
- **Format:** MP4 with H.264 video codec
- **Resolution:** Works best with 1280x720 or similar
- **FPS:** 24-60 FPS supported

---

## Testing with Your Own Videos

### Upload a Real Cricket Delivery Video

```powershell
# Use any cricket delivery video file
# Requirements:
# - MP4 format (or AVI/MOV with H.264 codec)
# - 10 seconds to 10 minutes duration
# - 30 FPS (24-60 FPS OK)
# - 1280x720 or higher resolution
```

**Via Web Interface:**
1. Click "Upload Delivery"
2. Select your cricket_delivery.mp4
3. Click "Process Delivery"
4. Wait for results

---

## Testing with Multiple Videos (2-3 Cameras, 2-3 Hours)

### For Live Multi-Camera Capture:

```powershell
# Terminal 1: Start backend
python drs_app.py --api --cameras 0,1,2 --record

# Terminal 2: Start Electron dashboard
cd dashboard/electron
npm start
```

This will:
- Capture from cameras 0, 1, 2 in real-time
- Record synchronized streams to `data/recordings/`
- Show live analysis in Electron dashboard
- Save decision logs

### For Long Recordings (2-3 hours offline):

```powershell
# Process pre-recorded videos
python drs_app.py --cameras 0,1,2 --headless --seconds 10800 --record
```

---

## Key Points

| Task | Time | Result |
|------|------|--------|
| Generate test video | 1-2 min | `test_delivery.mp4` |
| Start backend | 30-60s | "Uvicorn running" |
| Start frontend | 10s | http://localhost:5173 |
| Upload 10s video | 5s | File uploaded |
| Process video | 20-30s | All metrics populated |
| **Total time** | **~2 minutes** | **Complete analysis** |

---

## Files Generated After Testing

After processing completes, check these folders:

```
data/testing/
|-- uploads/          # Your uploaded videos
|-- outputs/          # Analysis results
|   |-- job_id/
|   |   |-- analyzed_video.mp4         # Annotated video
|   |   |-- animation.mp4              # Clean DRS animation
|   |   |-- report.pdf                 # PDF report
|   |   |-- results.json               # Full data JSON
|   |   `-- results.csv                # Tracking points CSV
|   `-- ...
`-- drs_testing.sqlite3  # Job database
```

---

## Next Steps

1. [OK] Test with 10-second synthetic video
2. [OK] Test with your own real cricket videos
3. [OK] Test multi-camera setup with 2-3 hours of data
4. [OK] Check accuracy gates and readiness status
5. [OK] Export reports (PDF, JSON, CSV)

---

**Questions?** Check `TESTING_GUIDE.md` for advanced troubleshooting.
