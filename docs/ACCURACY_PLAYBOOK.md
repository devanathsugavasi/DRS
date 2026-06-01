# Accuracy Playbook For Cricket DRS

This project now has the software structure for serious DRS testing, but high accuracy comes from the capture setup and training data.

## Minimum Reliable Setup

- Use 120 FPS or higher when possible.
- Use short shutter speed to reduce ball blur.
- Use at least two cameras for LBW:
  - side-on camera for pitching/impact/no-ball
  - stump-line or front-on camera for line and wicket impact
- Lock camera zoom, focus, exposure, and resolution before calibration.
- Record calibration after placing cameras in their final match positions.

## Calibration Workflow

1. Print a checkerboard with known square size.
2. Capture 25-40 images per camera at different positions and tilt angles.
3. Place pitch markers at:
   - popping crease
   - bowling crease
   - return creases
   - stump centers
4. Save calibration data per camera.
5. Recalibrate whenever camera position, zoom, or focus changes.

Without this, single-camera trajectory depth is only an approximation.

After validation, write the measured readiness metrics:

```powershell
.\.venv\Scripts\python.exe scripts\write_calibration_readiness.py `
  --reprojection-error-px 1.10 `
  --homography-error-cm 3.5 `
  --pitch-coordinate-error-cm 4.0 `
  --per-camera-json "{""cam0"":{""reprojection_error_px"":1.05},""cam1"":{""reprojection_error_px"":1.14}}"
```

The platform will not show `OUT`, `NOT OUT`, or `UMPIRE'S CALL` unless calibration metrics pass the configured thresholds.

## YOLO Dataset Requirements

Train a multi-class model with:

- `ball`
- `bat`
- `front_pad`
- `back_pad`
- `stumps`
- `crease`

Recommended starting dataset size:

- 2,000+ labeled ball frames minimum for early testing
- 8,000-20,000 labeled frames for robust local tournament use
- Include red ball, white ball, old ball, new ball, shadow, sunlight, motion blur, pads, bat, wicketkeeper, umpire, crowd, and pitch wear

Use `training/drs_yolo_dataset.yaml` and train with:

```powershell
.\.venv\Scripts\python.exe scripts\train_yolo_drs.py --epochs 120 --imgsz 1280 --device 0
```

Evaluate before replacing the active model:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_yolo_drs.py --model models\training_runs\drs_yolov8\weights\best.pt
```

Only replace `models/cricket_ball_yolov8.pt` after validation:

```powershell
Copy-Item models\training_runs\drs_yolov8\weights\best.pt models\cricket_ball_yolov8.pt
```

## Reliability Levels

The testing platform now reports measurable tracking reliability:

- `high`: usable for strong testing evidence
- `medium`: useful for review, but not final
- `low`: do not trust the decision; improve camera angle, lighting, or model

The score includes detection coverage, max missing gap, trajectory smoothness, jump rejection, and average confidence. Kalman-filled points are shown as predictions and do not count as real ball detections.

For actual match decisions, treat any low or medium reliability result as "needs human/operator review".

## Decision Gates

The UI displays `REVIEW INCONCLUSIVE` unless all gates pass:

- model mAP50 >= 0.88
- ball recall >= 0.90
- calibration reprojection error <= 1.5 px
- homography validation error <= 5 cm
- tracking reliability is medium/high with max missing gap <= 5 frames
- sync error <= 8 ms
- replay FPS >= 24
- decision confidence >= 70%

## Clean DRS Animation

The export `clean_drs_animation.mp4` removes the batter/background completely and renders only:

- pitch
- crease
- stumps
- tracked ball path
- decision and confidence panel

This is the right output for DRS-style replay graphics.
