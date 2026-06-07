# Public Baselines and Datasets

This project can run now with public baseline assets, but public weights do not make the system tournament-accurate by themselves. Final `OUT`, `NOT OUT`, and `UMPIRE'S CALL` decisions remain gated by measured model, calibration, tracking, sync, replay, and confidence metrics.

## Recommended Immediate Baseline

Use `scripts/bootstrap_public_assets.py` to install a cricket-ball detector:

```powershell
.\.venv\Scripts\python.exe scripts\bootstrap_public_assets.py
```

The script downloads the Hugging Face model `ashishgimekar/cricket-ball-yolo` to `models/cricket_ball_yolov8.pt` and writes `models/model_evaluation.json` with metrics set to `null`. This is intentional. The model card says it is a fine-tuned YOLOv8 single-class cricket ball detector trained on the Kaggle Cricket Ball Dataset for YOLO, but it does not provide your local held-out mAP, recall, calibration, or sync measurements.

Use it for:

- Upload workflow testing
- Real-time detector smoke checks
- Ball overlay and tracking pipeline checks
- UI animation and replay verification

Do not use it for final LBW decisions until local readiness gates pass.

## Best Public Data Options Found

| Asset | What it gives | Current use |
| --- | --- | --- |
| Hugging Face `ashishgimekar/cricket-ball-yolo` | Public YOLOv8 cricket-ball weights, single class `ball` | Immediate detector baseline |
| Kaggle `kushagra3204/cricket-ball-dataset-for-yolo` | 1778 YOLOv8-format annotated cricket ball images, CC0/Public Domain | Best small public training/evaluation seed |
| Roboflow Cricket Ball Tracking DATASET | Public dataset family reported at 34.4k images across tracking/detection projects | Larger optional training/evaluation source after license/API checks |
| TrackNet/TrackNetV3 research datasets | Sports-ball trajectory model ideas and transfer learning direction | Architecture reference, not a cricket DRS drop-in |

## Download Dataset Seeds

Kaggle requires `kaggle.json` credentials:

```powershell
pip install kaggle
kaggle datasets download -d kushagra3204/cricket-ball-dataset-for-yolo -p data\datasets --unzip
```

Roboflow Universe downloads require an account/API key. Verify the license of the exact version before training:

```powershell
pip install roboflow
```

Then use the export snippet shown by Roboflow for your chosen version.

## Readiness Still Required

After collecting or downloading evaluation footage, run:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_yolo_drs.py --model models\cricket_ball_yolov8.pt --data training\drs_yolo_dataset.yaml
```

The dashboard should show final DRS decisions only when:

- Ball recall meets the configured target
- Calibration status is valid
- Homography error is within the configured cm threshold
- Tracking reliability is medium/high
- Sync error is below threshold
- Decision confidence is above threshold

If any gate fails, the correct output is `REVIEW INCONCLUSIVE`.

## Demo Videos

Generate synced sample videos for UI and replay testing:

```powershell
.\.venv\Scripts\python.exe scripts\generate_test_video.py --dual --output data\testing\demo_delivery.mp4
```

These videos include DRS-style ball trajectory, bounce marker, impact marker, stumps, and a batter-removed overlay. They are synthetic pipeline test assets, not model accuracy proof.
