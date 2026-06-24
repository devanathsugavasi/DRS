# DRS — Master Plan (single-camera, honest scope)

**Mindset:** stop building toward the architecture diagram. The diagram assumes a 2–6 camera broadcast truck. We have **one fixed stand camera, 1080p @ 50fps**. Build toward *"a single-camera 2D delivery analyser that is honest about what it can't see."* Multi-camera 3D is a **future phase, gated on actually having more cameras.**

---

## What one fixed camera CAN and CANNOT do

| Achievable (honest) | NOT achievable from one fixed camera |
|---|---|
| Detect + track the ball down the pitch (motion tracking works, straightness 1.00) | True 3D ball **height** at impact → "hitting stumps" is a **guess, not a measurement** |
| Estimate **pitching point** + **bounce** on the 2D pitch plane (pitch is a known flat rectangle → single-plane homography, valid) | Real 3D triangulation (no depth from one view) |
| **Line** decisions: pitched in/outside line, impact in/outside line | UltraEdge / HotSpot (no thermal cam, no proven synced mic) |
| Clean 2D replay overlay on a pitch diagram | "sub-8ms multi-cam sync" gates (nothing to sync) |
| Auto-flag deliveries | |

**The gate behavior is correct:** when height/calibration confidence is low, the system returns `UMPIRE'S CALL` / `REVIEW INCONCLUSIVE`. That is honesty, not failure. Lean into it.

---

## LOCKED DECISION — Crop ROI

**Pitch-zone crop = `(0.27, 0.28, 0.73, 0.90)`** of the full 1920×1080 frame.
- Pixel box: `(518, 302) → (1402, 972)`, crop size **884×670**.
- Verified on clips 00003/05/06: resolves both stump sets, crease lines, batsman, pitch corridor. Ball becomes ~10–15px.
- Constant in `scripts/extract_pitch_crops.py` and `scripts/motion_ball_finder.py`.
- May be nudged ±a few % after marking calibration corners (Loop 1.3), but this is the working value. **Loop 1 starts now.**

---

## The ball-detection truth (answers "ball not detected / detecting line wrong")

The wrong boxes (lines, players, umpire shirts) come from **YOLO**, which **cannot work** on a ~5px ball among players. Retraining YOLO the normal way fails because:
- the ball is too small to label reliably by eye,
- the auto-labeler produced ~66% false labels (umpires/jerseys).

**The ball IS detected correctly — by motion tracking** (3-frame differencing + trajectory linking), proven on a real ball. So:

- **Primary fix:** replace YOLO with the **motion tracker** as the ball source in the pipeline (Loop 3). This removes the "detecting line wrong" problem entirely.
- **Optional later:** *if* we still want a YOLO ball detector, generate its training labels **from the motion tracker** (which gives correct ball positions), not from the broken auto-labeler. Train on the 884×670 crops. This is the only honest way to "train again for the ball."

---

## The four loops (each closes before the next opens)

### LOOP 1 — Lock single-camera ground truth  *(1–2 weeks)*
1. ~~Pick crop ROI~~ → **DONE, locked above.**
2. Confirm **10–15 real delivery arcs** visually from the 320 candidates (`deliveries.json`).
3. Calibrate the **pitch plane only**: mark the 4 crease/stump corners in the frame → homography pixel→pitch-mm. One plane, one camera = valid.
4. **Exit:** click a delivery → see the ball's 2D path mapped onto a real pitch diagram.

### LOOP 2 — Honest pitching + line  *(2–3 weeks)*
1. Detect **bounce point** from the tracked arc (lowest point / direction change).
2. Map pitching point + impact line onto the calibrated pitch plane.
3. Report **line decisions only** confidently (pitched in/outside line, impact in/outside line). **Height stays inconclusive.**
4. **Exit:** "pitched outside leg" type calls work and you'd trust them.

### LOOP 3 — Wire into the existing engine  *(2 weeks)*
1. Feed the 2D arc into `core/lbw_engine.py` (the authoritative one). **Archive the duplicate `lbw.py`.**
2. Replace YOLO ball detection with the motion tracker as the ball source.
3. Let the readiness gates run: one camera → low height confidence → `UMPIRE'S CALL` / `INCONCLUSIVE`. Correct behavior.
4. **Exit:** full delivery → decision → dashboard, end to end, on a real clip.

### LOOP 4 — Clean up, then decide on camera #2  *(ongoing)*
1. Resolve dead code flagged in audits: `lbw.py`, `sync.py`, `tracker.py`, old Electron root files.
2. **Then** decide: is a second fixed camera (square-leg) worth it? That is what unlocks real height/3D.
3. Until then: an honest **2D line-assist tool** for an academy — defensible and useful.

---

## Current status (2026-06-16)

```
[DONE]  Loop 1.1  crop ROI locked (0.27,0.28,0.73,0.90)
[DONE]  motion tracker works (real ball, straightness 1.00)
[DONE]  delivery auto-detection (320 candidates / clip, 74 in-corridor)
[DONE]  camera capture spec for future multi-cam (docs/CAMERA_SPEC.md)
[NOW]   Loop 1.2  confirm 10-15 real delivery arcs
[NEXT]  Loop 1.3  pitch-plane homography calibration
```
