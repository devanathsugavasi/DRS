# DRS — Camera & Footage Capture Spec

**Why this exists:** the Session-1 footage (single wide handheld stand camera, ball ~5 px) cannot support reliable LBW tracking — the ball merges with players. This spec defines how to film so the existing DRS pipeline (`core/`) works properly. Follow it for the next capture session.

---

## The one rule that matters most

**The ball must be at least ~30 px wide in the frame during a delivery.**
Everything below exists to achieve that. If the ball is <15 px, no software can track it reliably.

---

## Camera placement

DRS needs the ball seen from angles that resolve **line** (was it in line with stumps) and **height/impact**. Minimum viable = 2 cameras; better = 3–4.

| Cam | Position | Purpose | Priority |
|---|---|---|---|
| A | **High behind the bowler's arm**, aligned with the pitch centre line | Line: pitched-in-line, impact-in-line, hitting stumps | **Required** |
| B | **Side-on, level with the popping crease**, ~square of the pitch | Height, bounce point, impact height | **Required** |
| C | High behind the **batsman's** stumps (mirror of A) | Second line view / confirmation | Recommended |
| D | Square on the **other** side | Edge detection / second side view | Optional |

- Mount on **tripods, locked off** (no panning/zooming during a delivery). Fixed framing is what makes background subtraction + calibration work.
- Each camera frames **the full pitch length + both sets of stumps + a margin** above the stumps for high balls.

---

## Camera settings

| Setting | Target | Why |
|---|---|---|
| Resolution | 1920×1080 min (4K better) | More px on the ball |
| Frame rate | **≥ 100 fps** (120/240 ideal) | Ball travels ~30–40 m in <0.5 s; 50 fps = big gaps between positions. High fps = dense, trackable arc |
| Shutter speed | **Fast, ≥ 1/1000 s** | Freezes the ball — no motion blur smearing it into a streak |
| ISO / exposure | As low as light allows | Less noise → cleaner background subtraction |
| Focus | **Manual, locked** on the pitch | Autofocus hunts and blurs |
| Zoom | Tight enough that the **ball ≥ 30 px**, wide enough to keep both stumps + bounce area | The core trade-off |
| Format | High bitrate, low compression | AVCHD/H.264 at low bitrate adds blocky artifacts that look like motion |

---

## Calibration (do once per camera setup, before play)

The pipeline already supports this (`core/pitch_calibration.py`, `scripts/calibrate.py`).

1. Place a **checkerboard** (known square size) flat on the pitch at several positions; capture from each camera. → lens intrinsics.
2. Record the **real-world pitch landmarks** (stump bases, crease lines, pitch corners — known cricket dimensions: pitch 20.12 m, stumps 22.86 cm wide, 71.12 cm high) visible in each camera. → pixel→ground homography.
3. Keep cameras **locked** after calibration. Any bump invalidates it.

Result: pixel tracks convert to real pitch millimetres → real LBW geometry instead of heuristics.

---

## Lighting & scene

- **Daylight, even, no harsh shadows across the pitch** if possible. Hard shadow edges trigger false motion.
- Plain background behind the pitch helps (sightscreen). Crowd/movement behind the bowler adds noise.
- Avoid filming **into** the sun.

---

## Per-delivery capture checklist

- [ ] Cameras locked, focused, calibrated this session
- [ ] ≥100 fps, ≥1/1000 s shutter confirmed
- [ ] Ball measures ≥30 px in a test still
- [ ] All cameras started / time-sync clap or clapperboard at the start (for `core/synchronization.py`)
- [ ] Record continuously through bowler run-up → delivery → impact → follow-through

---

## Minimum cheap setup (if budget-limited)

- **2 phones** that shoot 1080p @ 120 fps (most modern phones do): one high behind bowler's arm, one side-on at crease.
- Both on tripods, locked, manual focus on pitch.
- Clap at the start for sync.
- This alone gets the ball to ~30–40 px and enables real line + height tracking.

---

**Bottom line:** the software is ready; it's currently starved of resolution and frame rate. Re-shoot to this spec and the existing pipeline produces real OUT / NOT OUT decisions.
