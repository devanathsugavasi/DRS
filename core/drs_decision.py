"""Production DRS decision path: calibration, trajectory, and LBW engine."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np

from config.settings import STUMP_HEIGHT_M, STUMP_WIDTH_M
from core.lbw_engine import LBWDecision, LBWDecisionEngine
from core.pitch_calibration import ManualPitchCalibrator
from core.trajectory import TrajectoryPredictor

STUMP_HALF_WIDTH_MM = (STUMP_WIDTH_M / 2.0) * 1000.0
STUMP_HEIGHT_MM = 711.0
DEFAULT_BALL_HEIGHT_M = 0.12


class DRSDecisionService:
    """Builds LBW decisions from tracked pixels using saved pitch calibration."""

    def __init__(self) -> None:
        self.calibrator = ManualPitchCalibrator()
        self.lbw = LBWDecisionEngine()
        self.trajectory = TrajectoryPredictor()

    def calibration_quality(self) -> tuple[float, str, bool]:
        profiles = self.calibrator.list_profiles()
        if not profiles:
            return 0.0, "No pitch calibration profiles saved.", False
        errors = [item.homography_error_cm for item in profiles if item.homography_error_cm is not None]
        avg_error = float(np.mean(errors)) if errors else 5.0
        score = max(0.0, min(1.0, 1.0 - (avg_error / 5.0)))
        return round(score, 3), f"{len(profiles)} camera profile(s), avg homography error {avg_error:.2f} cm", True

    def pixel_to_pitch_mm(self, camera_id: int, px: float, py: float) -> tuple[float, float] | None:
        profile = self.calibrator.load_profile(camera_id)
        if profile is None or not profile.homography:
            return None
        wx, wy = self.calibrator.pixel_to_world(px, py, profile.homography)
        lateral_mm = (wx - (STUMP_WIDTH_M / 2.0)) * 1000.0
        along_mm = wy * 1000.0
        return lateral_mm, along_mm

    def tracks_to_pitch_path(
        self,
        tracks: list[dict[str, Any]],
        camera_id: int,
    ) -> list[dict[str, float]]:
        path: list[dict[str, float]] = []
        for point in tracks:
            mapped = self.pixel_to_pitch_mm(camera_id, float(point["x"]), float(point["y"]))
            if mapped is None:
                continue
            lateral_mm, along_mm = mapped
            path.append(
                {
                    "lateral_mm": lateral_mm,
                    "along_mm": along_mm,
                    "confidence": float(point.get("confidence", 0.0)),
                    "timestamp_ms": float(point.get("timestamp_ms", 0.0)),
                    "frame_id": int(point.get("frame_id", 0)),
                }
            )
        return path

    def estimate_bounce(self, pitch_path: list[dict[str, float]]) -> dict[str, float] | None:
        if len(pitch_path) < 5:
            return None
        along = np.array([item["along_mm"] for item in pitch_path], dtype=float)
        deltas = np.diff(along)
        candidates = np.where(np.diff(np.sign(deltas)) != 0)[0]
        if candidates.size == 0:
            idx = int(len(pitch_path) * 0.45)
        else:
            idx = int(candidates[0]) + 1
        idx = max(0, min(idx, len(pitch_path) - 1))
        return pitch_path[idx]

    def estimate_impact(
        self,
        pitch_path: list[dict[str, float]],
        bounce: dict[str, float] | None,
        impact_px: list[int] | None,
        camera_id: int,
    ) -> dict[str, float] | None:
        if impact_px and len(impact_px) == 2:
            mapped = self.pixel_to_pitch_mm(camera_id, float(impact_px[0]), float(impact_px[1]))
            if mapped:
                return {"lateral_mm": mapped[0], "along_mm": mapped[1]}
        if bounce is None:
            return None
        bounce_idx = pitch_path.index(bounce)
        for item in pitch_path[bounce_idx:]:
            if item["along_mm"] > bounce["along_mm"] + 80:
                return item
        return pitch_path[-1] if pitch_path else None

    def stump_hit_probability(
        self,
        pitch_path: list[dict[str, float]],
        bounce: dict[str, float] | None,
    ) -> tuple[float, list[dict[str, float]]]:
        if len(pitch_path) < 3:
            return 0.0, []
        positions_m = []
        times_s = []
        t0 = pitch_path[0]["timestamp_ms"] / 1000.0
        for item in pitch_path[-24:]:
            positions_m.append((item["along_mm"] / 1000.0, item["lateral_mm"] / 1000.0, DEFAULT_BALL_HEIGHT_M))
            times_s.append((item["timestamp_ms"] / 1000.0) - t0)
        try:
            prediction = self.trajectory.predict_from_world_points(
                positions_m,
                times_s,
                wicket_x_m=positions_m[-1][0] + 2.0,
                stump_half_width_m=STUMP_WIDTH_M / 2.0,
                stump_height_m=STUMP_HEIGHT_M / 1000.0,
            )
        except ValueError:
            return 0.0, []
        extension = [
            {"x": point.x, "y": point.y, "z": point.z}
            for point in prediction.points[:: max(1, len(prediction.points) // 12)]
        ]
        if prediction.wicket_collision:
            return 0.86, extension
        if bounce and abs(bounce["lateral_mm"]) <= STUMP_HALF_WIDTH_MM:
            return 0.64, extension
        return 0.28, extension

    def build_decision(
        self,
        fused_tracks: list[dict[str, Any]],
        camera_results: list[dict[str, Any]],
        dual: bool,
        calibration_readiness: Any,
        sync_readiness: Any,
        readiness_gate: Any,
        model_dict: dict[str, Any],
        edge_analysis: dict[str, Any] | None = None,
        hotspot_analysis: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        main = camera_results[0]
        camera_id = int(main.get("camera_id", 0)) + 1
        pitch_path = self.tracks_to_pitch_path(fused_tracks or main.get("tracking_points", []), camera_id)
        cal_score, cal_reason, calibrated = self.calibration_quality()
        bounce = self.estimate_bounce(pitch_path) if pitch_path else None
        impact = self.estimate_impact(pitch_path, bounce, main.get("impact_point_px"), camera_id)
        tracking_quality = float(np.mean([cam.get("confidence", 0.0) for cam in camera_results])) if camera_results else 0.0
        stump_prob, extension = self.stump_hit_probability(pitch_path, bounce)

        if not calibrated or bounce is None or impact is None:
            lbw = self.lbw.evaluate(None, None, None, None, tracking_quality)
            proposed = "REVIEW INCONCLUSIVE"
            gate_confidence = tracking_quality
        else:
            impact_height_mm = 350.0 if impact["along_mm"] > (bounce["along_mm"] + 120) else 520.0
            lbw = self.lbw.evaluate(
                bounce["lateral_mm"],
                impact["lateral_mm"],
                impact_height_mm,
                stump_prob,
                tracking_quality,
            )
            proposed = self._verdict_label(lbw.verdict)
            gate_confidence = lbw.confidence

        quality = [cam.get("tracking_quality", {}) for cam in camera_results]
        gate = readiness_gate.evaluate(proposed, gate_confidence, model_dict, quality, calibration_readiness, sync_readiness)
        display = gate.display_decision
        ball_speed = float(np.mean([cam["ball_speed_kmh"] for cam in camera_results])) if camera_results else 0.0

        pitching_status = self._line_status(bounce["lateral_mm"] if bounce else None)
        impact_status = self._line_status(impact["lateral_mm"] if impact else None)
        wicket_status = "HITTING" if stump_prob >= 0.72 else "MISSING" if stump_prob < 0.5 else "UMPIRE_CALL"

        return {
            "ball_speed_kmh": round(ball_speed, 2),
            "pitching_location": self._format_mm_point(bounce),
            "impact_location": self._format_mm_point(impact),
            "pitching_location_mm": bounce,
            "impact_location_mm": impact,
            "predicted_wicket_impact": wicket_status.lower(),
            "pitching_status": pitching_status,
            "impact_status": impact_status,
            "wicket_status": wicket_status,
            "raw_lbw_recommendation": proposed,
            "lbw_recommendation": display,
            "confidence_score": round(gate_confidence, 3),
            "uncertainty": round(1.0 - gate_confidence, 3),
            "reliability": "high" if tracking_quality >= 0.78 else "medium" if tracking_quality >= 0.58 else "low",
            "tracking_quality": quality,
            "gate": gate.to_dict(),
            "model_metrics": model_dict,
            "calibration_metrics": calibration_readiness.to_dict(),
            "sync_metrics": sync_readiness.to_dict(),
            "calibration_confidence": cal_score,
            "calibration_reason": cal_reason,
            "lbw_engine": asdict(lbw),
            "trajectory_pitch_mm": pitch_path,
            "trajectory_3d": extension,
            "edge_analysis": edge_analysis or {},
            "hotspot_analysis": hotspot_analysis or {},
            "reasoning": self._build_reasoning(lbw, gate, cal_score, tracking_quality, dual),
            "notes": [
                "Decisions use lbw_engine.py with calibrated pitch coordinates when profiles exist.",
                "OUT/NOT OUT is hidden unless readiness gates pass.",
            ],
        }

    def _verdict_label(self, verdict: str) -> str:
        if verdict == "OUT":
            return "OUT"
        if verdict == "NOT_OUT":
            return "NOT OUT"
        if verdict == "UMPIRE_CALL":
            return "UMPIRE'S CALL"
        return "REVIEW INCONCLUSIVE"

    def _line_status(self, lateral_mm: float | None) -> str:
        if lateral_mm is None:
            return "UNKNOWN"
        if lateral_mm < -STUMP_HALF_WIDTH_MM:
            return "OUTSIDE LEG" if lateral_mm < 0 else "OUTSIDE OFF"
        if abs(lateral_mm) <= STUMP_HALF_WIDTH_MM:
            return "IN LINE"
        return "OUTSIDE OFF" if lateral_mm > 0 else "OUTSIDE LEG"

    def _format_mm_point(self, point: dict[str, float] | None) -> Any:
        if not point:
            return "unknown"
        return [round(point["lateral_mm"], 1), round(point["along_mm"], 1)]

    def _build_reasoning(
        self,
        lbw: LBWDecision,
        gate: Any,
        cal_score: float,
        tracking_quality: float,
        dual: bool,
    ) -> list[str]:
        lines = [lbw.explanation]
        if cal_score < 0.6:
            lines.append("Calibration quality is below target; import or re-mark pitch points.")
        if tracking_quality < 0.58:
            lines.append("Tracking quality is limiting decision confidence.")
        if dual:
            lines.append("Dual-camera fusion improved trajectory stability.")
        if gate.failed_gates:
            lines.append("Failed gates: " + ", ".join(gate.failed_gates))
        return lines
