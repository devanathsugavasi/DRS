"""Map pipeline outputs to Electron dashboard decision payloads."""

from __future__ import annotations

from datetime import datetime
from typing import Any


def map_summary_to_dashboard_decision(summary: dict[str, Any], job_id: str | None = None) -> dict[str, Any]:
    """Convert testing pipeline summary into broadcast dashboard decision state."""
    raw = summary.get("raw_lbw_recommendation", summary.get("lbw_recommendation", "REVIEW INCONCLUSIVE"))
    status = str(raw).replace(" ", "_").upper()
    if status == "NOT_OUT":
        status = "NOT_OUT"
    lbw = summary.get("lbw_engine", {})
    trajectory = summary.get("trajectory_3d") or []
    bounce = summary.get("pitching_location_mm")
    impact = summary.get("impact_location_mm")
    extension = summary.get("trajectory_3d") or []

    return {
        "status": status if status in {"OUT", "NOT_OUT", "UMPIRE_CALL", "PROCESSING"} else "REVIEW_INCONCLUSIVE",
        "outcome": summary.get("lbw_recommendation", "Review inconclusive"),
        "time": datetime.now().isoformat(timespec="seconds"),
        "over": "--",
        "ball": "--",
        "decision": summary.get("lbw_recommendation", "REVIEW INCONCLUSIVE"),
        "job_id": job_id,
        "ball_confidence": summary.get("confidence_score"),
        "tracking_confidence": summary.get("confidence_score"),
        "calibration_confidence": summary.get("calibration_confidence"),
        "prediction_confidence": lbw.get("stump_hit_probability"),
        "model_confidence": summary.get("model_metrics", {}).get("map50"),
        "overall_confidence": summary.get("confidence_score"),
        "impact_point": _mm_to_xyz(impact),
        "impact_marker": _mm_to_xyz(impact),
        "bounce_point": _mm_to_xyz(bounce),
        "wicket_zone_status": summary.get("wicket_status", "--"),
        "ball_speed_kmh": summary.get("ball_speed_kmh"),
        "trajectory": trajectory,
        "predicted_extension": extension,
        "wicket_prediction": {
            "stump": lbw.get("stump_hit_zone", "UNKNOWN"),
            "umpire_call": raw == "UMPIRE'S CALL",
            "collision": extension[-1] if extension else None,
        },
        "pitching_status": summary.get("pitching_status"),
        "impact_status": summary.get("impact_status"),
        "wicket_status": summary.get("wicket_status"),
        "timeline": _timeline_from_summary(summary),
        "explanation": " ".join(summary.get("reasoning", [])) or "Analysis complete.",
        "edge_analysis": summary.get("edge_analysis", {}),
        "hotspot_analysis": summary.get("hotspot_analysis", {}),
    }


def _mm_to_xyz(point: dict[str, float] | None) -> dict[str, float] | None:
    if not point:
        return None
    return {
        "x": float(point.get("along_mm", 0.0)) / 1000.0,
        "y": float(point.get("lateral_mm", 0.0)) / 1000.0,
        "z": 0.05,
    }


def _timeline_from_summary(summary: dict[str, Any]) -> list[dict[str, str]]:
    bounce = summary.get("pitching_location_mm")
    impact = summary.get("impact_location_mm")
    decision = summary.get("lbw_recommendation", "Pending")
    return [
        {"label": "Appeal", "status": "complete"},
        {"label": "Ball Detected", "status": "complete"},
        {"label": "Bounce Detected", "status": "complete" if bounce else "pending"},
        {"label": "Impact Detected", "status": "complete" if impact else "pending"},
        {"label": "Wicket Predicted", "status": "complete"},
        {"label": "Decision Generated", "status": "active"},
        {"label": decision, "status": "complete" if decision not in {"REVIEW INCONCLUSIVE", "Pending"} else "pending"},
    ]
