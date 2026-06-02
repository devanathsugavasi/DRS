"""MCC Law 36 aware LBW decision engine with confidence gates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


Verdict = Literal["OUT", "NOT_OUT", "UMPIRE_CALL", "REVIEW_INCONCLUSIVE"]


@dataclass(slots=True)
class LBWDecision:
    verdict: Verdict
    confidence: float
    pitched_in_line: bool
    impact_in_line: bool
    stump_hit_probability: float
    stump_hit_zone: str
    impact_height_mm: float
    ball_tracking_quality: float
    explanation: str


class LBWDecisionEngine:
    """Applies LBW decision rules without claiming accuracy from missing data."""

    def __init__(self, umpire_call_margin: float = 0.08) -> None:
        self.umpire_call_margin = umpire_call_margin

    def evaluate(
        self,
        pitch_y_mm: float | None,
        impact_y_mm: float | None,
        impact_height_mm: float | None,
        stump_hit_probability: float | None,
        tracking_quality: float,
        shot_attempted: bool | None = None,
        batter_is_right_handed: bool = True,
    ) -> LBWDecision:
        if None in {pitch_y_mm, impact_y_mm, impact_height_mm, stump_hit_probability}:
            return LBWDecision(
                "REVIEW_INCONCLUSIVE",
                0.0,
                False,
                False,
                0.0,
                "UNKNOWN",
                float(impact_height_mm or 0.0),
                tracking_quality,
                "LBW data is incomplete; calibration and impact detection are required.",
            )

        assert pitch_y_mm is not None and impact_y_mm is not None
        assert impact_height_mm is not None and stump_hit_probability is not None
        outside_leg = pitch_y_mm < -114.3 if batter_is_right_handed else pitch_y_mm > 114.3
        pitched_in_line = not outside_leg
        impact_in_line = abs(impact_y_mm) <= 114.3
        zone = self._stump_zone(impact_y_mm, impact_height_mm)

        if not pitched_in_line:
            return LBWDecision("NOT_OUT", 0.92, False, impact_in_line, stump_hit_probability, zone, impact_height_mm, tracking_quality, "Ball pitched outside leg stump.")
        if not impact_in_line and shot_attempted:
            return LBWDecision("NOT_OUT", 0.86, pitched_in_line, False, stump_hit_probability, zone, impact_height_mm, tracking_quality, "Impact outside line with a shot offered.")
        if tracking_quality < 0.58:
            return LBWDecision("REVIEW_INCONCLUSIVE", tracking_quality, pitched_in_line, impact_in_line, stump_hit_probability, zone, impact_height_mm, tracking_quality, "Tracking quality below decision threshold.")
        if stump_hit_probability >= 0.72:
            verdict: Verdict = "OUT"
        elif stump_hit_probability >= 0.50 - self.umpire_call_margin:
            verdict = "UMPIRE_CALL"
        else:
            verdict = "NOT_OUT"
        confidence = min(0.98, max(0.0, (stump_hit_probability * 0.75) + (tracking_quality * 0.25)))
        return LBWDecision(verdict, confidence, pitched_in_line, impact_in_line, stump_hit_probability, zone, impact_height_mm, tracking_quality, self._explain(verdict))

    def _stump_zone(self, impact_y_mm: float, impact_height_mm: float) -> str:
        vertical = "TOP" if impact_height_mm > 520 else "BOTTOM" if impact_height_mm < 230 else "MIDDLE"
        horizontal = "OFF" if impact_y_mm > 45 else "LEG" if impact_y_mm < -45 else "MIDDLE"
        return f"{vertical}_{horizontal}"

    def _explain(self, verdict: Verdict) -> str:
        if verdict == "OUT":
            return "Pitched legal, impact acceptable, and projected path clearly hits stumps."
        if verdict == "UMPIRE_CALL":
            return "Projected stump impact is within margin-of-error zone."
        if verdict == "NOT_OUT":
            return "Projected path does not clearly satisfy LBW criteria."
        return "Review inconclusive due to insufficient evidence."
