"""Pitch map tracking and line/length classification."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(slots=True)
class PitchDelivery:
    ball_id: str
    pitch_x_mm: float
    pitch_y_mm: float
    line_type: str
    length_type: str


class PitchMap:
    def __init__(self) -> None:
        self.deliveries: list[PitchDelivery] = []

    def add_delivery(self, ball_id: str, pitch_x_mm: float, pitch_y_mm: float) -> PitchDelivery:
        delivery = PitchDelivery(
            ball_id,
            pitch_x_mm,
            pitch_y_mm,
            self.classify_line(pitch_y_mm),
            self.classify_length(pitch_x_mm),
        )
        self.deliveries.append(delivery)
        return delivery

    def classify_line(self, pitch_y_mm: float) -> str:
        if pitch_y_mm > 170:
            return "OUTSIDE_OFF"
        if pitch_y_mm > 60:
            return "ON_OFF_STUMP"
        if pitch_y_mm > -60:
            return "MIDDLE_STUMP"
        if pitch_y_mm > -170:
            return "ON_LEG_STUMP"
        return "OUTSIDE_LEG"

    def classify_length(self, pitch_x_mm: float) -> str:
        if pitch_x_mm <= 0:
            return "FULL_TOSS"
        if pitch_x_mm < 1800:
            return "YORKER"
        if pitch_x_mm < 4200:
            return "FULL"
        if pitch_x_mm < 6500:
            return "GOOD_LENGTH"
        if pitch_x_mm < 8500:
            return "SHORT_OF_GOOD"
        if pitch_x_mm < 11000:
            return "SHORT"
        return "BOUNCER"

    def to_json(self) -> dict:
        return {"deliveries": [asdict(item) for item in self.deliveries]}
