"""DRS appeal review management rules."""

from __future__ import annotations

from dataclasses import dataclass
import time


@dataclass(slots=True)
class ReviewRequest:
    team: str
    appeal_type: str
    timestamp: float
    reviews_remaining: int


@dataclass(slots=True)
class ReviewResolution:
    team: str
    verdict: str
    review_lost: bool
    reviews_remaining: int
    reason: str


class AppealManager:
    MAX_REVIEWS_PER_INNINGS = 2

    def __init__(self) -> None:
        self.reviews = {"batting": self.MAX_REVIEWS_PER_INNINGS, "fielding": self.MAX_REVIEWS_PER_INNINGS}
        self.active: ReviewRequest | None = None
        self.history: list[ReviewResolution] = []

    def request_review(self, team: str, appeal_type: str) -> ReviewRequest | None:
        key = team.lower()
        if self.reviews.get(key, 0) <= 0:
            return None
        self.active = ReviewRequest(key, appeal_type.upper(), time.time(), self.reviews[key])
        return self.active

    def resolve_review(self, decision) -> ReviewResolution:
        if self.active is None:
            raise ValueError("No active review to resolve")
        verdict = getattr(decision, "verdict", None) or getattr(decision, "decision", None) or str(decision)
        review_lost = verdict not in {"UMPIRE_CALL", "UMPIRE'S CALL", "REVIEW_INCONCLUSIVE"}
        if review_lost:
            self.reviews[self.active.team] = max(0, self.reviews[self.active.team] - 1)
        resolution = ReviewResolution(
            self.active.team,
            verdict,
            review_lost,
            self.reviews[self.active.team],
            "Umpire's Call/inconclusive retains review." if not review_lost else "Review consumed by resolved decision.",
        )
        self.history.append(resolution)
        self.active = None
        return resolution

    def get_reviews_remaining(self, team: str) -> int:
        return self.reviews.get(team.lower(), 0)

    def get_session_summary(self) -> dict:
        return {
            "reviews_remaining": dict(self.reviews),
            "active_review": self.active,
            "history": [item.__dict__ for item in self.history],
        }
