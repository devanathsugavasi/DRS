from core.appeal_manager import AppealManager


class Decision:
    def __init__(self, verdict):
        self.verdict = verdict


def test_review_counter_decrements_on_resolved_decision():
    manager = AppealManager()
    manager.request_review("fielding", "LBW")
    resolution = manager.resolve_review(Decision("OUT"))
    assert resolution.review_lost
    assert manager.get_reviews_remaining("fielding") == 1


def test_umpires_call_retains_review():
    manager = AppealManager()
    manager.request_review("fielding", "LBW")
    resolution = manager.resolve_review(Decision("UMPIRE_CALL"))
    assert not resolution.review_lost
    assert manager.get_reviews_remaining("fielding") == 2
