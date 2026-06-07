from pathlib import Path

import pytest

from core.testing_pipeline import AnalysisOptions, DeliveryTestingPipeline


SCENARIOS = [
    ("lbw_out_in_line.mp4", "OUT"),
    ("lbw_not_out_outside_leg.mp4", "NOT_OUT"),
    ("lbw_not_out_misses_stumps.mp4", "NOT_OUT"),
    ("lbw_inconclusive_tracking_lost.mp4", "REVIEW INCONCLUSIVE"),
]


@pytest.mark.parametrize(("clip", "expected"), SCENARIOS)
def test_lbw_decision(clip, expected, tmp_path):
    path = Path("tests/fixtures/scenarios") / clip
    if not path.exists():
        pytest.skip(f"Fixture {clip} not available")
    pipeline = DeliveryTestingPipeline()
    result = pipeline.process(f"e2e_{path.stem}", [path], AnalysisOptions(max_frames=90))
    actual = result["summary"]["lbw_recommendation"]
    assert actual == expected, (
        f"{clip}: expected {expected}, got {actual}\n"
        f"Gates: {result['summary'].get('gate', {})}"
    )
