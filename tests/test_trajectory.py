from core.trajectory import TrajectoryPoint, TrajectoryPredictor


def test_wicket_collision_interpolates_between_samples() -> None:
    predictor = TrajectoryPredictor()
    points = [
        TrajectoryPoint(0.00, -0.04, 0.0, 0.2, 1.0, 0.0, 0.0),
        TrajectoryPoint(0.01, 0.04, 0.0, 0.2, 1.0, 0.0, 0.0),
    ]

    collision = predictor._find_wicket_collision(points, 0.0, 0.1143, 0.711)

    assert collision is not None
    assert collision.x == 0.0
    assert collision.z == 0.2
