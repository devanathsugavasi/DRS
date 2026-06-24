from core.tracker import ExtendedCricketBallTracker


def test_ekf_tracks_synthetic_path():
    tracker = ExtendedCricketBallTracker()
    truth = []
    for idx in range(20):
        x = 100 + idx * 8
        y = 200 - idx * 2 + 0.2 * idx * idx
        truth.append((x, y))
        tracker.update((x - 4, y - 4, x + 4, y + 4), 0, idx * 16.6)
    latest = tracker.get_trajectory_3d()[-1]
    true_x, true_y = truth[-1]
    assert abs(latest.x - true_x) / true_x < 0.05
    assert abs(latest.y - true_y) / true_y < 0.05
    assert tracker.get_ball_velocity_kph() >= 0
