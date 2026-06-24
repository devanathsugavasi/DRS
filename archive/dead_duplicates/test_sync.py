import numpy as np

from core.sync import SyncFrame, SyncManager


def test_sync_alignment_with_offset():
    manager = SyncManager()
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    manager.add_frame(SyncFrame(0, 100.0, 1, frame))
    manager.add_frame(SyncFrame(1, 106.0, 1, frame))
    manager.detect_audio_click_anchor({0: [(100.0, 1.0)], 1: [(106.0, 1.0)]})
    assert manager.get_sync_offset(1) == -6.0
    assert manager.get_aligned_frame(1, 100.0) is not None
