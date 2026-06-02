import numpy as np

from core.audio_analyzer import AudioAnalyzer


def test_edge_transient_detected():
    analyzer = AudioAnalyzer()
    samples = np.random.default_rng(1).normal(0, 0.005, analyzer.sample_rate).astype(np.float32)
    samples[analyzer.sample_rate // 2] = 1.0
    analyzer.add_samples(samples, 0.0)
    result = analyzer.detect_edge_at(500.0)
    assert result.has_edge
    assert result.edge_confidence > 0


def test_clean_noise_no_edge():
    analyzer = AudioAnalyzer()
    samples = np.zeros(analyzer.sample_rate, dtype=np.float32)
    analyzer.add_samples(samples, 0.0)
    result = analyzer.detect_edge_at(500.0)
    assert not result.has_edge
