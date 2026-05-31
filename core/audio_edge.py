"""Audio waveform and FFT based bat-edge detector."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from config.settings import (
    AUDIO_DIR,
    AUDIO_SAMPLE_RATE,
    EDGE_FREQ_HIGH_HZ,
    EDGE_FREQ_LOW_HZ,
    EDGE_SPIKE_THRESHOLD,
)
from utils.helpers import save_csv, save_json, timestamp_str


@dataclass(slots=True)
class EdgeEvent:
    timestamp_ms: float
    energy: float
    z_score: float
    probability: float


class AudioEdgeDetector:
    """Detects sharp high-frequency edge events near impact frames."""

    def __init__(
        self,
        sample_rate: int = AUDIO_SAMPLE_RATE,
        low_hz: float = EDGE_FREQ_LOW_HZ,
        high_hz: float = EDGE_FREQ_HIGH_HZ,
        threshold_z: float = EDGE_SPIKE_THRESHOLD,
    ) -> None:
        self.sample_rate = sample_rate
        self.low_hz = low_hz
        self.high_hz = high_hz
        self.threshold_z = threshold_z
        self.energy_history: list[float] = []
        self.events: list[EdgeEvent] = []

    def process_chunk(self, samples: np.ndarray, timestamp_ms: float) -> EdgeEvent | None:
        samples = np.asarray(samples, dtype=np.float32).reshape(-1)
        if samples.size == 0:
            return None
        samples -= np.mean(samples)
        window = np.hanning(samples.size)
        spectrum = np.abs(np.fft.rfft(samples * window))
        freqs = np.fft.rfftfreq(samples.size, d=1.0 / self.sample_rate)
        band = spectrum[(freqs >= self.low_hz) & (freqs <= self.high_hz)]
        energy = float(np.mean(band ** 2)) if band.size else 0.0

        baseline = np.asarray(self.energy_history[-120:], dtype=np.float32)
        mean = float(np.mean(baseline)) if baseline.size else energy
        std = float(np.std(baseline)) if baseline.size else 1.0
        z_score = (energy - mean) / max(std, 1e-6)
        probability = float(1.0 / (1.0 + np.exp(-(z_score - self.threshold_z))))

        self.energy_history.append(energy)
        if z_score >= self.threshold_z:
            event = EdgeEvent(timestamp_ms, energy, z_score, probability)
            self.events.append(event)
            return event
        return None

    def nearest_event_probability(self, video_timestamp_ms: float, window_ms: float = 35.0) -> float:
        probabilities = [
            event.probability for event in self.events if abs(event.timestamp_ms - video_timestamp_ms) <= window_ms
        ]
        return max(probabilities) if probabilities else 0.0

    def export(self, fmt: str = "json", stem: str | None = None) -> Path:
        stem = stem or timestamp_str()
        rows = [asdict(event) for event in self.events]
        path = AUDIO_DIR / f"edge_events_{stem}.{fmt.lower()}"
        if fmt.lower() == "csv":
            return save_csv(rows, path)
        return save_json(rows, path)
