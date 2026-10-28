"""Deterministic synthetic EEG for acquisition and replay development."""

import math

import numpy as np


class FakeEEGSource:
    """Continuous sine plus Gaussian noise, with optional periodic alpha bursts."""

    def __init__(
        self, sample_rate: float = 250.0, n_channels: int = 1,
        frequency: float = 12.0, noise_std: float = 0.2,
        alpha_bursts: bool = False, seed: int | None = None,
    ) -> None:
        if not all(math.isfinite(x) for x in (sample_rate, frequency, noise_std)):
            raise ValueError("Signal parameters must be finite")
        if sample_rate <= 0 or not 0 < frequency < sample_rate / 2 or noise_std < 0:
            raise ValueError("Require positive rate, frequency below Nyquist, nonnegative noise")
        if alpha_bursts and sample_rate <= 20:
            raise ValueError("10 Hz alpha bursts require sample_rate > 20")
        if isinstance(n_channels, bool) or not isinstance(n_channels, int) or n_channels < 1:
            raise ValueError("n_channels must be a positive integer")
        self.sample_rate = float(sample_rate)
        self.n_channels = n_channels
        self.frequency = frequency
        self.noise_std = noise_std
        self.alpha_bursts = alpha_bursts
        self._rng = np.random.default_rng(seed)
        self._offset = 0

    def read(self, n_samples: int) -> np.ndarray:
        """Generate the next block, preserving phase across calls."""
        if isinstance(n_samples, bool) or not isinstance(n_samples, int) or n_samples < 1:
            raise ValueError("n_samples must be a positive integer")
        time = (np.arange(n_samples) + self._offset) / self.sample_rate
        wave = np.sin(2 * np.pi * self.frequency * time)
        if self.alpha_bursts:
            # Two seconds of high-amplitude 10 Hz alpha every six seconds.
            wave += (time % 6 >= 4) * 4 * np.sin(2 * np.pi * 10 * time)
        # Sample-major noise makes chunked reads equivalent to one long read.
        noise = self._rng.normal(0, self.noise_std, (n_samples, self.n_channels)).T
        self._offset += n_samples
        return np.asarray(wave[None, :] + noise, dtype=np.float64)
