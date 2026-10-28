"""Session recordings using the same NPZ contract as replay."""

from pathlib import Path

import numpy as np

from mindarm.interfaces import EEGSource


class EEGRecording:
    """Buffer finite acquisition blocks and write a pickle-free archive on shutdown."""

    def __init__(self, source: EEGSource, path: Path, enabled: bool = True) -> None:
        self._source = source
        self.path = path
        self._enabled = enabled
        self._chunks: list[np.ndarray] = []

    def capture(self, n_samples: int) -> None:
        """Read and validate a block even if recording has been disabled."""
        samples = self._source.read(n_samples)
        if samples.shape != (self._source.n_channels, n_samples) or not np.isfinite(samples).all():
            raise ValueError("EEG source returned invalid shape or non-finite data")
        if self._enabled:
            self._chunks.append(samples.copy())

    def save(self) -> Path | None:
        """Save sampled sessions; return None if no samples were recorded."""
        if not self._chunks:
            return None
        self.path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(self.path, data=np.concatenate(self._chunks, axis=1),
                            sample_rate=np.array(self._source.sample_rate, dtype=np.float64),
                            n_channels=np.array(self._source.n_channels, dtype=np.int64))
        self._chunks.clear()
        return self.path
