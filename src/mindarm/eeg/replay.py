"""Safe NPZ playback with explicit end-of-file behavior."""

from pathlib import Path

import numpy as np


class ReplayEEGSource:
    """Replay data, sample_rate, n_channels arrays without loading pickled objects."""

    def __init__(self, path: Path, loop: bool = True) -> None:
        with np.load(path, allow_pickle=False) as archive:
            raw = archive["data"]
            rate = archive["sample_rate"]
            channels = archive["n_channels"]
            if rate.shape != () or channels.shape != ():
                raise ValueError("sample_rate and n_channels must be scalar arrays")
            if raw.dtype.kind not in "fiu" or rate.dtype.kind not in "fiu" or channels.dtype.kind not in "iu":
                raise ValueError("EEG data/rate must be numeric and n_channels an integer")
            self.sample_rate = float(rate)
            self.n_channels = int(channels)
            self._data = np.asarray(raw, dtype=np.float64)
        if not np.isfinite(self.sample_rate) or self.sample_rate <= 0 or self.n_channels < 1:
            raise ValueError("Invalid recording metadata")
        if self._data.ndim != 2 or self._data.shape[0] != self.n_channels or self._data.shape[1] == 0:
            raise ValueError("Recording must have shape (n_channels, nonzero samples)")
        if not np.isfinite(self._data).all():
            raise ValueError("Recording contains non-finite samples")
        self._offset = 0
        self._loop = loop

    def read(self, n_samples: int) -> np.ndarray:
        """Return exactly the requested samples or raise EOFError without advancing."""
        if isinstance(n_samples, bool) or not isinstance(n_samples, int) or n_samples < 1:
            raise ValueError("n_samples must be a positive integer")
        end = self._offset + n_samples
        size = self._data.shape[1]
        if not self._loop and end > size:
            raise EOFError("EEG recording exhausted")
        indices = np.arange(self._offset, end) % size
        self._offset = end % size if self._loop else end
        return self._data[:, indices].copy()
