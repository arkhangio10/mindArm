"""Live single-channel EEG read from the laptop viewer's server-sent event stream."""

from __future__ import annotations

import json
import threading
import time
import urllib.request
from collections import deque

import numpy as np


class StreamEEGSource:
    """Buffer ``data: {"s": [volts, ...]}`` lines from an HTTP stream; keep only the newest samples.

    The buffer holds at most ``max_buffer_s`` seconds so an idle orchestrator never reads
    stale signal when it reengages. ``read`` waits for the requested block (at most the
    block duration plus two seconds) and raises ``RuntimeError`` if the stream dies.
    """

    def __init__(self, url: str, sample_rate: float = 250.0, max_buffer_s: float = 0.5,
                 connect_timeout: float = 5.0) -> None:
        if sample_rate <= 0 or max_buffer_s <= 0:
            raise ValueError("Require a positive sample rate and buffer length")
        self.sample_rate = float(sample_rate)
        self.n_channels = 1
        self.url = url
        self._buffer: deque[float] = deque(maxlen=max(1, int(max_buffer_s * sample_rate)))
        self._lock = threading.Lock()
        self._ready = threading.Event()
        self._error: Exception | None = None
        self._received = False
        self._thread = threading.Thread(target=self._pump, name="mindarm-eeg-stream", daemon=True)
        self._thread.start()
        if not self._ready.wait(connect_timeout) or not self._received:
            raise OSError(f"No EEG samples from {url} within {connect_timeout:.0f} s: {self._error}")

    def _pump(self) -> None:
        try:
            with urllib.request.urlopen(self.url, timeout=10) as response:
                for raw in response:
                    line = raw.decode("utf-8", errors="replace")
                    if not line.startswith("data:"):
                        continue
                    samples = json.loads(line[5:]).get("s", [])
                    if not samples:
                        continue
                    with self._lock:
                        self._buffer.extend(float(value) for value in samples)
                    self._received = True
                    self._ready.set()
            self._error = EOFError("EEG stream ended")
        except Exception as error:  # noqa: BLE001 - any failure must surface on the next read
            self._error = error
        finally:
            self._ready.set()

    def read(self, n_samples: int) -> np.ndarray:
        """Return the next ``(1, n_samples)`` block in volts, waiting briefly for it to arrive."""
        if isinstance(n_samples, bool) or not isinstance(n_samples, int) or n_samples < 1:
            raise ValueError("n_samples must be a positive integer")
        deadline = time.monotonic() + n_samples / self.sample_rate + 2.0
        while True:
            with self._lock:
                if len(self._buffer) >= n_samples:
                    block = [self._buffer.popleft() for _ in range(n_samples)]
                    return np.asarray(block, dtype=np.float64)[None, :]
            if self._error is not None:
                raise RuntimeError(f"EEG stream lost: {self._error}")
            if time.monotonic() > deadline:
                raise RuntimeError("EEG stream stalled")
            time.sleep(0.005)
