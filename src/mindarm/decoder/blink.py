"""Voluntary double-blink gesture captured with electrodes; it is eye movement, not brain activity."""

from __future__ import annotations

import logging
import math
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from mindarm.interfaces import Decoder, DecoderEvent, EEGSource

logger = logging.getLogger(__name__)


class _Biquad:
    """Direct-form II transposed second-order section."""

    def __init__(self, b: tuple[float, float, float], a: tuple[float, float, float]) -> None:
        self._b, self._a = b, a
        self._z1 = self._z2 = 0.0

    @classmethod
    def notch(cls, sample_rate: float, frequency: float, q: float = 30.0) -> _Biquad:
        w0 = 2 * math.pi * frequency / sample_rate
        alpha, cosine = math.sin(w0) / (2 * q), math.cos(w0)
        a0 = 1 + alpha
        return cls((1 / a0, -2 * cosine / a0, 1 / a0), (1.0, -2 * cosine / a0, (1 - alpha) / a0))

    def step(self, x: float) -> float:
        y = self._b[0] * x + self._z1
        self._z1 = self._b[1] * x - self._a[1] * y + self._z2
        self._z2 = self._b[2] * x - self._a[2] * y
        return y


class _MovingMean:
    """Running mean over the last ``length`` samples."""

    def __init__(self, length: int) -> None:
        self._window: deque[float] = deque(maxlen=max(1, length))
        self._sum = 0.0

    def step(self, x: float) -> float:
        if len(self._window) == self._window.maxlen:
            self._sum -= self._window[0]
        self._window.append(x)
        self._sum += x
        return self._sum / len(self._window)


@dataclass(frozen=True)
class BlinkGesture:
    """One detected double blink: time of the second peak, its amplitude, and the peak gap."""

    time_s: float
    amplitude_mv: float
    gap_s: float


class BlinkDetector:
    """Fixed per-sample rule on one channel in volts (parameters come from configuration).

    Mains notch at 60 and 120 Hz, a one-second moving-mean baseline, 40 ms smoothing, then
    positive peaks above ``threshold_mv`` that are at most ``max_width_s`` wide. Two such
    peaks ``min_gap_s``..``max_gap_s`` apart form a gesture, at most once per ``refractory_s``.
    """

    def __init__(self, sample_rate: float, threshold_mv: float = 300.0, min_gap_s: float = 0.1,
                 max_gap_s: float = 0.8, refractory_s: float = 1.5, max_width_s: float = 0.4,
                 mains_hz: float = 60.0) -> None:
        if sample_rate <= 0 or threshold_mv <= 0 or refractory_s <= 0 or max_width_s <= 0:
            raise ValueError("Require positive sample rate, threshold, refractory, and width")
        if not 0 < min_gap_s < max_gap_s:
            raise ValueError("Require 0 < min_gap_s < max_gap_s")
        if not 0 < mains_hz < sample_rate / 4:
            raise ValueError("Mains frequency and its harmonic must lie below Nyquist")
        self.sample_rate = float(sample_rate)
        self.threshold_mv, self.min_gap_s, self.max_gap_s = threshold_mv, min_gap_s, max_gap_s
        self.refractory_s, self.max_width_s = refractory_s, max_width_s
        self._notches = [_Biquad.notch(sample_rate, mains_hz), _Biquad.notch(sample_rate, 2 * mains_hz)]
        self._baseline = _MovingMean(int(round(sample_rate)))
        self._smooth = _MovingMean(int(round(0.04 * sample_rate)))
        self._index = 0
        self._run_start: int | None = None
        self._run_peak = (0, 0.0)
        self._last_peak: tuple[float, float] | None = None
        self._last_event = -math.inf

    def feed(self, samples: np.ndarray) -> list[BlinkGesture]:
        """Consume one channel's samples in volts and return the gestures completed inside them."""
        gestures: list[BlinkGesture] = []
        for raw in np.asarray(samples, dtype=np.float64).ravel():
            value = raw
            for notch in self._notches:
                value = notch.step(value)
            level = self._smooth.step(value - self._baseline.step(value)) * 1000.0
            if level > self.threshold_mv:
                if self._run_start is None:
                    self._run_start, self._run_peak = self._index, (self._index, level)
                elif level > self._run_peak[1]:
                    self._run_peak = (self._index, level)
            elif self._run_start is not None:
                gesture = self._end_run()
                if gesture is not None:
                    gestures.append(gesture)
            self._index += 1
        return gestures

    def _end_run(self) -> BlinkGesture | None:
        assert self._run_start is not None
        width = (self._index - self._run_start) / self.sample_rate
        self._run_start = None
        if width > self.max_width_s:
            return None
        peak_time, amplitude = self._run_peak[0] / self.sample_rate, self._run_peak[1]
        previous, self._last_peak = self._last_peak, (peak_time, amplitude)
        if previous is None:
            return None
        gap = peak_time - previous[0]
        if not self.min_gap_s <= gap <= self.max_gap_s or peak_time - self._last_event <= self.refractory_s:
            return None
        self._last_event = peak_time
        return BlinkGesture(time_s=peak_time, amplitude_mv=max(amplitude, previous[1]), gap_s=gap)


class BlinkDecoder:
    """Keyboard events first (keys always cancel); otherwise one pending gesture as SELECT_TARGET."""

    def __init__(self, keyboard: Decoder, detector: BlinkDetector,
                 record: Callable[[dict], None] | None = None) -> None:
        self._keyboard, self._detector, self._record = keyboard, detector, record
        self._pending: BlinkGesture | None = None

    def feed(self, samples: np.ndarray) -> None:
        """Run the detector on an acquired block; keep only the newest gesture until polled."""
        for gesture in self._detector.feed(samples[0]):
            self._pending = gesture
            logger.info("Double blink detected: %.0f mV, peaks %.2f s apart", gesture.amplitude_mv, gesture.gap_s)
            if self._record is not None:
                self._record({"type": "gesture", "gesture": "double_blink", "signal_time_s": gesture.time_s,
                              "amplitude_mv": gesture.amplitude_mv, "gap_s": gesture.gap_s})

    def poll(self) -> DecoderEvent | None:
        event = self._keyboard.poll()
        if event is not None or self._pending is None:
            return event
        self._pending = None
        return DecoderEvent(kind="SELECT_TARGET")

    def close(self) -> None:
        close = getattr(self._keyboard, "close", None)
        if close is not None:
            close()


class TappedEEGSource:
    """Forward reads from a source and hand every acquired block to a consumer."""

    def __init__(self, source: EEGSource, consumer: Callable[[np.ndarray], None]) -> None:
        self._source, self._consumer = source, consumer

    @property
    def sample_rate(self) -> float:
        return self._source.sample_rate

    @property
    def n_channels(self) -> int:
        return self._source.n_channels

    def read(self, n_samples: int) -> np.ndarray:
        block = self._source.read(n_samples)
        self._consumer(block)
        return block
