"""Double-blink detector, gesture decoder, EEG tap, and the live stream source (offline)."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pytest

from conftest import ScriptedDecoder
from mindarm.decoder.blink import BlinkDecoder, BlinkDetector, TappedEEGSource
from mindarm.eeg.fake import FakeEEGSource
from mindarm.eeg.stream import StreamEEGSource

FS = 250.0


def blink_signal(peak_times: list[float], seconds: float = 6.0, amplitude_v: float = 0.5) -> np.ndarray:
    """Forehead-like volts: 1.65 V offset, 60 Hz pickup, and 80 ms wide positive blink pulses."""
    t = np.arange(int(seconds * FS)) / FS
    signal = 1.65 + 0.02 * np.sin(2 * np.pi * 60 * t)
    for peak in peak_times:
        signal += amplitude_v * np.exp(-0.5 * ((t - peak) / 0.03) ** 2)
    return signal


def test_double_blink_is_one_gesture_with_its_gap() -> None:
    detector = BlinkDetector(FS)
    gestures = detector.feed(blink_signal([2.0, 2.3]))
    assert len(gestures) == 1
    assert gestures[0].amplitude_mv > 300
    assert gestures[0].gap_s == pytest.approx(0.3, abs=0.02)
    assert gestures[0].time_s == pytest.approx(2.3, abs=0.05)


@pytest.mark.parametrize("peaks", [[2.0], [2.0, 3.5], [2.0, 2.04]])
def test_single_far_or_merged_blinks_are_not_gestures(peaks: list[float]) -> None:
    assert BlinkDetector(FS).feed(blink_signal(peaks)) == []


def test_weak_blinks_stay_below_the_configured_threshold() -> None:
    assert BlinkDetector(FS, threshold_mv=300).feed(blink_signal([2.0, 2.3], amplitude_v=0.2)) == []
    assert len(BlinkDetector(FS, threshold_mv=150).feed(blink_signal([2.0, 2.3], amplitude_v=0.2))) == 1


def test_refractory_blocks_a_second_pair_but_not_a_later_one() -> None:
    assert len(BlinkDetector(FS).feed(blink_signal([2.0, 2.3, 3.0, 3.3]))) == 1
    assert len(BlinkDetector(FS).feed(blink_signal([2.0, 2.3, 4.5, 4.8], seconds=7))) == 2


def test_chunked_feed_matches_one_long_feed() -> None:
    signal = blink_signal([1.0, 1.3, 4.0, 4.4], seconds=7)
    whole = BlinkDetector(FS).feed(signal)
    chunked_detector = BlinkDetector(FS)
    chunked = [g for start in range(0, len(signal), 25) for g in chunked_detector.feed(signal[start:start + 25])]
    assert chunked == whole and len(whole) == 2


def test_detector_rejects_bad_parameters() -> None:
    with pytest.raises(ValueError):
        BlinkDetector(FS, min_gap_s=0.8, max_gap_s=0.1)
    with pytest.raises(ValueError):
        BlinkDetector(FS, threshold_mv=0)


def test_decoder_prefers_keys_and_reports_a_gesture_once() -> None:
    keyboard = ScriptedDecoder()
    records: list[dict[str, Any]] = []
    decoder = BlinkDecoder(keyboard, BlinkDetector(FS), record=records.append)
    assert decoder.poll() is None
    decoder.feed(blink_signal([2.0, 2.3])[None, :])
    keyboard.send("TOGGLE_ENGAGE")
    assert decoder.poll().kind == "TOGGLE_ENGAGE"
    assert decoder.poll().kind == "SELECT_TARGET"
    assert decoder.poll() is None
    assert records == [{"type": "gesture", "gesture": "double_blink", "signal_time_s": pytest.approx(2.3, abs=0.05),
                        "amplitude_mv": pytest.approx(records[0]["amplitude_mv"]), "gap_s": pytest.approx(0.3, abs=0.02)}]
    assert json.dumps(records[0])


def test_tapped_source_forwards_every_block_unchanged() -> None:
    seen: list[np.ndarray] = []
    tapped = TappedEEGSource(FakeEEGSource(seed=3), seen.append)
    block = tapped.read(25)
    assert tapped.sample_rate == 250.0 and tapped.n_channels == 1
    assert block.shape == (1, 25) and seen and seen[0] is block


class FakeResponse:
    """Iterable of SSE lines standing in for urllib's response."""

    def __init__(self, lines: list[bytes]) -> None:
        self._lines = lines

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def __iter__(self):
        return iter(self._lines)


def test_stream_source_reads_blocks_then_reports_loss(monkeypatch: pytest.MonkeyPatch) -> None:
    lines = [b": comment\n", b"data: " + json.dumps({"s": [1.6, 1.7, 1.8], "rate": 250}).encode() + b"\n",
             b"data: " + json.dumps({"s": [1.9, 2.0]}).encode() + b"\n"]
    monkeypatch.setattr("mindarm.eeg.stream.urllib.request.urlopen", lambda url, timeout: FakeResponse(lines))
    source = StreamEEGSource("http://127.0.0.1:1/stream", 250.0, connect_timeout=2)
    assert source.read(2).tolist() == [[1.6, 1.7]]
    assert source.read(3).tolist() == [[1.8, 1.9, 2.0]]
    with pytest.raises(RuntimeError):
        source.read(1)
    with pytest.raises(ValueError):
        source.read(0)


def test_stream_source_fails_setup_without_samples(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(url: str, timeout: float) -> FakeResponse:
        raise ConnectionRefusedError("viewer not running")
    monkeypatch.setattr("mindarm.eeg.stream.urllib.request.urlopen", refuse)
    with pytest.raises(OSError, match="viewer not running"):
        StreamEEGSource("http://127.0.0.1:1/stream", 250.0, connect_timeout=1)
