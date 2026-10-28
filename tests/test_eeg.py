"""Signal continuity, recording integrity, and replay edge cases."""

from pathlib import Path

import numpy as np
import pytest

from mindarm.eeg.fake import FakeEEGSource
from mindarm.eeg.recording import EEGRecording
from mindarm.eeg.replay import ReplayEEGSource


def test_fake_shape_frequency_and_continuity() -> None:
    source = FakeEEGSource(n_channels=2, noise_std=0.0, seed=1)
    samples = source.read(250)
    assert samples.shape == (2, 250)
    assert np.argmax(abs(np.fft.rfft(samples[0]))) == 12
    full = FakeEEGSource(n_channels=2, seed=3).read(100)
    split = FakeEEGSource(n_channels=2, seed=3)
    np.testing.assert_allclose(np.concatenate([split.read(37), split.read(63)], axis=1), full)


def test_alpha_burst_has_high_10_hz_amplitude() -> None:
    data = FakeEEGSource(noise_std=0.0, alpha_bursts=True).read(1500)[0]
    baseline = abs(np.fft.rfft(data[:500]))
    burst = abs(np.fft.rfft(data[1000:]))
    assert burst[20] > 10 * max(baseline[20], 1)


@pytest.mark.parametrize("count", [0, -1, True, 1.5])
def test_invalid_sample_count(count: int) -> None:
    with pytest.raises(ValueError):
        FakeEEGSource().read(count)


def test_record_replay_roundtrip_and_eof(tmp_path: Path) -> None:
    source = FakeEEGSource(n_channels=2, seed=5)
    recording = EEGRecording(source, tmp_path / "record.npz")
    recording.capture(20)
    recording.capture(30)
    assert recording.save() == recording.path
    replay = ReplayEEGSource(recording.path, loop=False)
    assert replay.sample_rate == 250.0
    assert replay.n_channels == 2
    with pytest.raises(EOFError):
        replay.read(51)
    expected = FakeEEGSource(n_channels=2, seed=5).read(50)
    np.testing.assert_array_equal(replay.read(50), expected)
    with pytest.raises(EOFError):
        replay.read(1)
    looping = ReplayEEGSource(recording.path)
    np.testing.assert_array_equal(looping.read(120), np.tile(expected, (1, 3))[:, :120])


@pytest.mark.parametrize("data,rate,channels", [
    (np.zeros((2, 5)), 250.0, 1), (np.zeros((1, 0)), 250.0, 1),
    (np.array([[np.nan]]), 250.0, 1), (np.zeros((1, 5)), -1.0, 1),
    (np.zeros((1, 5)), float("nan"), 1), (np.zeros((1, 5)), 250.0, 1.5),
])
def test_replay_rejects_invalid_recordings(tmp_path: Path, data: np.ndarray, rate: float, channels: int) -> None:
    path = tmp_path / "bad.npz"
    np.savez(path, data=data, sample_rate=rate, n_channels=channels)
    with pytest.raises(ValueError):
        ReplayEEGSource(path)
