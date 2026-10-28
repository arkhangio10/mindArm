"""Shared deterministic fixtures; prevent accidental network access offline."""

from __future__ import annotations

import socket
from collections import deque
from pathlib import Path
from typing import Any

import pytest
from dotenv import load_dotenv

from mindarm.arm.simulated import SimulatedArm
from mindarm.config import Settings, load_settings
from mindarm.eeg.fake import FakeEEGSource
from mindarm.interfaces import DecoderEvent, Scene
from mindarm.orchestrator import Orchestrator
from mindarm.perception.fake import FakePerception
from mindarm.planner.fake import FakePlanner
from mindarm.session_log import SessionLog

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env", override=False)


@pytest.fixture(autouse=True)
def offline_by_default(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    if request.node.get_closest_marker("integration") is None:
        def refuse(*args: Any, **kwargs: Any) -> None:
            raise AssertionError("Network access is forbidden in offline tests")
        monkeypatch.setattr(socket.socket, "connect", refuse)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    value = load_settings(ROOT / "config" / "settings.yaml")
    value.planner = "fake"
    value.logs_dir = tmp_path / "logs"
    value.recordings_dir = tmp_path / "recordings"
    return value


@pytest.fixture
def scene(settings: Settings) -> Scene:
    return FakePerception(settings.scene_path).detect()


class ScriptedDecoder:
    """Allow tests to supply one intentional event at a time."""

    def __init__(self) -> None:
        self.events: deque[DecoderEvent] = deque()

    def poll(self) -> DecoderEvent | None:
        return self.events.popleft() if self.events else None

    def send(self, kind: str, index: int | None = None) -> None:
        self.events.append(DecoderEvent.model_validate({"kind": kind, "index": index}))


@pytest.fixture
def app(settings: Settings) -> Orchestrator:
    return Orchestrator(
        FakeEEGSource(seed=1), ScriptedDecoder(), FakePerception(settings.scene_path),
        FakePlanner(), SimulatedArm.from_file(settings.joint_limits_path), settings,
        SessionLog(settings.logs_dir), emit=lambda _: None,
    )
