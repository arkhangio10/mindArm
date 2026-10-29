"""The real-arm adapter against an in-memory stand-in for the Jetson service; no SSH."""

from __future__ import annotations

import time
from queue import Empty, Queue
from threading import Thread
from typing import Any

import pytest
import yaml

from conftest import ScriptedDecoder
from mindarm.arm.jetson import STAGGER_S, JetsonArm
from mindarm.arm.simulated import ArmSettings
from mindarm.config import Settings
from mindarm.factory import create_components
from mindarm.interfaces import ArmController, ArmTransport, MotionPlan, MotionPrimitive, Scene
from mindarm.planner.fake import FakePlanner
from mindarm.session_log import SessionLog

HOME = {"base": 1500, "shoulder": 1450, "elbow": 1500, "wrist_pitch": 1600, "wrist_roll": 1500, "gripper": 1500}


class FakeService:
    """Answer like the Jetson service: instant moves, optional rejection, or a move held until stop."""

    def __init__(self, positions: dict[str, int | None]) -> None:
        self.positions = dict(positions)
        self.sent: list[dict[str, Any]] = []
        self.replies: Queue[dict[str, Any]] = Queue()
        self.reject: str | None = None
        self.hold = False
        self.pending: dict[str, Any] | None = None
        self.link_lost = False
        self.closed = False
        self._reply(0, True)

    def _reply(self, request_id: int, ok: bool, error: str | None = None, stopped: bool = False) -> None:
        self.replies.put({"id": request_id, "ok": ok, "stopped": stopped,
                          "positions": dict(self.positions), "error": error})

    def send(self, message: dict[str, Any]) -> None:
        self.sent.append(message)
        if message["cmd"] == "stop":
            if self.pending is not None:
                self._reply(self.pending["id"], False, "Stopped before the target", stopped=True)
                self.pending = None
        elif self.hold:
            self.pending = message
        elif self.reject is not None:
            self._reply(message["id"], False, self.reject)
        else:
            self.positions.update(message["targets"])
            self._reply(message["id"], True)

    def receive(self, timeout: float) -> dict[str, Any] | None:
        if self.link_lost:
            raise ConnectionError("Jetson arm link closed")
        try:
            return self.replies.get(timeout=timeout)
        except Empty:
            return None

    def close(self) -> None:
        self.closed = True


def arm_settings(settings: Settings) -> ArmSettings:
    return ArmSettings.model_validate(yaml.safe_load(settings.jetson_arm_path.read_text(encoding="utf-8")))


def started_arm(settings: Settings, positions: dict[str, int | None] | None = None) -> tuple[JetsonArm, FakeService]:
    service = FakeService(positions or dict.fromkeys(HOME))
    return JetsonArm(arm_settings(settings), service), service


def plan(*actions: str) -> MotionPlan:
    return MotionPlan(primitives=[MotionPrimitive(action=action) for action in actions])


def test_startup_energizes_released_joints_at_home(settings: Settings) -> None:
    arm, service = started_arm(settings)
    assert isinstance(arm, ArmController) and isinstance(service, ArmTransport)
    assert service.sent == [{"cmd": "start", "id": 1, "targets": HOME, "duration": 1.5, "stagger": STAGGER_S}]
    assert arm.state() == {joint: float(pulse) for joint, pulse in HOME.items()}


def test_startup_ramps_energized_joints_within_speed_limits(settings: Settings) -> None:
    _, service = started_arm(settings, {**dict.fromkeys(HOME), "shoulder": 1000})
    assert service.sent[0]["duration"] == pytest.approx(450 / 300)
    _, service = started_arm(settings, {**dict.fromkeys(HOME), "shoulder": 2000})
    assert service.sent[0]["duration"] == pytest.approx(550 / 300)


def test_failed_start_fails_setup(settings: Settings) -> None:
    service = FakeService(dict.fromkeys(HOME))
    service.reject = "gripper: 1500 us outside [1130, 1950]"
    with pytest.raises(RuntimeError, match="start failed"):
        JetsonArm(arm_settings(settings), service)


def test_plan_moves_real_arm_with_checked_durations(settings: Settings, scene: Scene) -> None:
    arm, service = started_arm(settings)
    planner = FakePlanner()
    result = arm.execute(planner.plan_motion(planner.propose_options(scene)[1], scene))
    assert result.success and result.executed_primitives == 5
    config, expected, current = arm_settings(settings), [], dict(HOME)
    for name in ["MOVE_ABOVE", "GRASP", "LIFT", "handover", "RELEASE"]:  # Whatever poses are taught.
        current = {**current, **(config.poses.get(name) or config.primitive_targets[name])}
        expected.append({joint: round(value) for joint, value in current.items()})
    moves = service.sent[1:]
    assert [move["cmd"] for move in moves] == ["move"] * 5
    assert [move["targets"] for move in moves] == expected
    assert all(move["duration"] == config.primitive_duration_s for move in moves)
    assert result.joint_angles == arm.state() == expected[-1]


def test_invalid_plans_send_nothing(settings: Settings) -> None:
    config = arm_settings(settings)
    config.poses["handover"]["base"] = 2100.0
    service = FakeService(dict.fromkeys(HOME))
    arm = JetsonArm(config, service)
    result = arm.execute(MotionPlan(primitives=[MotionPrimitive(action="GRASP"),
                                               MotionPrimitive(action="MOVE_TO_POSE", pose_name="handover")]))
    assert not result.success and "outside" in result.error and result.executed_primitives == 0
    config = arm_settings(settings)
    config.joints["gripper"].max_speed = 100.0
    fast = FakeService(dict.fromkeys(HOME))
    slow_gripper = JetsonArm(config, fast)
    result = slow_gripper.execute(plan("GRASP"))
    assert not result.success and "speed" in result.error
    assert len(service.sent) == len(fast.sent) == 1


def test_service_rejection_and_link_loss_are_reported(settings: Settings) -> None:
    arm, service = started_arm(settings)
    service.reject = "gripper: 1130 us outside [1200, 1950]"
    result = arm.execute(plan("RELEASE", "GRASP"))
    assert not result.success and result.error == service.reject and result.executed_primitives == 0
    service.reject = None
    service.link_lost = True
    result = arm.execute(plan("GRASP"))
    assert not result.success and "link" in result.error
    assert arm.state()["gripper"] == 1500


def test_cancel_stops_in_flight_move_then_home_runs(settings: Settings) -> None:
    arm, service = started_arm(settings)
    service.hold = True
    results = []
    worker = Thread(target=lambda: results.append(arm.execute(plan("RELEASE", "GRASP"))))
    worker.start()
    while service.pending is None:
        time.sleep(0.001)
    arm.cancel()
    worker.join(timeout=2)
    assert results and not results[0].success and results[0].executed_primitives == 0
    assert "Stopped" in results[0].error
    service.hold = False
    assert arm.execute(plan("HOME")).success
    assert [message["cmd"] for message in service.sent] == ["start", "move", "stop", "move"]


def test_factory_selects_jetson_arm(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    services: list[FakeService] = []

    def transport(command: list[str]) -> FakeService:
        assert command == settings.jetson_command
        services.append(FakeService(dict(HOME)))
        return services[-1]

    class ClosableDecoder(ScriptedDecoder):
        """Stand in for the keyboard, which Components.close() also releases."""

        def close(self) -> None:
            pass

    monkeypatch.setattr("mindarm.factory.KeyboardDecoder", ClosableDecoder)
    monkeypatch.setattr("mindarm.factory.SshTransport", transport)
    assert settings.arm == "simulated"
    settings.arm = "jetson"
    components = create_components(settings, SessionLog(settings.logs_dir))
    assert isinstance(components.arm, JetsonArm)
    components.close()
    assert services[0].closed
