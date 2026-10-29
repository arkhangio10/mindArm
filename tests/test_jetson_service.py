"""The Jetson arm service's safety logic against a fake I2C bus and scripted stdin."""

from __future__ import annotations

import importlib.util
import time
from typing import Any

import yaml

from conftest import ROOT

spec = importlib.util.spec_from_file_location("mindarm_arm_service", ROOT / "jetson" / "mindarm_arm.py")
assert spec is not None and spec.loader is not None
service: Any = importlib.util.module_from_spec(spec)
spec.loader.exec_module(service)

HOME = {"base": 1500, "shoulder": 1450, "elbow": 1500, "wrist_pitch": 1600, "wrist_roll": 1500, "gripper": 1500}


class FakeBus:
    """PCA9685 registers at power-on defaults, recording every write."""

    def __init__(self) -> None:
        self.registers = {service.MODE1: 0x11, service.PRESCALE: 0x1E}
        for channel in range(6):
            self.registers[service.LED0 + 4 * channel + 3] = service.FULL_OFF
        self.writes: list[tuple[int, Any]] = []

    def read_byte_data(self, address: int, register: int) -> int:
        return self.registers.get(register, 0)

    def write_byte_data(self, address: int, register: int, value: int) -> None:
        self.writes.append((register, value))
        self.registers[register] = value

    def write_i2c_block_data(self, address: int, register: int, data: list[int]) -> None:
        self.writes.append((register, list(data)))
        for offset, value in enumerate(data):
            self.registers[register + offset] = value


class FakePCA:
    """Channels already energized at given pulses."""

    def __init__(self, positions: dict[str, int | None]) -> None:
        self.positions = positions
        self.writes: list[tuple[int, int]] = []

    def read_us(self, channel: int) -> int | None:
        return self.positions[service.JOINTS[channel]]

    def write_us(self, channel: int, us: int) -> None:
        self.writes.append((channel, us))


class FakeLines:
    """Scripted stdin: timeouts until call number `line_at` returns a line."""

    def __init__(self, line_at: int | None = None, closed: bool = False) -> None:
        self.calls, self.line_at, self.closed = 0, line_at, closed

    def next(self, timeout: float | None) -> str | None:
        self.calls += 1
        if self.calls == self.line_at:
            return '{"cmd": "stop"}'
        time.sleep(timeout or 0)
        return None


def move(targets: dict[str, Any], duration: Any = 0.1, request_id: int = 1) -> dict[str, Any]:
    return {"cmd": "move", "id": request_id, "targets": targets, "duration": duration}


def test_power_on_chip_is_configured_with_every_channel_released() -> None:
    bus = FakeBus()
    arm = service.Arm(service.PCA9685(bus), FakeLines())
    assert bus.registers[service.PRESCALE] == service.PRESCALE_50HZ
    assert bus.registers[service.MODE1] == 0x21  # Awake, auto-increment, all-call kept.
    assert bus.writes[0] == (service.ALL_LED_OFF_H, service.FULL_OFF)
    assert arm.positions == dict.fromkeys(service.JOINTS)
    assert not arm.handle(move(HOME))["ok"]  # A released joint needs start first.


def test_running_chip_is_adopted_with_its_last_pulses() -> None:
    bus = FakeBus()
    bus.registers.update({service.MODE1: 0x21, service.PRESCALE: service.PRESCALE_50HZ})
    service.PCA9685(bus).write_us(1, 1450)
    bus.writes.clear()
    arm = service.Arm(service.PCA9685(bus), FakeLines())
    assert bus.writes == []
    assert arm.positions == {**dict.fromkeys(service.JOINTS), "shoulder": 1450}


def test_invalid_requests_are_rejected_without_motion() -> None:
    pca = FakePCA(dict(HOME))
    arm = service.Arm(pca, FakeLines())
    requests = [
        move({**HOME, "extra": 1500}), move({k: v for k, v in HOME.items() if k != "base"}),
        move({**HOME, "base": 1500.5}), move({**HOME, "base": True}), move({**HOME, "wrist_pitch": 1100}),
        move({**HOME, "gripper": 1950}, duration=0.1), move(HOME, duration=0), move(HOME, duration=11),
        move(HOME, duration=float("nan")), {"cmd": "jump", "id": 1}, [1, 2],
        {"cmd": "start", "id": 1, "targets": HOME, "duration": 1.0, "stagger": 3},
    ]
    for request in requests:
        reply = arm.handle(request)
        assert reply["ok"] is False and reply["stopped"] is False and reply["error"], request
    assert pca.writes == [] and arm.positions == HOME
    assert arm.handle({"cmd": "stop"}) is None


def test_start_energizes_released_channels_in_order_then_moves_ramp() -> None:
    lines = FakeLines()
    pca = FakePCA({**dict.fromkeys(service.JOINTS), "shoulder": 1400})
    arm = service.Arm(pca, lines)
    reply = arm.handle({"cmd": "start", "id": 1, "targets": HOME, "duration": 0.1, "stagger": 0})
    assert reply == {"id": 1, "ok": True, "stopped": False, "positions": HOME, "error": None}
    assert [channel for channel, _ in pca.writes[:5]] == [0, 2, 3, 4, 5]
    shoulder = [us for channel, us in pca.writes[5:] if channel == 1]
    assert shoulder == sorted(shoulder) and shoulder[-1] == 1450
    pca.writes.clear()
    reply = arm.handle(move({**HOME, "gripper": 1580}, duration=0.1, request_id=2))
    assert reply["ok"] and reply["positions"]["gripper"] == 1580
    gripper = [us for channel, us in pca.writes if channel == 5]
    assert len(gripper) > 1 and gripper == sorted(gripper) and gripper[-1] == 1580
    assert {channel for channel, _ in pca.writes} == {5}


def test_input_or_end_of_input_stops_a_ramp_where_it_is() -> None:
    for lines in (FakeLines(line_at=3), FakeLines(closed=True)):
        arm = service.Arm(FakePCA(dict(HOME)), lines)
        reply = arm.handle(move({**HOME, "gripper": 1950}, duration=0.5))
        assert reply["ok"] is False and reply["stopped"] is True
        assert 1500 <= reply["positions"]["gripper"] < 1950


def test_laptop_configuration_stays_inside_the_service_envelope() -> None:
    config = yaml.safe_load((ROOT / "config" / "jetson_arm.yaml").read_text(encoding="utf-8"))
    assert list(config["joints"]) == service.JOINTS
    for joint, limit in config["joints"].items():
        low, high = service.LIMITS_US[joint]
        assert low <= limit["min_angle"] < limit["max_angle"] <= high
        assert limit["max_speed"] <= service.MAX_SPEED_US_S
    assert config["poses"]["home"] == HOME
