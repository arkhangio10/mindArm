"""Pose teaching: frame splitting, save rules, YAML rewriting, and operator moves; no SSH or sockets."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import yaml

from conftest import ROOT
from mindarm.arm.jetson import JetsonArm
from mindarm.arm.simulated import ArmSettings
from mindarm.teach import MoveRequest, TeachApp, plan_check, split_frames, taught, write_targets
from test_jetson_arm import HOME, FakeService


def copied_config(tmp_path: Path) -> Path:
    path = tmp_path / "jetson_arm.yaml"
    shutil.copy(ROOT / "config" / "jetson_arm.yaml", path)
    return path


def load(path: Path) -> ArmSettings:
    return ArmSettings.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def test_split_frames_keeps_only_complete_jpegs() -> None:
    frames, rest = split_frames(b"junk\xff\xd8one\xff\xd9\xff\xd8two\xff\xd9\xff\xd8thr")
    assert frames == [b"\xff\xd8one\xff\xd9", b"\xff\xd8two\xff\xd9"]
    assert rest == b"\xff\xd8thr"
    assert split_frames(b"no frame yet") == ([], b"")


def test_saving_follows_per_target_joint_rules_and_keeps_comments(tmp_path: Path) -> None:
    path = copied_config(tmp_path)
    settings = load(path)
    above = {**HOME, "shoulder": 1600, "elbow": 1400, "gripper": 1950}
    settings = taught(settings, "MOVE_ABOVE", above)
    settings = taught(settings, "GRASP", {**above, "gripper": 1200})
    settings = taught(settings, "LIFT", {**above, "shoulder": 1550, "gripper": 1200})
    settings = taught(settings, "handover", {**HOME, "base": 1700, "gripper": 1950})
    settings = taught(settings, "RELEASE", {**HOME, "base": 1700, "gripper": 1900})
    write_targets(path, settings)
    saved = load(path)
    arm_joints = {joint: value for joint, value in above.items() if joint != "gripper"}
    assert saved.primitive_targets == {
        "MOVE_ABOVE": above, "GRASP": {"gripper": 1200}, "LIFT": {**arm_joints, "shoulder": 1550},
        "RELEASE": {"gripper": 1900},
    }
    assert saved.poses == {"home": HOME, "handover": {**HOME, "base": 1700, "gripper": 1200}}
    assert "# Real arm: PWM pulse widths" in path.read_text(encoding="utf-8")
    assert plan_check(saved).startswith("Plan check OK")
    too_fast = taught(saved, "MOVE_ABOVE", {**above, "shoulder": 2000})
    assert "shoulder: speed" in plan_check(too_fast)


def test_operator_moves_respect_limits_and_visits_use_saved_targets(tmp_path: Path) -> None:
    path = copied_config(tmp_path)
    service = FakeService(dict(HOME))
    arm = JetsonArm(load(path), service)
    app = TeachApp(arm, load(path), path, None)
    app.move(MoveRequest(targets={"shoulder": 1600}))
    assert service.sent[-1]["targets"]["shoulder"] == 1600
    assert service.sent[-1]["duration"] == pytest.approx(150 / 300)
    sent = len(service.sent)
    app.move(MoveRequest(targets={"wrist_pitch": 1100}))
    assert "outside" in app.message
    app.move(MoveRequest(targets={"elbo": 1500}))
    assert "Unknown joints" in app.message and len(service.sent) == sent
    app.save("MOVE_ABOVE")
    assert app.message.startswith("Saved MOVE_ABOVE")
    assert load(path).primitive_targets["MOVE_ABOVE"]["shoulder"] == 1600
    app.visit("HOME")
    assert arm.state()["shoulder"] == 1450
    app.visit("MOVE_ABOVE")
    assert app.state()["joints"]["shoulder"] == {"min": 1000.0, "max": 2000.0, "value": 1600.0}
