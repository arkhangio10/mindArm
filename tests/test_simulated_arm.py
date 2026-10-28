"""Independent joint/speed enforcement and whole-submission preflight."""

import yaml

from mindarm.arm.simulated import ArmSettings, SimulatedArm
from mindarm.config import Settings
from mindarm.interfaces import MotionPlan, MotionPrimitive, Scene
from mindarm.planner.fake import FakePlanner


def arm_settings(settings: Settings) -> ArmSettings:
    return ArmSettings.model_validate(yaml.safe_load(settings.joint_limits_path.read_text(encoding="utf-8")))


def test_motion_updates_six_joints(settings: Settings, scene: Scene) -> None:
    arm = SimulatedArm.from_file(settings.joint_limits_path)
    planner = FakePlanner()
    result = arm.execute(planner.plan_motion(planner.propose_options(scene)[1], scene))
    assert result.success and result.executed_primitives == 5
    assert len(arm.state()) == 6
    assert arm.state()["gripper"] == 0
    assert arm.state()["shoulder"] == 65
    snapshot = arm.state()
    snapshot["base"] = -999
    assert arm.state()["base"] == 90


def test_out_of_bounds_plan_has_no_partial_side_effects(settings: Settings) -> None:
    config = arm_settings(settings)
    config.poses["handover"]["base"] = 181.0
    arm = SimulatedArm(config)
    before = arm.state()
    result = arm.execute(MotionPlan(primitives=[MotionPrimitive(action="GRASP"),
                                               MotionPrimitive(action="MOVE_TO_POSE", pose_name="handover")]))
    assert not result.success and "angle" in result.error
    assert result.executed_primitives == 0 and arm.history == []
    assert arm.state() == before


def test_speed_violation_is_rejected(settings: Settings) -> None:
    config = arm_settings(settings)
    config.joints["gripper"].max_speed = 1.0
    arm = SimulatedArm(config)
    result = arm.execute(MotionPlan(primitives=[MotionPrimitive(action="GRASP")]))
    assert not result.success and "speed" in result.error
    assert arm.state()["gripper"] == 0


def test_joint_count_is_configurable() -> None:
    config = ArmSettings.model_validate({
        "joints": {"a": {"min_angle": 0.0, "max_angle": 180.0, "max_speed": 60.0}},
        "poses": {"home": {"a": 0.0}, "handover": {"a": 60.0}},
        "primitive_targets": {name: {"a": 20.0} for name in ["MOVE_ABOVE", "GRASP", "LIFT", "RELEASE"]},
        "primitive_duration_s": 2.0,
    })
    arm = SimulatedArm(config)
    assert arm.execute(MotionPlan(primitives=[MotionPrimitive(action="GRASP")])).success
    assert arm.state() == {"a": 20.0}
