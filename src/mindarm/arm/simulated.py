"""A deterministic joint simulation; no inverse kinematics or hardware control."""

from __future__ import annotations

import logging
from pathlib import Path

import yaml
from pydantic import Field, model_validator

from mindarm.interfaces import Contract, ExecutionResult, MotionPlan, MotionPrimitive

LOGGER = logging.getLogger(__name__)


class JointLimit(Contract):
    """Angle bounds in degrees and positive maximum speed in degrees/second."""

    min_angle: float
    max_angle: float
    max_speed: float = Field(gt=0)

    @model_validator(mode="after")
    def check_bounds(self) -> JointLimit:
        if self.min_angle >= self.max_angle:
            raise ValueError("min_angle must be below max_angle")
        return self


class ArmSettings(Contract):
    """Local, configurable joint set and deterministic primitive targets."""

    joints: dict[str, JointLimit] = Field(min_length=1)
    poses: dict[str, dict[str, float]]
    primitive_targets: dict[str, dict[str, float]]
    primitive_duration_s: float = Field(gt=0)

    @model_validator(mode="after")
    def check_targets(self) -> ArmSettings:
        if set(self.poses) != {"home", "handover"}:
            raise ValueError("Exactly home and handover poses are required")
        if set(self.primitive_targets) != {"MOVE_ABOVE", "GRASP", "LIFT", "RELEASE"}:
            raise ValueError("Provide targets for MOVE_ABOVE, GRASP, LIFT, RELEASE")
        for pose in self.poses.values():
            if set(pose) != set(self.joints):
                raise ValueError("Every named pose must define every configured joint")
        for target in self.primitive_targets.values():
            if not target or not set(target) <= set(self.joints):
                raise ValueError("Primitive targets must name configured joints")
        return self


class SimulatedArm:
    """Preflight each submitted plan, then update virtual angles and log primitives."""

    def __init__(self, settings: ArmSettings) -> None:
        self._settings = settings.model_copy(deep=True)
        self._angles = dict(settings.poses["home"])
        self.history: list[MotionPrimitive] = []
        self._check_move(self._angles, self._angles, settings.primitive_duration_s)

    @classmethod
    def from_file(cls, path: Path) -> SimulatedArm:
        """Load joint limits and local motion targets from YAML."""
        return cls(ArmSettings.model_validate(yaml.safe_load(path.read_text(encoding="utf-8"))))

    def state(self) -> dict[str, float]:
        return self._angles.copy()

    def _check_move(self, start: dict[str, float], target: dict[str, float], duration: float) -> None:
        for joint, angle in target.items():
            limit = self._settings.joints[joint]
            if not limit.min_angle <= angle <= limit.max_angle:
                raise ValueError(f"{joint}: angle {angle} outside [{limit.min_angle}, {limit.max_angle}]")
            speed = abs(angle - start[joint]) / duration
            if speed > limit.max_speed + 1e-9:
                raise ValueError(f"{joint}: speed {speed:.3f} exceeds {limit.max_speed}")

    def _target(self, current: dict[str, float], primitive: MotionPrimitive) -> tuple[dict[str, float], float]:
        target = current.copy()
        duration = self._settings.primitive_duration_s
        if primitive.action in ("HOME", "MOVE_TO_POSE"):
            pose = "home" if primitive.action == "HOME" else primitive.pose_name
            target.update(self._settings.poses[str(pose)])
        else:
            target.update(self._settings.primitive_targets[primitive.action])
        if primitive.action == "HOME":
            # Recovery may take longer, but it never bypasses speed limits.
            duration = max(duration, max(
                abs(target[name] - current[name]) / limit.max_speed
                for name, limit in self._settings.joints.items()
            ))
        return target, duration

    def execute(self, plan: MotionPlan) -> ExecutionResult:
        staged: list[tuple[MotionPrimitive, dict[str, float], float]] = []
        current = self.state()
        try:
            # Revalidate even model_construct()/post-validation mutations at this boundary.
            plan = MotionPlan.model_validate(plan.model_dump())
            for primitive in plan.primitives:
                target, duration = self._target(current, primitive)
                self._check_move(current, target, duration)
                staged.append((primitive, target, duration))
                current = target
        except ValueError as error:
            return ExecutionResult(success=False, executed_primitives=0, joint_angles=self.state(), error=str(error))
        for primitive, target, duration in staged:
            self._angles = target
            self.history.append(primitive.model_copy(deep=True))
            LOGGER.info("%s %s -> %s (virtual %.2fs)", primitive.action,
                        primitive.object_label or primitive.pose_name or "", target, duration)
        return ExecutionResult(success=True, executed_primitives=len(staged), joint_angles=self.state())
