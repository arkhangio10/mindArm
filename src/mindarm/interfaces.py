"""Shared module contracts and the complete allowed planner vocabulary."""

from __future__ import annotations

from typing import Any, Literal, Protocol, runtime_checkable

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator


class Contract(BaseModel):
    """Reject unknown fields, non-finite numbers, and implicit type coercion."""

    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class DecoderEvent(Contract):
    """Zero-based selection, engagement toggle, or explicit shutdown."""

    kind: Literal["SELECT", "TOGGLE_ENGAGE", "QUIT"]
    index: int | None = Field(default=None, ge=0, le=3)

    @model_validator(mode="after")
    def check_index(self) -> DecoderEvent:
        if (self.kind == "SELECT") != (self.index is not None):
            raise ValueError("Only SELECT requires an index")
        return self


class DetectedObject(Contract):
    """A labeled detection in normalized image coordinates."""

    label: str = Field(min_length=1, max_length=100)
    x: float = Field(ge=0, le=1)
    y: float = Field(ge=0, le=1)
    confidence: float = Field(ge=0, le=1)


class Scene(Contract):
    """Text-serializable detections; no images are sent to the planner."""

    objects: list[DetectedObject]


class Option(Contract):
    """A displayed intent, never a servo command."""

    label: str = Field(min_length=1, max_length=100)
    description: str = Field(min_length=1, max_length=500)


class OptionsResponse(Contract):
    """The planner's JSON envelope, including exactly one Home choice."""

    options: list[Option] = Field(min_length=4, max_length=4)

    @model_validator(mode="after")
    def check_home(self) -> OptionsResponse:
        labels = [option.label for option in self.options]
        if labels.count("Home") != 1:
            raise ValueError('Exactly one option must be named "Home"')
        if len({label.strip().casefold() for label in labels}) != 4:
            raise ValueError("Option labels must be distinct")
        return self


class MotionPrimitive(Contract):
    """Closed motion vocabulary with action-specific argument validation."""

    action: Literal["HOME", "MOVE_ABOVE", "GRASP", "LIFT", "MOVE_TO_POSE", "RELEASE"]
    object_label: str | None = Field(default=None, min_length=1, max_length=100)
    pose_name: Literal["home", "handover"] | None = None

    @model_validator(mode="after")
    def check_arguments(self) -> MotionPrimitive:
        if (self.action == "MOVE_ABOVE") != (self.object_label is not None):
            raise ValueError("Only MOVE_ABOVE requires object_label")
        if (self.action == "MOVE_TO_POSE") != (self.pose_name is not None):
            raise ValueError("Only MOVE_TO_POSE requires pose_name")
        return self


class MotionPlan(Contract):
    """A bounded, nonempty sequence of validated motion primitives."""

    primitives: list[MotionPrimitive] = Field(min_length=1, max_length=32)

    def validate_scene(self, scene: Scene) -> MotionPlan:
        """Reject references to objects absent from the current scene."""
        labels = {obj.label for obj in scene.objects}
        for primitive in self.primitives:
            if primitive.action == "MOVE_ABOVE" and primitive.object_label not in labels:
                raise ValueError(f"Unknown scene object: {primitive.object_label}")
        return self


class ExecutionResult(Contract):
    """Execution outcome, including partial progress and final joint angles."""

    success: bool
    executed_primitives: int = Field(ge=0)
    joint_angles: dict[str, float]
    error: str | None = None


def home_plan() -> MotionPlan:
    """Return a fresh safe fallback plan."""
    return MotionPlan(primitives=[MotionPrimitive(action="HOME")])


@runtime_checkable
class EEGSource(Protocol):
    """Provide finite samples shaped (channels, samples)."""

    @property
    def sample_rate(self) -> float: ...

    @property
    def n_channels(self) -> int: ...

    def read(self, n_samples: int) -> np.ndarray: ...


@runtime_checkable
class Decoder(Protocol):
    """Provide nonblocking intent events independently of acquisition."""

    def poll(self) -> DecoderEvent | None: ...


@runtime_checkable
class Perception(Protocol):
    """Describe the current scene as normalized detections."""

    def detect(self) -> Scene: ...


@runtime_checkable
class Planner(Protocol):
    """Propose four intents and map one intent to allowed primitives."""

    def propose_options(self, scene: Scene) -> list[Option]: ...

    def plan_motion(self, option: Option, scene: Scene) -> MotionPlan: ...


@runtime_checkable
class ArmController(Protocol):
    """Enforce local joint/speed limits in the arm's unit; execute() may block, cancel() must not."""

    def execute(self, plan: MotionPlan) -> ExecutionResult: ...

    def state(self) -> dict[str, float]: ...

    def cancel(self) -> None: ...


@runtime_checkable
class ArmTransport(Protocol):
    """Exchange JSON-line messages with an arm service; receive() returns None on timeout."""

    def send(self, message: dict[str, Any]) -> None: ...

    def receive(self, timeout: float) -> dict[str, Any] | None: ...

    def close(self) -> None: ...
