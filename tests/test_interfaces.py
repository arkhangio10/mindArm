"""Input and output boundaries reject malformed or unsafe contracts."""

import pytest
from pydantic import ValidationError

from mindarm.interfaces import DecoderEvent, DetectedObject, MotionPlan, MotionPrimitive, OptionsResponse, Scene
from mindarm.planner.fake import safe_options


@pytest.mark.parametrize("payload", [
    {"kind": "SELECT"}, {"kind": "SELECT", "index": -1}, {"kind": "SELECT", "index": 4},
    {"kind": "SELECT", "index": True}, {"kind": "TOGGLE_ENGAGE", "index": 1},
    {"kind": "QUIT", "index": 0}, {"kind": "UNKNOWN"},
])
def test_invalid_event(payload: dict) -> None:
    with pytest.raises(ValidationError):
        DecoderEvent.model_validate(payload)


@pytest.mark.parametrize("payload", [
    {"action": "SET_SERVO", "angle": 90}, {"action": "HOME", "angles": [90]},
    {"action": "MOVE_ABOVE"}, {"action": "GRASP", "object_label": "cup"},
    {"action": "MOVE_TO_POSE", "pose_name": "unsafe"},
    {"action": "HOME", "pose_name": "home"},
])
def test_invalid_primitive(payload: dict) -> None:
    with pytest.raises(ValidationError):
        MotionPrimitive.model_validate(payload)


@pytest.mark.parametrize("value", [-0.1, 1.1, float("nan"), float("inf")])
def test_detection_bounds(value: float) -> None:
    with pytest.raises(ValidationError):
        DetectedObject(label="cup", x=value, y=0.5, confidence=0.9)


def test_exactly_four_unique_options_with_home() -> None:
    options = safe_options()
    assert len(OptionsResponse(options=options).options) == 4
    for invalid in (options[:3], options + [options[0]], [options[0]] * 4):
        with pytest.raises(ValidationError):
            OptionsResponse(options=invalid)


def test_empty_and_unknown_object_plan(scene: Scene) -> None:
    with pytest.raises(ValidationError):
        MotionPlan(primitives=[])
    plan = MotionPlan(primitives=[MotionPrimitive(action="MOVE_ABOVE", object_label="missing")])
    with pytest.raises(ValueError, match="Unknown scene object"):
        plan.validate_scene(scene)
