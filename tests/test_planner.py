"""Offline planner tests exercising actual SDK response models without HTTP."""

from typing import Any, cast
from unittest.mock import Mock

import pytest
from openai import APIConnectionError, OpenAI
from openai.types.chat import ChatCompletion
import httpx

from mindarm.interfaces import MotionPlan, Option, OptionsResponse, Scene, home_plan
from mindarm.planner.fake import FakePlanner, safe_options
from mindarm.planner.nemotron import NemotronPlanner, PlannerUnavailable


def make_planner(outputs: list[str], finish_reason: str = "stop") -> tuple[NemotronPlanner, Mock, list[dict[str, Any]]]:
    client = Mock()
    client.chat.completions.create.side_effect = [ChatCompletion.model_validate({
        "id": "offline", "object": "chat.completion", "created": 0, "model": "test",
        "choices": [{"index": 0, "finish_reason": finish_reason,
                     "message": {"role": "assistant", "content": raw}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
    }) for raw in outputs]
    records: list[dict[str, Any]] = []
    return NemotronPlanner(cast(OpenAI, client), record=records.append), client, records


def test_fake_second_option_and_home(scene: Scene) -> None:
    planner = FakePlanner()
    options = planner.propose_options(scene)
    OptionsResponse(options=options)
    plan = planner.plan_motion(options[1], scene)
    assert plan.primitives[0].object_label == "bottle"
    assert plan.primitives[-1].action == "RELEASE"
    assert planner.plan_motion(options[3], scene) == home_plan()
    assert len(planner.propose_options(Scene(objects=[]))) == 4


def test_one_retry_then_option_fallback(scene: Scene) -> None:
    planner, client, records = make_planner(["not json", "{}"])
    assert planner.propose_options(scene) == safe_options()
    assert client.chat.completions.create.call_count == 2
    retry = client.chat.completions.create.call_args_list[1].kwargs
    assert "Validation failed" in retry["messages"][-1]["content"]
    calls = [record for record in records if record["type"] == "planner_call"]
    assert len(calls) == 2
    assert all(record["latency_ms"] >= 0 and record["usage"]["total_tokens"] == 30 for record in calls)
    assert records[-1]["type"] == "planner_fallback"


def test_repair_succeeds_without_fallback(scene: Scene) -> None:
    valid = OptionsResponse(options=safe_options()).model_dump_json()
    planner, client, records = make_planner(["{}", valid])
    assert planner.propose_options(scene) == safe_options()
    assert client.chat.completions.create.call_count == 2
    assert records[-1]["status"] == "ok"
    assert all(record["type"] != "planner_fallback" for record in records)


@pytest.mark.parametrize("raw", [
    '{"primitives":[{"action":"SET_SERVO","angle":200}]}',
    '{"primitives":[{"action":"HOME","speed":99}]}',
    '{"primitives":[{"action":"MOVE_ABOVE","object_label":"missing"}]}',
    '{"primitives":[{"action":"MOVE_TO_POSE","pose_name":"unknown"}]}',
    '{"primitives":[]}', "invalid json",
])
def test_unsafe_plan_retries_then_home(scene: Scene, raw: str) -> None:
    planner, client, _ = make_planner([raw, raw])
    assert planner.plan_motion(Option(label="Pick up cup", description="Pick up cup"), scene) == home_plan()
    assert client.chat.completions.create.call_count == 2


def test_home_intent_cannot_be_changed_by_model(scene: Scene) -> None:
    raw = '{"primitives":[{"action":"GRASP"}]}'
    planner, client, _ = make_planner([raw, raw])
    assert planner.plan_motion(safe_options()[3], scene) == home_plan()
    assert client.chat.completions.create.call_count == 2


def test_valid_json_with_truncated_finish_is_rejected(scene: Scene) -> None:
    raw = home_plan().model_dump_json()
    planner, client, records = make_planner([raw, raw], finish_reason="length")
    assert planner.plan_motion(safe_options()[3], scene) == home_plan()
    assert client.chat.completions.create.call_count == 2
    assert records[0]["status"] == "invalid_output"


def test_network_failure_is_logged_and_propagated(scene: Scene) -> None:
    planner, client, records = make_planner([])
    client.chat.completions.create.side_effect = APIConnectionError(request=httpx.Request("POST", "https://example.invalid"))
    with pytest.raises(PlannerUnavailable, match="APIConnectionError"):
        planner.propose_options(scene)
    assert client.chat.completions.create.call_count == 1
    assert records[0]["status"] == "api_error"
    assert records[0]["usage"] is None


def test_missing_api_key_is_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NEBIUS_API_KEY", raising=False)
    with pytest.raises(ValueError, match="NEBIUS_API_KEY"):
        NemotronPlanner()
