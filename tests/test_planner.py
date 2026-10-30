"""Offline planner tests exercising actual SDK response models without HTTP."""

from typing import Any, cast
from unittest.mock import Mock

import pytest
from openai import APIConnectionError, AuthenticationError, InternalServerError, OpenAI
from openai.types.chat import ChatCompletion
import httpx

from mindarm.config import PlannerOperation
from mindarm.interfaces import MotionPlan, Option, OptionsResponse, Scene, home_plan
from mindarm.planner.fake import FakePlanner, safe_options
from mindarm.planner.nemotron import MAX_CALLS, NemotronPlanner, PlannerUnavailable
from mindarm.planner.prompts import motion_schema

REQUEST = httpx.Request("POST", "https://example.invalid")


def completion(raw: str, finish_reason: str = "stop") -> ChatCompletion:
    return ChatCompletion.model_validate({
        "id": "offline", "object": "chat.completion", "created": 0, "model": "test",
        "choices": [{"index": 0, "finish_reason": finish_reason,
                     "message": {"role": "assistant", "content": raw}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
    })


def status_error(kind: type[Exception], code: int) -> Exception:
    return kind("offline", response=httpx.Response(code, request=REQUEST), body=None)


def make_planner(outputs: list[str | Exception], finish_reason: str = "stop",
                 **operations: PlannerOperation) -> tuple[NemotronPlanner, Mock, list[dict[str, Any]]]:
    client = Mock()
    client.chat.completions.create.side_effect = [
        raw if isinstance(raw, Exception) else completion(raw, finish_reason) for raw in outputs]
    records: list[dict[str, Any]] = []
    return NemotronPlanner(cast(OpenAI, client), record=records.append, **operations), client, records


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


def test_network_failure_is_retried_once_then_propagated(scene: Scene) -> None:
    planner, client, records = make_planner([APIConnectionError(request=REQUEST)] * 2)
    with pytest.raises(PlannerUnavailable, match="APIConnectionError"):
        planner.propose_options(scene)
    assert client.chat.completions.create.call_count == 2
    assert [record["status"] for record in records] == ["transient_error", "api_error"]
    assert [record["call"] for record in records] == [1, 2]
    assert all(record["usage"] is None and record["max_calls"] == MAX_CALLS for record in records)


def test_transient_server_error_retry_is_not_the_repair(scene: Scene) -> None:
    planner, client, records = make_planner([status_error(InternalServerError, 500), home_plan().model_dump_json()])
    assert planner.plan_motion(safe_options()[3], scene) == home_plan()
    retry = client.chat.completions.create.call_args_list[1].kwargs
    assert all("Validation failed" not in str(message["content"]) for message in retry["messages"])
    assert [(record["attempt"], record["status"]) for record in records] == [(1, "transient_error"), (1, "ok")]


def test_calls_per_operation_are_bounded(scene: Scene) -> None:
    planner, client, records = make_planner(["{}", status_error(InternalServerError, 503), "{}"])
    assert planner.propose_options(scene) == safe_options()
    assert client.chat.completions.create.call_count == MAX_CALLS == 3
    assert [(record["call"], record["attempt"]) for record in records[:3]] == [(1, 1), (2, 2), (3, 2)]
    assert records[-1]["type"] == "planner_fallback"


def test_client_errors_are_not_retried(scene: Scene) -> None:
    planner, client, records = make_planner([status_error(AuthenticationError, 401)])
    with pytest.raises(PlannerUnavailable, match="AuthenticationError"):
        planner.propose_options(scene)
    assert client.chat.completions.create.call_count == 1
    assert records[0]["status"] == "api_error"


def test_strict_schema_and_reasoning_per_operation(scene: Scene) -> None:
    valid = OptionsResponse(options=safe_options()).model_dump_json()
    planner, client, records = make_planner([valid, home_plan().model_dump_json()],
                                            propose=PlannerOperation(reasoning=False),
                                            plan=PlannerOperation(reasoning=True))
    planner.propose_options(scene)
    planner.plan_motion(safe_options()[3], scene)
    propose, plan = (call.kwargs for call in client.chat.completions.create.call_args_list)
    assert propose["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}
    assert plan["extra_body"] == {"chat_template_kwargs": {"enable_thinking": True}}
    assert plan["response_format"]["type"] == "json_schema"
    assert plan["response_format"]["json_schema"]["strict"] is True
    assert plan["response_format"]["json_schema"]["schema"] == motion_schema(scene)
    assert [(record["reasoning"], record["strict_schema"]) for record in records] == [(False, True), (True, True)]


def test_json_object_mode_remains_selectable(scene: Scene) -> None:
    planner, client, _ = make_planner([home_plan().model_dump_json()],
                                      plan=PlannerOperation(reasoning=False, strict_schema=False))
    planner.plan_motion(safe_options()[3], scene)
    assert client.chat.completions.create.call_args.kwargs["response_format"] == {"type": "json_object"}


def test_motion_schema_limits_arguments_per_action(scene: Scene) -> None:
    steps = {tuple(step["properties"]["action"]["enum"]): step
             for step in motion_schema(scene)["properties"]["primitives"]["items"]["anyOf"]}
    assert set(steps[("HOME", "GRASP", "LIFT", "RELEASE")]["properties"]) == {"action"}
    assert steps[("MOVE_TO_POSE",)]["properties"]["pose_name"]["enum"] == ["home", "handover"]
    assert steps[("MOVE_ABOVE",)]["properties"]["object_label"]["enum"] == ["bottle", "cup", "sports ball"]
    assert all(step["additionalProperties"] is False for step in steps.values())
    empty = motion_schema(Scene(objects=[]))["properties"]["primitives"]["items"]["anyOf"]
    assert all("MOVE_ABOVE" not in step["properties"]["action"]["enum"] for step in empty)


def test_missing_api_key_is_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NEBIUS_API_KEY", raising=False)
    with pytest.raises(ValueError, match="NEBIUS_API_KEY"):
        NemotronPlanner()
