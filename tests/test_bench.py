"""The live benchmark's bookkeeping, driven by a simulated client; no network."""

from typing import Any, cast
from unittest.mock import Mock

from openai import OpenAI
from openai.types import Model
from openai.types.chat import ChatCompletion

from mindarm.bench import model_prices, rejection_reason, report, run
from mindarm.config import PlannerOperation
from mindarm.interfaces import OptionsResponse, Scene, home_plan
from mindarm.planner.fake import FakePlanner
from mindarm.planner.nemotron import NemotronPlanner

PRICES = {"prompt": 1e-6, "input_cache_read": 1e-7, "completion": 2e-6}
LABEL_ON_GRASP = '{"primitives":[{"action":"GRASP","object_label":"cup"},{"action":"LIFT","object_label":"cup"}]}'


def completion(raw: str) -> ChatCompletion:
    return ChatCompletion.model_validate({
        "id": "offline", "object": "chat.completion", "created": 0, "model": "test",
        "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": raw}}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150,
                  "prompt_tokens_details": {"cached_tokens": 60},
                  "completion_tokens_details": {"reasoning_tokens": 30}},
    })


def test_run_counts_first_try_repairs_reasons_tokens_and_cost(scene: Scene) -> None:
    goals = FakePlanner().propose_options(scene)  # Pick up cup, bottle, sports ball, then Home
    pickup = FakePlanner().plan_motion(goals[0], scene).model_dump_json()
    proposals = [OptionsResponse(options=goals).model_dump_json()] * 2
    plans = [LABEL_ON_GRASP, pickup, home_plan().model_dump_json()]
    client = Mock()
    client.chat.completions.create.side_effect = [completion(raw) for raw in proposals + plans]
    records: list[dict[str, Any]] = []
    planner = NemotronPlanner(cast(OpenAI, client), record=records.append)

    propose, plan = run(planner, scene, 2, records, PRICES)

    assert (propose["first_try"], propose["repaired"], propose["fallback"], propose["calls"]) == (2, 0, 0, 2)
    assert (plan["first_try"], plan["repaired"], plan["fallback"], plan["calls"]) == (1, 1, 0, 3)
    assert (propose["expected"], plan["expected"]) == (2, 1)  # [HOME] for "Pick up bottle" is not its handover
    assert plan["outputs"][1] == home_plan().model_dump(mode="json") and len(propose["outputs"][0]) == 4
    assert plan["rejections"] == {"primitives.[i]: Value error, Only MOVE_ABOVE requires object_label": 1}
    assert plan["tokens"] == {"prompt": 300, "cached": 180, "completion": 150, "reasoning": 90}
    assert abs(plan["cost_usd"] - (120 * 1e-6 + 180 * 1e-7 + 150 * 2e-6)) < 1e-12
    assert plan["p50_s"] <= plan["p95_s"] <= plan["max_s"]
    operations = {"propose": PlannerOperation(reasoning=False), "plan": PlannerOperation(reasoning=True)}
    assert "Only MOVE_ABOVE requires object_label" in report("test", operations, [propose, plan])


def test_rejection_reason_accepts_plain_errors() -> None:
    assert rejection_reason("Unknown scene object: lamp") == "Unknown scene object: lamp"


def test_prices_come_from_the_verbose_model_list() -> None:
    client = Mock()
    client.models.list.return_value = [Model.model_validate({
        "id": "m", "object": "model", "created": 0, "owned_by": "x",
        "pricing": {"prompt": "0.0000003", "input_cache_read": "0.00000006", "completion": "0.0000009"}})]
    assert model_prices(cast(OpenAI, client), "m") == {"prompt": 3e-7, "input_cache_read": 6e-8, "completion": 9e-7}
    assert model_prices(cast(OpenAI, client), "other") is None
