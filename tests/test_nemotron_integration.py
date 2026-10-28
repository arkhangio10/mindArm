"""Explicit opt-in live test; fallback cannot masquerade as API success."""

import os
from typing import Any

import pytest

from mindarm.arm.simulated import SimulatedArm
from mindarm.config import Settings
from mindarm.interfaces import OptionsResponse, Scene
from mindarm.planner.nemotron import NemotronPlanner


@pytest.mark.integration
@pytest.mark.skipif(not os.getenv("NEBIUS_API_KEY", "").strip(), reason="NEBIUS_API_KEY is not configured")
def test_live_nemotron_proposal_and_motion(scene: Scene, settings: Settings) -> None:
    records: list[dict[str, Any]] = []
    planner = NemotronPlanner(record=records.append)
    options = planner.propose_options(scene)
    OptionsResponse(options=options)
    plan = planner.plan_motion(options[1], scene).validate_scene(scene)
    result = SimulatedArm.from_file(settings.joint_limits_path).execute(plan)
    assert result.success
    assert not any(record["type"] == "planner_fallback" for record in records)
    calls = [record for record in records if record["type"] == "planner_call"]
    assert {record["operation"] for record in calls if record["status"] == "ok"} == {"propose_options", "plan_motion"}
    assert all(record["latency_ms"] > 0 and record["usage"] is not None for record in calls)
