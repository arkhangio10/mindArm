"""Full cycles, cancellations, failures, and stale background results."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from threading import Event
from typing import cast

import pytest

from conftest import ScriptedDecoder
from mindarm.interfaces import ExecutionResult, MotionPlan, Option, Scene
from mindarm.orchestrator import Orchestrator, State
from mindarm.planner.fake import FakePlanner


def advance(app: Orchestrator, predicate: Callable[[], bool]) -> None:
    deadline = time.monotonic() + 3
    while not predicate() and time.monotonic() < deadline:
        app.tick()
        time.sleep(0.001)
    assert predicate(), f"Stopped in {app.state}: {app.last_error}"


def send(app: Orchestrator, kind: str, index: int | None = None) -> None:
    cast(ScriptedDecoder, app.decoder).send(kind, index)
    app.tick()


def engage(app: Orchestrator) -> None:
    send(app, "TOGGLE_ENGAGE")
    advance(app, lambda: app.state == State.SELECTING)


def records(app: Orchestrator) -> list[dict]:
    return [json.loads(line) for line in app.log.path.read_text(encoding="utf-8").splitlines()]


def test_full_second_option_cycle_and_recording(app: Orchestrator) -> None:
    engage(app)
    send(app, "SELECT", 1)
    advance(app, lambda: app.completed_cycles == 1)
    assert app.state == State.PROPOSING
    assert app.arm.state()["shoulder"] == 65
    cycle = [record for record in records(app) if record["type"] == "cycle"]
    assert len(cycle) == 1
    assert cycle[0]["selection"] == 1
    assert cycle[0]["plan"]["primitives"][0]["object_label"] == "bottle"
    assert cycle[0]["result"]["executed_primitives"] == 5
    assert set(cycle[0]["latencies"]) == {"perception_ms", "propose_ms", "plan_ms", "execute_ms"}
    send(app, "QUIT")
    assert not app.running and app.state == State.IDLE
    assert app.arm.state()["shoulder"] == 90
    assert list(app.settings.recordings_dir.glob("*.npz"))


@pytest.mark.parametrize("state", list(State))
def test_toggle_from_every_state(app: Orchestrator, state: State) -> None:
    if state != State.IDLE:
        send(app, "TOGGLE_ENGAGE")
    if state in (State.SELECTING, State.PLANNING, State.EXECUTING):
        advance(app, lambda: app.state == State.SELECTING)
    if state in (State.PLANNING, State.EXECUTING):
        send(app, "SELECT", 0)
    if state == State.EXECUTING:
        advance(app, lambda: app.state == State.EXECUTING)
        app.tick()  # Submit the first primitive; cancel while it may still be in flight.
    assert app.state == state
    send(app, "TOGGLE_ENGAGE")
    assert app.state == (State.PROPOSING if state == State.IDLE else State.IDLE)
    assert app.arm.state()["shoulder"] == 90
    app.shutdown()


def test_stale_proposal_cannot_reengage_or_start_extra_workers(app: Orchestrator) -> None:
    started, release = Event(), Event()

    class SlowPlanner(FakePlanner):
        """Hold the first response while the main thread accepts user input."""

        calls = 0

        def propose_options(self, scene: Scene) -> list[Option]:
            self.calls += 1
            if self.calls == 1:
                started.set()
                assert release.wait(2)
            return super().propose_options(scene)

    slow = SlowPlanner()
    app.planner = slow
    try:
        send(app, "TOGGLE_ENGAGE")
        app.tick()
        assert started.wait(1)
        send(app, "TOGGLE_ENGAGE")
        assert app.state == State.IDLE
        send(app, "TOGGLE_ENGAGE")
        for _ in range(5):
            app.tick()
        assert slow.calls == 1
        release.set()
        advance(app, lambda: app.state == State.SELECTING)
        assert slow.calls == 2  # Reengagement uses a new scene/options request.
    finally:
        release.set()
        app.shutdown()


def test_slow_motion_result_is_discarded_after_toggle(app: Orchestrator) -> None:
    started, release = Event(), Event()

    class SlowMotion(FakePlanner):
        """Delay motion planning to exercise cancellation."""

        def plan_motion(self, option: Option, scene: Scene) -> MotionPlan:
            started.set()
            assert release.wait(2)
            return super().plan_motion(option, scene)

    app.planner = SlowMotion()
    try:
        engage(app)
        send(app, "SELECT", 0)
        app.tick()
        assert started.wait(1)
        send(app, "TOGGLE_ENGAGE")
        release.set()
        app._worker.join(timeout=1)
        for _ in range(5):
            app.tick()
        assert app.state == State.IDLE
        assert app.completed_cycles == 0
        assert [item.action for item in app.arm.history] == ["HOME"]
    finally:
        release.set()
        app.shutdown()


def test_disengage_cancels_in_flight_primitive_before_home(app: Orchestrator) -> None:
    started, cancelled = Event(), Event()
    arm = app.arm

    class BlockingArm:
        """Hold a non-HOME primitive like a real ramp until cancel() is called."""

        def __init__(self) -> None:
            self.calls: list[str] = []

        def execute(self, plan: MotionPlan) -> ExecutionResult:
            action = plan.primitives[0].action
            self.calls.append(action)
            if action != "HOME":
                started.set()
                assert cancelled.wait(2)
                return ExecutionResult(success=False, executed_primitives=0,
                                       joint_angles=arm.state(), error="Stopped before the target")
            return arm.execute(plan)

        def state(self) -> dict[str, float]:
            return arm.state()

        def cancel(self) -> None:
            self.calls.append("cancel")
            cancelled.set()

    app.arm = BlockingArm()
    try:
        engage(app)
        send(app, "SELECT", 1)
        advance(app, started.is_set)
        for _ in range(3):
            app.tick()  # Input keeps being polled while the primitive is in flight.
        assert app.state == State.EXECUTING
        send(app, "TOGGLE_ENGAGE")
        assert app.state == State.IDLE
        assert app.arm.calls == ["MOVE_ABOVE", "cancel", "HOME"]
        assert not app._worker.is_alive()
        assert app.completed_cycles == 0 and app.last_error is None
    finally:
        cancelled.set()
        app.shutdown()


@pytest.mark.parametrize("phase", ["perception", "options", "plan", "eeg", "arm"])
def test_errors_home_idle_and_log(app: Orchestrator, monkeypatch: pytest.MonkeyPatch, phase: str) -> None:
    def fail(*args: object) -> None:
        raise RuntimeError("injected failure")

    if phase == "perception":
        monkeypatch.setattr(app.perception, "detect", fail)
    elif phase == "options":
        monkeypatch.setattr(app.planner, "propose_options", fail)
    elif phase == "eeg":
        monkeypatch.setattr(app.eeg, "read", fail)
    else:
        engage(app)
        if phase == "plan":
            monkeypatch.setattr(app.planner, "plan_motion", fail)
        else:
            execute = app.arm.execute

            def fail_motion(plan: MotionPlan) -> ExecutionResult:
                if plan.primitives[0].action == "HOME":
                    return execute(plan)
                return ExecutionResult(success=False, executed_primitives=0,
                                       joint_angles=app.arm.state(), error="injected failure")

            monkeypatch.setattr(app.arm, "execute", fail_motion)
        send(app, "SELECT", 0)
    if phase in ("perception", "options", "eeg"):
        send(app, "TOGGLE_ENGAGE")
    advance(app, lambda: app.last_error is not None)
    assert app.state == State.IDLE
    assert app.arm.state()["shoulder"] == 90
    assert any(record.get("error") == "injected failure" for record in records(app))
    assert any(record["type"] == "cycle" and record["status"] == "error" for record in records(app))
    app.shutdown()


def test_selection_before_options_is_ignored(app: Orchestrator) -> None:
    send(app, "SELECT", 1)
    assert app.state == State.IDLE
    assert app.completed_cycles == 0
    app.shutdown()


def test_multiple_cycles_repropose_cleanly(app: Orchestrator) -> None:
    engage(app)
    for count in (1, 2):
        send(app, "SELECT", 1)
        advance(app, lambda: app.completed_cycles == count)
        if count == 1:
            advance(app, lambda: app.state == State.SELECTING)
    cycles = [record for record in records(app) if record["type"] == "cycle"]
    assert len(cycles) == 2
    assert all(record["status"] == "completed" for record in cycles)
    app.shutdown()


def test_failed_console_cannot_prevent_home(app: Orchestrator) -> None:
    engage(app)
    send(app, "SELECT", 0)
    advance(app, lambda: app.state == State.EXECUTING)
    advance(app, lambda: app.arm.state()["shoulder"] != 90)

    def broken_output(message: str) -> None:
        raise BrokenPipeError("console closed")

    app._emit = broken_output
    try:
        with pytest.raises(BrokenPipeError):
            send(app, "TOGGLE_ENGAGE")
        assert app.state == State.IDLE
        assert app.arm.state()["shoulder"] == 90
    finally:
        app._emit = lambda _: None
        app.shutdown()
