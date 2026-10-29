"""Tick-driven state machine with cancellable intent and serialized arm execution."""

from __future__ import annotations

import time
from collections.abc import Callable
from concurrent.futures import Future
from enum import Enum
from threading import Thread
from typing import Any

from mindarm.config import Settings
from mindarm.eeg.recording import EEGRecording
from mindarm.interfaces import (
    ArmController, Decoder, EEGSource, ExecutionResult, MotionPlan,
    Option, OptionsResponse, Perception, Planner, Scene, home_plan,
)
from mindarm.session_log import SessionLog, utc_now


class State(str, Enum):
    """Externally observable orchestration states."""

    IDLE = "IDLE"
    PROPOSING = "PROPOSING"
    SELECTING = "SELECTING"
    PLANNING = "PLANNING"
    EXECUTING = "EXECUTING"


class Orchestrator:
    """Poll input first; discard stale work and return HOME on disengagement/error."""

    def __init__(
        self, eeg: EEGSource, decoder: Decoder, perception: Perception,
        planner: Planner, arm: ArmController, settings: Settings,
        log: SessionLog, emit: Callable[[str], None] = print,
    ) -> None:
        self.eeg, self.decoder, self.perception = eeg, decoder, perception
        self.planner, self.arm, self.settings, self.log = planner, arm, settings, log
        self.state = State.IDLE
        self.running = True
        self.completed_cycles = 0
        self.last_error: str | None = None
        self._emit = emit
        self._cycle: dict[str, Any] | None = None
        self._future: Future[Any] | None = None
        self._worker: Thread | None = None
        self._scene: Scene | None = None
        self._options: list[Option] = []
        self._selection: int | None = None
        self._plan: MotionPlan | None = None
        self._executed = 0
        self._execution_start = 0.0
        self._next_sample = 0.0
        self._recording = EEGRecording(eeg, settings.recordings_dir / f"{log.session_id}.npz", settings.record_eeg)

    def _transition(self, state: State) -> None:
        self.state = state
        self._emit(state.value)

    def _submit(self, function: Callable[[], Any]) -> None:
        # At most one worker/request can be live, even after repeated disengagement.
        if self._worker is not None and self._worker.is_alive():
            return
        future: Future[Any] = Future()
        self._future = future

        def work() -> None:
            try:
                future.set_result(function())
            except Exception as error:
                future.set_exception(error)

        self._worker = Thread(target=work, name="mindarm-planner", daemon=True)
        self._worker.start()

    def _finish_cycle(self, status: str, **fields: Any) -> None:
        if self._cycle is not None:
            cycle, self._cycle = self._cycle, None
            self.log.write({**cycle, **fields, "type": "cycle", "status": status, "finished_at": utc_now()})

    def _disengage(self, reason: str, error: Exception | None = None) -> None:
        self._future = None  # A stale worker can finish, but its result can never execute.
        executing = self.state == State.EXECUTING
        # Recovery must happen even if the output stream or logger has failed.
        self.state = State.IDLE
        recovery: ExecutionResult | None = None
        recovery_error: str | None = None
        try:
            if executing and self._worker is not None and self._worker.is_alive():
                # Stop the in-flight primitive and wait for it, so arm execution stays serialized.
                self.arm.cancel()
                self._worker.join()
            recovery = self.arm.execute(home_plan())
            if not recovery.success:
                recovery_error = recovery.error or "Arm rejected HOME"
        except Exception as home_error:
            recovery_error = str(home_error)
        if error is not None or recovery_error is not None:
            self.last_error = str(error) if error is not None else recovery_error
        details = {
            "reason": reason, "error": str(error) if error is not None else None,
            "error_type": type(error).__name__ if error is not None else None,
            "recovery": recovery.model_dump(mode="json") if recovery else None,
            "recovery_error": recovery_error,
        }
        self._finish_cycle("error" if error or recovery_error else "cancelled", **details)
        self.log.write({"type": "disengage", **details})
        self._plan = None
        self._emit(State.IDLE.value)
        if error is not None or recovery_error is not None:
            self._emit(f"Error: {self.last_error}")
        if recovery_error is not None:
            self._emit(f"HOME failed: {recovery_error}")

    def _begin_cycle(self) -> None:
        if self._cycle is None:
            self._cycle = {
                "started_at": utc_now(), "scene": None, "options": None,
                "selection": None, "plan": None, "result": None, "latencies": {},
            }
            self._scene, self._options, self._selection = None, [], None

    def _propose(self) -> None:
        assert self._cycle is not None
        if self._future is None:
            def propose() -> tuple[Scene, list[Option], dict[str, float]]:
                start = time.perf_counter()
                scene = self.perception.detect()
                detected = time.perf_counter()
                options = self.planner.propose_options(scene)
                return scene, options, {"perception_ms": (detected - start) * 1000,
                                        "propose_ms": (time.perf_counter() - detected) * 1000}
            self._submit(propose)
        elif self._future.done():
            self._scene, options, latencies = self._future.result()
            self._future = None
            self._options = OptionsResponse.model_validate({"options": [o.model_dump() for o in options]}).options
            self._cycle.update(scene=self._scene.model_dump(mode="json"),
                               options=[o.model_dump(mode="json") for o in self._options], latencies=latencies)
            self._transition(State.SELECTING)
            for index, option in enumerate(self._options, start=1):
                self._emit(f"{index}. {option.label}: {option.description}")

    def _planning(self) -> None:
        assert self._scene is not None and self._selection is not None and self._cycle is not None
        if self._future is None:
            option, scene = self._options[self._selection], self._scene

            def plan() -> tuple[MotionPlan, float]:
                start = time.perf_counter()
                result = self.planner.plan_motion(option, scene)
                return result, (time.perf_counter() - start) * 1000

            self._submit(plan)
        elif self._future.done():
            proposed, latency = self._future.result()
            self._future = None
            self._plan = MotionPlan.model_validate(proposed.model_dump()).validate_scene(self._scene)
            if self._options[self._selection].label == "Home" and self._plan != home_plan():
                raise ValueError("Home option must produce exactly [HOME]")
            self._cycle["plan"] = self._plan.model_dump(mode="json")
            self._cycle["latencies"]["plan_ms"] = latency
            self._executed = 0
            self._execution_start = time.perf_counter()
            self._transition(State.EXECUTING)

    def _execute(self) -> None:
        assert self._plan is not None and self._cycle is not None
        primitive = self._plan.primitives[self._executed]
        if self._future is None:
            self._submit(lambda: self.arm.execute(MotionPlan(primitives=[primitive])))
            return
        if not self._future.done():
            return
        result: ExecutionResult = self._future.result()
        self._future = None
        self._cycle["latencies"]["execute_ms"] = (time.perf_counter() - self._execution_start) * 1000
        self._cycle["result"] = result.model_dump(mode="json")
        self._cycle["result"]["executed_primitives"] = self._executed + result.executed_primitives
        if not result.success:
            raise RuntimeError(result.error or "Arm execution failed")
        if result.executed_primitives != 1:
            raise RuntimeError("Arm did not execute exactly one primitive")
        self._executed += 1
        self._emit(f"Executed {primitive.action}")
        if self._executed == len(self._plan.primitives):
            self._finish_cycle("completed")
            self.completed_cycles += 1
            self.last_error = None
            self._transition(State.PROPOSING)

    def tick(self) -> None:
        """Process one input event, acquisition block, and bounded state action."""
        if not self.running:
            return
        try:
            event = self.decoder.poll()
            if event is not None:
                if event.kind == "QUIT":
                    self.shutdown("quit")
                    return
                if event.kind == "TOGGLE_ENGAGE":
                    if self.state == State.IDLE:
                        self._transition(State.PROPOSING)
                    else:
                        self._disengage("toggle")
                    return
                if event.kind == "SELECT" and self.state == State.SELECTING:
                    self._selection = event.index
                    assert self._cycle is not None
                    self._cycle["selection"] = event.index
                    self._transition(State.PLANNING)
                    return
            if self.state == State.IDLE:
                return
            if self.state == State.PROPOSING:
                self._begin_cycle()
            now = time.monotonic()
            if now >= self._next_sample:
                self._recording.capture(self.settings.eeg_chunk_samples)
                self._next_sample = now + self.settings.eeg_chunk_samples / self.eeg.sample_rate
            if self.state == State.PROPOSING:
                self._propose()
            elif self.state == State.PLANNING:
                self._planning()
            elif self.state == State.EXECUTING:
                self._execute()
        except Exception as error:
            self._disengage("error", error)

    def shutdown(self, reason: str = "shutdown") -> None:
        """Return HOME, save replayable EEG, and finish the session once."""
        if not self.running:
            return
        self.running = False
        try:
            self._disengage(reason)
        finally:
            path = self._recording.save()
            self.log.write({"type": "session_end", "reason": reason,
                            "recording": str(path) if path else None,
                            "completed_cycles": self.completed_cycles})

    def run(self, once: bool = False) -> None:
        """Run until quit/Ctrl+C, or until one completed/failed cycle with --once."""
        self._emit("IDLE - e: engage/disengage | 1-4: select | q: quit")
        try:
            while self.running:
                self.tick()
                if once and (self.completed_cycles > 0 or self.last_error is not None):
                    break
                time.sleep(self.settings.poll_interval)
        except KeyboardInterrupt:
            pass
        finally:
            self.shutdown()
