"""Drive the real arm through the Jetson service over SSH stdio, after the simulator's preflight."""

from __future__ import annotations

import json
import logging
import subprocess
import time
from queue import Empty, Queue
from threading import Lock, Thread
from typing import Any

from pydantic import Field

from mindarm.arm.simulated import ArmSettings, SimulatedArm
from mindarm.interfaces import ArmTransport, Contract, ExecutionResult, MotionPlan

LOGGER = logging.getLogger(__name__)

STAGGER_S = 0.5
MIN_MOVE_S = 0.05
LAUNCH_TIMEOUT_S = 20.0
REPLY_MARGIN_S = 5.0


class ServiceReply(Contract):
    """One service line; id 0 is the launch report. Positions are µs, or None when released."""

    id: int = Field(ge=0)
    ok: bool
    stopped: bool
    positions: dict[str, int | None]
    error: str | None


class SshTransport:
    """Run the Jetson arm service through an SSH client and exchange JSON lines on its stdio."""

    def __init__(self, command: list[str]) -> None:
        self._process = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            # Ctrl+C must reach only MindArm, so it can still command HOME over this link.
            creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0), start_new_session=True)
        self._lines: Queue[bytes | None] = Queue()
        Thread(target=self._read, name="mindarm-jetson-link", daemon=True).start()

    def _read(self) -> None:
        assert self._process.stdout is not None
        for line in self._process.stdout:
            self._lines.put(line)
        self._lines.put(None)

    def send(self, message: dict[str, Any]) -> None:
        assert self._process.stdin is not None
        self._process.stdin.write(json.dumps(message).encode("utf-8") + b"\n")
        self._process.stdin.flush()

    def receive(self, timeout: float) -> dict[str, Any] | None:
        try:
            line = self._lines.get(timeout=timeout)
        except Empty:
            return None
        if line is None:
            self._lines.put(None)  # Every later receive also reports the closed link.
            raise ConnectionError("Jetson arm link closed")
        return json.loads(line)

    def close(self) -> None:
        """Close stdin; the service then stops and holds its last pulses."""
        assert self._process.stdin is not None
        self._process.stdin.close()
        try:
            self._process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._process.kill()


class JetsonArm(SimulatedArm):
    """Validate exactly like the simulator, then ramp the real servos through the Jetson service."""

    def __init__(self, settings: ArmSettings, transport: ArmTransport) -> None:
        super().__init__(settings)
        self._transport = transport
        self._execute_lock = Lock()
        self._send_lock = Lock()
        self._cancels = 0
        self._next_id = 1
        launch = self._reply(0, LAUNCH_TIMEOUT_S)
        known = {joint: pulse for joint, pulse in launch.positions.items() if pulse is not None}
        home = self._settings.poses["home"]
        duration = max([self._settings.primitive_duration_s] + [
            abs(home[joint] - pulse) / self._settings.joints[joint].max_speed for joint, pulse in known.items()])
        LOGGER.info("Energizing released joints at HOME, one every %.1fs: %s", STAGGER_S,
                    ", ".join(joint for joint in home if joint not in known) or "none")
        reply = self._request({"cmd": "start", "targets": self._pulses(home), "duration": duration,
                               "stagger": STAGGER_S}, len(home) * STAGGER_S + duration + REPLY_MARGIN_S)
        if not reply.ok:
            raise RuntimeError(f"Jetson arm start failed: {reply.error}")
        self._angles = self._positions(reply)

    def cancel(self) -> None:
        self._cancels += 1
        with self._send_lock:
            self._transport.send({"cmd": "stop"})

    def close(self) -> None:
        """End the session; the service stops and holds its last pulses."""
        self._transport.close()

    def execute(self, plan: MotionPlan) -> ExecutionResult:
        cancels = self._cancels
        with self._execute_lock:
            try:
                staged = self._stage(plan)
            except ValueError as error:
                return self._failed(0, str(error))
            for done, (primitive, target, duration) in enumerate(staged):
                if cancels != self._cancels:
                    return self._failed(done, "Cancelled before motion")
                error = self._move(target, duration)
                if error is not None:
                    return self._failed(done, error)
                self.history.append(primitive.model_copy(deep=True))
                LOGGER.info("%s %s -> %s (%.2fs)", primitive.action,
                            primitive.object_label or primitive.pose_name or "", self._angles, duration)
            return ExecutionResult(success=True, executed_primitives=len(staged), joint_angles=self.state())

    def move_to(self, target: dict[str, float]) -> ExecutionResult:
        """Move to an operator-chosen pose for teaching; not part of ArmController, so no planner reaches it."""
        with self._execute_lock:
            if not set(target) <= set(self._settings.joints):
                return self._failed(0, "Unknown joints: " + ", ".join(sorted(set(target) - set(self._settings.joints))))
            current = self.state()
            target = {**current, **target}
            duration = max([MIN_MOVE_S] + [abs(target[joint] - current[joint]) / limit.max_speed
                                           for joint, limit in self._settings.joints.items()])
            try:
                self._check_move(current, target, duration)
            except ValueError as error:
                return self._failed(0, str(error))
            error = self._move(target, duration)
            if error is not None:
                return self._failed(0, error)
            return ExecutionResult(success=True, executed_primitives=1, joint_angles=self.state())

    def _move(self, target: dict[str, float], duration: float) -> str | None:
        try:
            reply = self._request({"cmd": "move", "targets": self._pulses(target), "duration": duration},
                                  duration + REPLY_MARGIN_S)
            self._angles = self._positions(reply)
        except (OSError, ValueError) as error:
            return f"Jetson arm link: {error}"
        return None if reply.ok else reply.error or "Jetson arm rejected the move"

    def _failed(self, executed: int, error: str) -> ExecutionResult:
        return ExecutionResult(success=False, executed_primitives=executed, joint_angles=self.state(), error=error)

    @staticmethod
    def _pulses(target: dict[str, float]) -> dict[str, int]:
        return {joint: round(value) for joint, value in target.items()}

    @staticmethod
    def _positions(reply: ServiceReply) -> dict[str, float]:
        positions = {joint: float(pulse) for joint, pulse in reply.positions.items() if pulse is not None}
        if len(positions) != len(reply.positions):
            raise ValueError("Jetson arm reports released joints")
        return positions

    def _request(self, message: dict[str, Any], timeout: float) -> ServiceReply:
        request_id, self._next_id = self._next_id, self._next_id + 1
        with self._send_lock:
            self._transport.send({**message, "id": request_id})
        return self._reply(request_id, timeout)

    def _reply(self, request_id: int, timeout: float) -> ServiceReply:
        deadline = time.monotonic() + timeout
        while True:
            message = self._transport.receive(max(0.0, deadline - time.monotonic()))
            if message is None:
                raise TimeoutError(f"no reply to request {request_id} within {timeout:.1f}s")
            reply = ServiceReply.model_validate(message)
            if reply.id != request_id:
                LOGGER.warning("Ignoring stale Jetson arm reply %d", reply.id)
                continue
            if set(reply.positions) != set(self._settings.joints):
                raise ValueError("Jetson arm joints do not match the configured joints")
            return reply
