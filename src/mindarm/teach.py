"""Teach the real arm's poses from a local web page, with the Jetson camera beside the controls."""

from __future__ import annotations

import argparse
import json
import logging
import re
import subprocess
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Condition, Thread
from typing import Any, Literal

import yaml

from mindarm.arm.jetson import JetsonArm, SshTransport
from mindarm.arm.simulated import ArmSettings, SimulatedArm
from mindarm.config import default_config_path, load_settings
from mindarm.interfaces import Contract, MotionPlan, MotionPrimitive, home_plan

CAMERA_COMMAND = ("v4l2-ctl -d /dev/video0 --set-fmt-video=width=640,height=480,pixelformat=MJPG "
                  "--set-parm=15 --stream-mmap --stream-to=- 2>/dev/null")
ARM_JOINTS = ["base", "shoulder", "elbow", "wrist_pitch", "wrist_roll"]
# GRASP and RELEASE only act on the gripper; LIFT and handover keep whatever is grasped.
SAVED_JOINTS = {"MOVE_ABOVE": ARM_JOINTS + ["gripper"], "GRASP": ["gripper"], "LIFT": ARM_JOINTS,
                "handover": ARM_JOINTS, "RELEASE": ["gripper"]}
DEMO_PLAN = MotionPlan(primitives=[
    MotionPrimitive(action="MOVE_ABOVE", object_label="bottle"), MotionPrimitive(action="GRASP"),
    MotionPrimitive(action="LIFT"), MotionPrimitive(action="MOVE_TO_POSE", pose_name="handover"),
    MotionPrimitive(action="RELEASE"),
])
Target = Literal["MOVE_ABOVE", "GRASP", "LIFT", "handover", "RELEASE"]


class MoveRequest(Contract):
    """Operator-chosen joint targets in µs."""

    targets: dict[str, float]


class NamedRequest(Contract):
    """A taught target to save or visit; HOME can only be visited."""

    name: Literal["HOME", "MOVE_ABOVE", "GRASP", "LIFT", "handover", "RELEASE"]


def split_frames(buffer: bytes) -> tuple[list[bytes], bytes]:
    """Split concatenated JPEGs; return complete frames and the unfinished remainder."""
    frames: list[bytes] = []
    while True:
        start = buffer.find(b"\xff\xd8")
        if start == -1:
            return frames, b""
        end = buffer.find(b"\xff\xd9", start + 2)
        if end == -1:
            return frames, buffer[start:]
        frames.append(buffer[start:end + 2])
        buffer = buffer[end + 2:]


def target_values(settings: ArmSettings, name: str) -> dict[str, float]:
    return settings.poses[name] if name in settings.poses else settings.primitive_targets[name]


def taught(settings: ArmSettings, name: Target, positions: dict[str, float]) -> ArmSettings:
    """Return settings with `name` taken from the arm's current positions."""
    data = settings.model_dump()
    values = {joint: positions[joint] for joint in SAVED_JOINTS[name]}
    if name == "handover":
        data["poses"]["handover"] = {**values, "gripper": settings.primitive_targets["GRASP"]["gripper"]}
    else:
        data["primitive_targets"][name] = values
    return ArmSettings.model_validate(data)


def write_targets(path: Path, settings: ArmSettings) -> None:
    """Rewrite only the taught lines, keeping the file's comments and layout."""
    text = path.read_text(encoding="utf-8")
    for name in SAVED_JOINTS:
        values = ", ".join(f"{joint}: {round(value)}" for joint, value in target_values(settings, name).items())
        text, count = re.subn(rf"^  {name}: \{{.*\}}$", f"  {name}: {{{values}}}", text, flags=re.MULTILINE)
        if count != 1:
            raise ValueError(f"{path} must contain exactly one line for {name}")
    path.write_text(text, encoding="utf-8", newline="")


def plan_check(settings: ArmSettings) -> str:
    """Preflight two demo cycles, as MindArm will run them, without touching the arm."""
    arm = SimulatedArm(settings)
    for _ in range(2):
        result = arm.execute(DEMO_PLAN)
        if not result.success:
            return f"Plan check failed: {result.error}"
    return f"Plan check OK at {settings.primitive_duration_s} s per primitive"


class CameraStream:
    """Keep the latest JPEG frame from v4l2-ctl's MJPEG output on the Jetson."""

    def __init__(self, command: list[str]) -> None:
        self.frame: bytes | None = None
        self.running = True
        self._ready = Condition()
        self._process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                         creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
                                         start_new_session=True)
        Thread(target=self._read, name="mindarm-camera", daemon=True).start()

    def _read(self) -> None:
        assert self._process.stdout is not None
        buffer = b""
        while chunk := self._process.stdout.read1(65536):
            frames, buffer = split_frames(buffer + chunk)
            if frames:
                with self._ready:
                    self.frame = frames[-1]
                    self._ready.notify_all()
        with self._ready:
            self.running = False
            self._ready.notify_all()

    def next_frame(self, timeout: float) -> bytes | None:
        with self._ready:
            self._ready.wait(timeout)
            return self.frame if self.running else None

    def close(self) -> None:
        self._process.kill()


class TeachApp:
    """Operator moves, pose saving, and pose visits; every move goes through JetsonArm's limits."""

    def __init__(self, arm: JetsonArm, settings: ArmSettings, path: Path, camera: CameraStream | None) -> None:
        self.arm, self.settings, self.path, self.camera = arm, settings, path, camera
        self.message = plan_check(settings)

    def state(self) -> dict[str, Any]:
        return {
            "joints": {joint: {"min": limit.min_angle, "max": limit.max_angle, "value": self.arm.state()[joint]}
                       for joint, limit in self.settings.joints.items()},
            "targets": {name: target_values(self.settings, name) for name in ["home", *SAVED_JOINTS]},
            "saved_joints": SAVED_JOINTS, "message": self.message,
        }

    def move(self, request: MoveRequest) -> None:
        result = self.arm.move_to(request.targets)
        self.message = result.error or "Moved"

    def save(self, name: Target) -> None:
        self.settings = taught(self.settings, name, self.arm.state())
        write_targets(self.path, self.settings)
        self.message = f"Saved {name} to {self.path.name}. {plan_check(self.settings)}"

    def visit(self, name: str) -> None:
        if name == "HOME":
            result = self.arm.execute(home_plan())
        else:
            result = self.arm.move_to(target_values(self.settings, name))
        self.message = result.error or f"At {name}"


class TeachHandler(BaseHTTPRequestHandler):
    """Serve the page, its JSON actions, and the camera as multipart MJPEG."""

    server: TeachServer

    def log_message(self, format: str, *args: Any) -> None:
        pass

    def _send(self, body: bytes, content_type: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int = 200) -> None:
        self._send(json.dumps(self.server.app.state()).encode("utf-8"), "application/json", status)

    def do_GET(self) -> None:
        if self.path == "/":
            self._send(PAGE.encode("utf-8"), "text/html; charset=utf-8")
        elif self.path == "/state":
            self._json()
        elif self.path == "/camera" and self.server.app.camera is not None:
            self._stream(self.server.app.camera)
        else:
            self._send(b"Not found", "text/plain", 404)

    def _stream(self, camera: CameraStream) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            while (frame := camera.next_frame(2.0)) is not None:
                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n%s\r\n"
                                 % (len(frame), frame))
        except OSError:
            pass  # The browser closed the stream.

    def do_POST(self) -> None:
        app = self.server.app
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))) or b"{}")
            if self.path == "/move":
                app.move(MoveRequest.model_validate(body))
            elif self.path == "/save":
                name = NamedRequest.model_validate(body).name
                if name == "HOME":
                    raise ValueError("HOME is the calibration rest pose and is not taught here")
                app.save(name)
            elif self.path == "/visit":
                app.visit(NamedRequest.model_validate(body).name)
            elif self.path == "/stop":
                app.arm.cancel()
                app.message = "Stop sent"
            else:
                return self._send(b"Not found", "text/plain", 404)
        except ValueError as error:
            app.message = f"Rejected: {error}"
            return self._json(400)
        self._json()


class TeachServer(ThreadingHTTPServer):
    """Localhost-only server carrying the teaching session."""

    def __init__(self, port: int, app: TeachApp) -> None:
        super().__init__(("127.0.0.1", port), TeachHandler)
        self.app = app


def main(argv: list[str] | None = None) -> int:
    """Energize the arm at HOME, serve the teaching page, and return HOME on Ctrl+C."""
    parser = argparse.ArgumentParser(description="Teach the real arm's poses in a local web page")
    parser.add_argument("--config", type=Path, default=default_config_path())
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--no-camera", action="store_true", help="Do not stream the Jetson camera")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    # Plan checks run on the simulator; keep its per-primitive lines out of the console.
    logging.getLogger("mindarm.arm.simulated").setLevel(logging.WARNING)
    camera: CameraStream | None = None
    arm: JetsonArm | None = None
    try:
        settings = load_settings(args.config)
        arm_settings = ArmSettings.model_validate(
            yaml.safe_load(settings.jetson_arm_path.read_text(encoding="utf-8")))
        arm = JetsonArm(arm_settings, SshTransport(settings.jetson_command))
        if not args.no_camera:
            camera = CameraStream(settings.jetson_command[:-1] + [CAMERA_COMMAND])
        server = TeachServer(args.port, TeachApp(arm, arm_settings, settings.jetson_arm_path, camera))
        print(f"Open http://localhost:{args.port}  (Ctrl+C returns HOME and exits)", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()
        return 0
    except (OSError, ValueError, RuntimeError) as error:
        print(f"MindArm teach: {error}", file=sys.stderr)
        return 1
    finally:
        if camera is not None:
            camera.close()
        if arm is not None:
            result = arm.execute(home_plan())
            print("HOME" if result.success else f"HOME failed: {result.error}", flush=True)
            arm.close()


PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MindArm pose teaching</title>
<style>
  :root { --bg: #f4f2ee; --panel: #ffffff; --ink: #1d1d1f; --muted: #6b6b70; --line: #dddad3;
          --accent: #2f6f62; --stop: #b3261e; }
  @media (prefers-color-scheme: dark) {
    :root { --bg: #161618; --panel: #222225; --ink: #ececee; --muted: #9a9aa0; --line: #38383c;
            --accent: #6fc2ae; --stop: #ff6b5f; }
  }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--bg); color: var(--ink);
         font: 15px/1.4 system-ui, -apple-system, "Segoe UI", sans-serif; }
  header { display: flex; align-items: center; justify-content: space-between; gap: 16px;
           padding: 14px 20px; border-bottom: 1px solid var(--line); }
  h1 { margin: 0; font-size: 18px; } header p { margin: 2px 0 0; color: var(--muted); font-size: 13px; }
  main { display: grid; grid-template-columns: minmax(0, 640px) minmax(0, 1fr); gap: 20px; padding: 20px; }
  @media (max-width: 1000px) { main { grid-template-columns: 1fr; } }
  section { background: var(--panel); border: 1px solid var(--line); border-radius: 10px; padding: 16px; }
  h2 { margin: 0 0 12px; font-size: 14px; text-transform: uppercase; letter-spacing: .06em; color: var(--muted); }
  #camera { width: 100%; aspect-ratio: 4 / 3; border-radius: 6px; background: #000; display: block; }
  .joint { display: grid; grid-template-columns: 110px 1fr 74px auto; align-items: center; gap: 10px;
           padding: 8px 0; border-top: 1px solid var(--line); }
  .joint:first-of-type { border-top: 0; }
  .joint b { font-weight: 600; } .joint small { display: block; color: var(--muted); font-size: 12px; }
  .value { font-variant-numeric: tabular-nums; text-align: right; }
  input[type=range] { width: 100%; accent-color: var(--accent); }
  button { font: inherit; border: 1px solid var(--line); background: var(--bg); color: var(--ink);
           border-radius: 6px; padding: 6px 10px; cursor: pointer; }
  button:hover:not(:disabled) { border-color: var(--accent); }
  button:disabled { opacity: .45; cursor: wait; }
  .steps { display: flex; gap: 4px; } .steps button { padding: 4px 7px; min-width: 40px; }
  .row { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 8px; }
  .hint { color: var(--muted); font-size: 13px; margin: 0 0 14px; }
  #stop { background: var(--stop); color: #fff; border-color: var(--stop); font-weight: 700; padding: 10px 18px; }
  #message { margin-top: 12px; padding: 10px 12px; border-radius: 6px; background: var(--bg); min-height: 42px; }
  .stack { display: grid; gap: 20px; align-content: start; }
</style>
</head>
<body>
<header>
  <div><h1>MindArm · pose teaching</h1>
  <p>Real arm. Every move is ramped inside the configured limits. HOME is the calibration rest pose.</p></div>
  <button id="stop" title="Stop the current ramp where it is">STOP</button>
</header>
<main>
  <div class="stack">
    <section><h2>Jetson camera</h2><img id="camera" src="/camera" alt="Live camera view"></section>
    <section><h2>Visit a saved target</h2><div class="row" id="visit"></div>
      <div id="message">Loading…</div></section>
  </div>
  <div class="stack">
    <section><h2>Joints (µs)</h2><div id="joints"></div></section>
    <section><h2>Save the current position as</h2>
      <p class="hint">Teach in order: MOVE_ABOVE with the gripper open, GRASP closed, then LIFT, handover, RELEASE.</p>
      <div id="save"></div></section>
  </div>
</main>
<script>
const STEPS = [-50, -10, 10, 50];
const HINTS = {MOVE_ABOVE: "arm joints + gripper (open it first)", GRASP: "gripper only",
  LIFT: "arm joints; keeps the grasp", handover: "arm joints; gripper stays at GRASP", RELEASE: "gripper only"};
let state = null, busy = false;

function setBusy(value) {
  busy = value;
  document.querySelectorAll("button:not(#stop), input").forEach(el => el.disabled = value);
}
async function call(path, body) {
  if (busy) return;
  setBusy(true);
  document.getElementById("message").textContent = "Moving…";
  try {
    const response = await fetch(path, body === undefined ? {} :
      {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body)});
    render(await response.json());
  } catch (error) {
    document.getElementById("message").textContent = "Connection lost: " + error;
  } finally { setBusy(false); }
}
function move(joint, value) { call("/move", {targets: {[joint]: value}}); }

function build() {
  const joints = document.getElementById("joints");
  Object.entries(state.joints).forEach(([name, joint], channel) => {
    const row = document.createElement("div");
    row.className = "joint";
    row.innerHTML = `<div><b>${name}</b><small>ch ${channel} · ${joint.min}–${joint.max}</small></div>
      <input type="range" min="${joint.min}" max="${joint.max}" step="10" aria-label="${name}">
      <div class="value"></div><div class="steps"></div>`;
    const slider = row.querySelector("input"), label = row.querySelector(".value");
    slider.oninput = () => label.textContent = slider.value + " µs";
    slider.onchange = () => move(name, Number(slider.value));
    STEPS.forEach(step => {
      const button = document.createElement("button");
      button.textContent = (step > 0 ? "+" : "") + step;
      button.onclick = () => move(name, state.joints[name].value + step);
      row.querySelector(".steps").append(button);
    });
    joints.append(row);
  });
  const save = document.getElementById("save"), visit = document.getElementById("visit");
  Object.keys(state.saved_joints).forEach(name => {
    const row = document.createElement("div");
    row.className = "row";
    row.innerHTML = `<button>${name}</button><span class="hint">${HINTS[name]}</span>`;
    row.querySelector("button").onclick = () => call("/save", {name});
    save.append(row);
  });
  ["HOME", ...Object.keys(state.saved_joints)].forEach(name => {
    const button = document.createElement("button");
    button.textContent = name;
    button.onclick = () => call("/visit", {name});
    visit.append(button);
  });
}
function render(next) {
  const first = state === null;
  state = next;
  if (first) build();
  document.querySelectorAll(".joint").forEach((row, index) => {
    const joint = Object.values(state.joints)[index];
    row.querySelector("input").value = joint.value;
    row.querySelector(".value").textContent = Math.round(joint.value) + " µs";
  });
  document.getElementById("message").textContent = state.message;
}
document.getElementById("stop").onclick = () => fetch("/stop", {method: "POST"}).then(r => r.json()).then(render);
document.getElementById("camera").onerror = event => event.target.alt = "Camera unavailable";
call("/state");
</script>
</body>
</html>
"""

if __name__ == "__main__":
    raise SystemExit(main())
