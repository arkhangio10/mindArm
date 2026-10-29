# MindArm specification — Step 2

Status: implemented walking skeleton plus its first real module, an optional Jetson-driven
arm (`arm: jetson`). Intent selection is still a keyboard-driven simulation. This document
is the source of truth for interfaces, observable behavior, safety boundaries, and open
risks. Update it whenever a contract changes.

## Scope and acceptance

The laptop demo runs without hardware. `python -m mindarm` uses real Nemotron by default
and fake EEG/perception, keyboard intent selection, and a simulated arm. After **e** and
four displayed options, **2** selects index 1 and a validated motion plan executes.
`--planner fake` provides the same loop offline. Python 3.10+ is required.

With `arm: jetson`, the same loop moves the physical 3D-printed arm through a service on
the Jetson Nano, reached over SSH stdio. The simulator remains the configured default and
the only arm used by the default tests.

Out of scope: serial/Arduino integration, camera-based perception (the camera is only viewed
while teaching poses), object detectors, SSVEP/alpha decoding,
flickering UI, dashboards, web search, robot foundation models, motor imagery, inverse
kinematics, and servo position feedback.

## Architecture and files

`src/mindarm/interfaces.py` defines Pydantic v2 models and runtime-checkable Protocols.
`config.py` loads validated YAML. `factory.py` is the only implementation-selection point.
The orchestrator depends on Protocols, so future adapters replace fakes without changing
the orchestration contract. `session_log.py` owns JSONL writes; `eeg/recording.py` owns
recording validation and persistence. `__main__.py` handles setup and terminal cleanup.
`arm/jetson.py` is the laptop adapter for the real arm. `jetson/mindarm_arm.py` is the
service that runs on the Jetson. It is not part of the package and is deployed by copying
it to the Jetson's home directory. `teach.py` is the operator's pose teaching tool.

The default editable install locates `config/settings.yaml` relative to the source checkout.
A non-editable installation must use `--config` pointing to a retained checkout/configuration.
Relative fixture/log/recording paths resolve from the selected YAML file's directory.
`.env` is read from that directory's parent, without overriding exported environment values.

## Interfaces

All interfaces are synchronous. `Decoder.poll()` must return immediately. EEG reads are
bounded in-memory operations. `ArmController.execute()` may block for real motion (about
1–2 s per primitive). Perception/planner calls and arm execution run on one background
worker. `ArmController.cancel()` must return immediately. It makes an in-progress
`execute()` stop at the current position and return `success=False`; it is a no-op when idle.

```python
class EEGSource(Protocol):
    @property
    def sample_rate(self) -> float: ...
    @property
    def n_channels(self) -> int: ...
    def read(self, n_samples: int) -> np.ndarray: ...

class Decoder(Protocol):
    def poll(self) -> DecoderEvent | None: ...

class Perception(Protocol):
    def detect(self) -> Scene: ...

class Planner(Protocol):
    def propose_options(self, scene: Scene) -> list[Option]: ...
    def plan_motion(self, option: Option, scene: Scene) -> MotionPlan: ...

class ArmController(Protocol):
    def execute(self, plan: MotionPlan) -> ExecutionResult: ...
    def state(self) -> dict[str, float]: ...
    def cancel(self) -> None: ...

class ArmTransport(Protocol):
    def send(self, message: dict[str, Any]) -> None: ...
    def receive(self, timeout: float) -> dict[str, Any] | None: ...  # None on timeout
    def close(self) -> None: ...
```

### Shared data models

Models forbid extra fields, implicit coercion, NaN, and infinity. All indexes are zero-based;
displayed keys are one-based. Joint positions (`state()`, `joint_angles`, limits, and poses)
use the configured arm's unit: degrees for the simulator, and PWM pulse width in
microseconds (µs) for the Jetson arm. The YAML fields keep their `*_angle` names.

| Model | Fields and invariants |
| --- | --- |
| `DecoderEvent` | `kind`: SELECT / TOGGLE_ENGAGE / QUIT; `index`: 0..3 required only for SELECT |
| `DetectedObject` | Nonempty `label`; finite `x`, `y`, `confidence` in [0, 1] |
| `Scene` | `objects: list[DetectedObject]`; an empty scene is allowed |
| `Option` | Nonempty `label` (max 100 characters), `description` (max 500) |
| `OptionsResponse` | JSON envelope `options`; exactly four, distinct labels, exactly one spelled `Home` |
| `MotionPrimitive` | Closed `action`; optional `object_label` and `pose_name`, constrained as below |
| `MotionPlan` | `primitives`, an ordered list of 1..32 primitives |
| `ExecutionResult` | `success: bool`, `executed_primitives: int >= 0`, `joint_angles: dict[str, float]`, `error: str \| None` |

`QUIT` is the approved extension to the original two decoder event kinds so keyboard
shutdown is explicit within the same decoder contract.

### Motion vocabulary

| Primitive | Required arguments | Meaning in this step |
| --- | --- | --- |
| HOME | None | Return to local home pose |
| MOVE_ABOVE | `object_label` found in the current scene | Apply a local approach target; log the label |
| GRASP | None | Apply local grip target |
| LIFT | None | Apply local lift target |
| MOVE_TO_POSE | `pose_name`: `home` or `handover` | Apply a local named pose |
| RELEASE | None | Apply local release target |

Unrelated arguments and all other actions are rejected. Optional arguments may be omitted
or null where unused. Scene object membership is checked by `MotionPlan.validate_scene()`
at planner and orchestrator boundaries. Home must map to exactly one HOME primitive.

Example plan:

```json
{"primitives":[{"action":"MOVE_ABOVE","object_label":"bottle"},{"action":"GRASP"},{"action":"LIFT"},{"action":"MOVE_TO_POSE","pose_name":"handover"},{"action":"RELEASE"}]}
```

## Implementations

### EEG

`FakeEEGSource` defaults to one channel, 250 Hz sampling, a 12 Hz unit-amplitude sine, and
Gaussian noise with standard deviation 0.2. Rate, channels, frequency, noise, and seed are
configurable. Frequency must be below Nyquist. Phase and seeded noise remain continuous
across reads. Optional alpha bursts add amplitude-4, 10 Hz sine waves during seconds 4-6
of each six-second signal period. The units are synthetic and uncalibrated.

`read(n_samples)` requires a positive integer and returns finite float64 arrays shaped
`(n_channels, n_samples)`. Reads do not sleep. Acquisition is paced by the orchestrator
while engaged, using the source's actual sample rate and configured chunk size.

`ReplayEEGSource` uses `np.load(..., allow_pickle=False)`. Required NPZ entries:

- `data`: finite numeric, nonempty two-dimensional channels x samples array.
- `sample_rate`: finite positive numeric scalar.
- `n_channels`: positive integer scalar matching the first data dimension.

With looping enabled, reads wrap seamlessly even for requests longer than the recording.
With looping disabled, a read exceeding remaining samples raises `EOFError` without
advancing or returning a partial block. End-of-file disengages through the normal error path.

`EEGRecording` checks every block, retains copies in memory, and writes a compressed NPZ
on orderly shutdown. It creates no archive when recording is disabled or no samples were
captured. Acquisition resumes after reengagement; the stored signal omits idle wall time.
Keyboard intent is deliberately independent of synthetic/replayed EEG in this step.

### Keyboard and perception

Keyboard keys **1-4** produce SELECT(0..3); **e** toggles engagement; **q** produces QUIT.
EOF also quits; Ctrl+C triggers cleanup. Windows console input uses `msvcrt`; POSIX TTYs
use cbreak mode with restoration. Redirected input uses a background reader. Control
input is polled before state work on every tick. Selection outside SELECTING is ignored.

`FakePerception` reloads and validates `fixtures/scene_table.json` per detection. The fixture
contains cup, bottle, and sports ball with normalized image coordinates and confidence.

### Planners

`FakePlanner` provides up to three unique object-pickup options plus Home, padding short
scenes with generic options. An object pickup yields MOVE_ABOVE, GRASP, LIFT,
MOVE_TO_POSE(handover), RELEASE; generic options yield HOME.

`NemotronPlanner` uses `openai.OpenAI` chat completions against the configured Nebius
endpoint. Defaults: `https://api.tokenfactory.us-central1.nebius.com/v1/` and
`nvidia/nemotron-3-super-120b-a12b`. The API key comes from `NEBIUS_API_KEY`.
The request uses `response_format={"type":"json_object"}`, temperature 0.1, a configurable
token budget (default 4096), and request timeout (default 30 seconds). Prompts include
the schema, action constraints, and scene as text. No images are sent.

1. Validate JSON, Pydantic fields, exact option count/Home, and motion arguments.
2. Reject non-stop finish reasons (including token-truncated responses).
3. On invalid output, retry once with the validation error included in the conversation.
4. After the second invalid output, return Wait / Rest / Reset / Home options or `[HOME]`.
5. Home and those generic resting labels must map to `[HOME]` even after a later valid response.

SDK automatic retries are disabled; at most two calls occur for an invalid-output operation.
Authentication, transport, and service errors are logged and raised as `PlannerUnavailable`;
the orchestrator returns HOME + IDLE. Errors log the API exception type, not raw upstream
bodies or credentials. Missing credentials fail startup. Every completed API attempt logs
latency and usage (null when the API did not supply it). Safe fallback is explicit in logs.

### Simulated arm and limits

`config/joint_limits.yaml` defines six joints by default. The joint set is configurable,
with min/max degrees and maximum degrees/second for each joint. Both named poses must
specify all joints. Primitive targets can modify subsets of configured joints.

The simulation maps primitives to these local constants. MOVE_ABOVE does not perform
inverse kinematics or convert image coordinates into physical coordinates; different
object labels currently share the same approach target.

For each submitted plan, `execute()` revalidates its models and preflights all intermediate
joint targets before modifying any joint. It rejects an invalid submission with
`success=False`, zero executed primitives, unchanged angles, and an error. It does not clamp.
Speed is `abs(target - current) / primitive_duration_s`; default virtual duration is 2 s.
HOME may lengthen that duration to satisfy speed limits; it never bypasses angle limits.

After validation, execution logs each primitive and updates angles instantly using virtual
time. There are no motors or movement sleeps. `state()` returns a defensive copy.
`cancel()` is a no-op because virtual motion is instantaneous.

The orchestrator submits one primitive at a time, making disengagement possible between
primitives (and, on the Jetson arm, during one). Consequently a later primitive's joint
rejection can occur after earlier primitives completed; the cycle records partial progress,
then attempts HOME. Atomic preflight applies to each `execute()` submission, not the entire
orchestrated cycle.

### Jetson arm

Hardware: a PCA9685 at `0x40` on Jetson I2C bus 1, at 50 Hz (prescaler 121). Channels 0–5
are base, shoulder, elbow, wrist_pitch, wrist_roll, and gripper. The servos have no position
feedback. `config/jetson_arm.yaml` uses the simulator's schema in µs and µs/s. The service
knows exactly these six joints, and the configured joint set must match them.

`JetsonArm` (laptop) applies the simulator's preflight unchanged: every target within
limits, `abs(target - current) / duration <= max_speed`, no clamping, and whole-submission
rejection before any motion. It then sends each primitive as one `move` with integer
absolute targets for all joints and the checked duration. The service ramps linearly over
that duration, so the physical speed is the checked speed. After each reply, including
rejected or stopped moves, `state()` takes the positions the service reports. A lock
serializes `execute()` calls. `cancel()` sends `stop` and marks executions that started
before it as cancelled, so they return before sending further moves.

Wire protocol: one JSON object per line over the SSH exec channel's stdin/stdout. The
Jetson's ufw and its disabled SSH port forwarding stay unchanged.

| Direction | Message | Meaning |
| --- | --- | --- |
| Jetson → laptop | `{"id": 0, "ok": true, "stopped": false, "positions": {...}, "error": null}` | Launch report. A position is the last commanded µs, or `null` when the channel is released |
| laptop → Jetson | `{"cmd": "start", "id": n, "targets": {...}, "duration": s, "stagger": s}` | Energize released channels at their targets, then ramp the others |
| laptop → Jetson | `{"cmd": "move", "id": n, "targets": {...}, "duration": s}` | Ramp all six joints to the targets in `duration` |
| laptop → Jetson | `{"cmd": "stop"}` | Stop the current ramp at its present pulses. No reply of its own; ignored when idle |
| Jetson → laptop | `{"id": n, "ok": bool, "stopped": bool, "positions": {...}, "error": str \| null}` | Exactly one reply per `start`/`move` |

The service (`jetson/mindarm_arm.py`) runs on the Jetson's Python 3.6 with only
`python3-smbus`. It enforces its own hard envelope, independent of the laptop
configuration. That envelope is the calibration table below and a speed cap of 1000 µs/s.
It rejects with `ok: false` and no motion:

- requests whose targets are not exactly the six joints, or are not integers;
- targets outside the envelope, speeds above the cap, or durations outside (0, 10] s;
- a `stagger` outside [0, 2] s;
- a `move` while any channel is released.

A ramp writes changed pulses every 20 ms, each as one four-byte I2C transaction. Between
steps the service checks stdin: any received line or end of input stops the ramp at the
pulses already written and replies `stopped: true`. A move that changes nothing returns at once.

Startup and the first move have no position feedback. At launch the service adopts the
PCA9685 if it is awake at prescaler 121, and reads each channel's last commanded pulse.
Otherwise it releases all channels and configures 50 Hz. `JetsonArm` then sends `start`
with the HOME pose. The service energizes the released channels one at a time, in channel
order, directly at their HOME pulse, waiting 0.5 s between them. Each of those servos jumps
at its own maximum speed from an unknown position, so the operator places the arm near
HOME by hand before starting MindArm. Already-energized channels then ramp to HOME within
the configured speed limits. A startup failure fails setup with exit status 1.

Disconnection, chosen behavior "stop and hold": on end of input, SIGHUP, SIGTERM, or a
broken stdout, the service stops any ramp and exits without releasing. The PCA9685 keeps
generating the last pulses, so the arm holds its posture. A normal quit commands HOME first.
After MindArm exits, the arm therefore stays energized at HOME, and nothing moves while the
link is down. Releasing is manual: cut the servo supply, or run `servo_test.py`, which
releases every channel when it starts (the shoulder and elbow then fall).

The SSH client uses keep-alives (2 s × 3), so a lost link ends the session. `JetsonArm`
waits for each reply up to the move duration + 5 s and otherwise returns a failure. The SSH
client runs in its own process group, so Ctrl+C reaches only MindArm, which can still
command HOME over the link.

Poses: HOME is the calibration rest vector, with the gripper half-open at 1500 µs. HOME
after an error therefore releases a grasped object. The other targets were taught on
2026-10-28 for one sponge at a taped pick spot, which every object label shares:

- MOVE_ABOVE: base 1600, shoulder 1230, elbow 2000, wrist 1400/1010, gripper open at 1900.
- GRASP: gripper 1150.
- LIFT: about 5 cm up.
- handover: base 1000, shoulder 1430, elbow 1800, gripper 1150; RELEASE drops the sponge there.
- RELEASE: gripper 1900.

The teaching operator was Claude, watching the camera, with the user confirming depth
and contact by eye. The teaching surfaced these observations:

- The straight joint-space path from HOME reaches the pick spot partly from the side. It
  surrounds the sponge only when the sponge sits exactly on the taped spot.
- A grip of 1450 µs dropped the sponge during a base swing; 1150 µs held it.
- The elbow (2000) and base (1000) targets sit at the configured limits. Those limits are
  the bring-up clamp, not measured stops.
- The wrist_pitch MG90S stalled silently once, from 1550 to 1750 µs, until it was freed by
  hand.
- The pick→handover move runs the base at its 400 µs/s limit.

### Pose teaching tool

`python -m mindarm.teach` energizes the arm exactly like MindArm's startup and serves a page
on `127.0.0.1:8766` only. The page shows the Jetson camera as MJPEG, from a second SSH session
running `v4l2-ctl` at 640×480 (`--no-camera` skips it). The command requests 15 fps; on
2026-10-28 the camera delivered about 23 fps at ~54 KB per frame (~1.3 MB/s over the USB link). It has one slider and
±10/±50 µs buttons per joint, "save as" buttons for the five taught targets, "visit" buttons,
and STOP (`cancel()`).

Operator moves use `JetsonArm.move_to()`. It applies the same limit and speed check, with the
duration set by the slowest joint's `max_speed` (at least 0.05 s), and the service envelope
still applies. It is not part of the `ArmController` Protocol, so planner output can never
reach it.

Saving rewrites only that target's line in `config/jetson_arm.yaml` and keeps the comments:

- MOVE_ABOVE stores all six joints; teach it at the grasp position with the gripper open.
- GRASP and RELEASE store only the gripper.
- LIFT stores the five arm joints.
- handover stores the five arm joints, plus the GRASP gripper value, so the object stays held.

HOME is not taught here. After each save, the tool preflights two consecutive demo pickup
cycles on the simulator at `primitive_duration_s` and reports any limit or speed violation.
A visit follows the same straight joint-space path that MindArm will take, only at the
speed-limited duration. Ctrl+C commands HOME and exits.

## Orchestrator and cancellation

| Current state | Event or completion | Next state / action |
| --- | --- | --- |
| IDLE | TOGGLE_ENGAGE | PROPOSING |
| PROPOSING | Scene and four options ready | SELECTING |
| SELECTING | SELECT(index) | PLANNING |
| PLANNING | Valid motion plan ready | EXECUTING |
| EXECUTING | Last primitive succeeds | Log cycle, then PROPOSING |
| Any engaged state | TOGGLE_ENGAGE | Discard pending result, HOME, IDLE |
| Any state | Error | HOME, IDLE, log error/recovery outcome |
| Any state | QUIT / EOF / Ctrl+C | HOME, IDLE, save recording, exit |

IDLE's engagement behavior takes precedence over the generic toggle-to-IDLE rule.
Choosing Home executes HOME and reproposes options while staying engaged. Pressing **e**
homes and disengages. An error is recoverable with a later **e**. Failed HOME is reported
explicitly; the program never claims recovery succeeded when it did not.

At most one daemon worker handles perception, planning, and arm execution. Toggling
invalidates its future; its eventual response cannot change state or execute a plan.
Reengagement waits for that worker to finish before issuing another request. This bounds
work during repeated toggles. An in-flight HTTP request itself is not cancelled; it can
consume tokens and emit a late call record while the process remains alive. Quit does not
wait for a planner worker.

Interruptible execution: in EXECUTING, each primitive runs on the worker while the main
loop keeps polling input. Disengagement, errors, and quit during EXECUTING call
`arm.cancel()`, wait for the in-flight `execute()` to return, and only then submit HOME, so
arm execution stays serialized. The HOME recovery runs on the main thread and is not
interruptible; keys pressed meanwhile are processed afterwards.

Default poll interval is 20 ms. EEG uses its independent chunk cadence. No new motion
primitive is submitted before the next input poll.

`--once` exits after one successful cycle or error, attempts HOME, and saves recordings.
CLI exit status is 0 after a normal run, or 1 for setup failures or an unrecovered recorded
error. A subsequent successful cycle clears the prior recoverable error status. The CLI
replaces characters that its console encoding cannot represent (for example U+2011 under
cp1252 when output is redirected on Windows), so model text cannot crash a cycle.

## JSONL schema and observability

Files: `logs/<UTC-session-id>-<unique-suffix>.jsonl`. Every line is one JSON object with
`type`, UTC `timestamp`, and `session_id`. Writes are locked across threads and flushed
by closing the file after each append. Logs never contain configured API keys.

Record types: `session_start`, `configuration`, `planner_call`, `planner_fallback`,
`cycle`, `disengage`, `session_end`, `startup_or_shutdown_error`.

Each started cycle emits exactly one `cycle` record on completion/cancellation/error:

- `started_at`, `finished_at`, `status` (completed / cancelled / error).
- `scene`, `options`, zero-based `selection`, `plan`, and `result` (null until available).
- `latencies`: available `perception_ms`, `propose_ms`, `plan_ms`, `execute_ms`, measured with a monotonic timer.
- Error/cancellation records add reason, error/type, HOME recovery result and recovery error.
- Joint values in `result` use the configured arm's unit; the `configuration` record names the arm.

`planner_call` adds operation, model, attempt (1 or 2), latency_ms, usage, status, and
validation/error information. `planner_fallback` identifies two rejected responses.
The session end links the NPZ recording and completed-cycle count. Abrupt process
termination can prevent final cycle/recording persistence; crash-safe recording is a later concern.

## Configuration and validation

Implementation keys: `eeg: fake | replay`, `decoder: keyboard`, `perception: fake`,
`planner: nemotron | fake`, `arm: simulated | jetson`. Unknown values/fields fail validation.
The `--planner` override supports offline runs without modifying defaults.

`jetson_arm_path` (default `jetson_arm.yaml`) holds the real arm's µs limits and poses.
`jetson_command` is the SSH argv that starts the service. It defaults to Windows' OpenSSH
client at its full path, because the Git `ssh` found first on this laptop's PATH does not
work over the Jetson's USB network link. The argv sets `BatchMode` and keep-alives,
connects to the alias `jetson-usb`, and runs `python3 -u ~/mindarm_arm.py`.

Committed configuration may name that SSH host alias and remote command. Credentials,
keys, secret values, and serial ports do not belong in configuration committed to this
repository. Servo pulses never come from the LLM.

## Tests

`pytest` excludes the `integration` marker by default and blocks socket connections in
offline test cases. It covers schemas, fake signals, alpha bursts, replay/recording,
keyboard mapping/EOF, fixtures, factory/path selection, model retries/fallbacks, speed and
joint rejection, simulated state, state transitions, stale planner cancellation, and the
actual module CLI in a subprocess. API tests mock SDK responses, including token usage.
It also covers cancellation of an in-flight arm primitive before HOME, and `JetsonArm`
with an in-memory fake transport: startup, preflight rejection without messages, service
rejection, cancellation, and link loss. The Jetson service logic is tested with a fake I2C
bus: chip adoption, envelope and speed rejection, staggered start, and stopping a ramp on
input. The pose teaching tool is tested for MJPEG frame splitting, the per-target save rules,
comment-preserving YAML rewriting, the demo-plan preflight, and limit-checked operator
moves and visits. The default suite never opens SSH.

`pytest -m integration` explicitly enables the live proposal + motion test. It skips if
`NEBIUS_API_KEY` is absent, validates both responses, executes the resulting plan in the
simulator, and asserts neither operation used fallback. A valid key alone does not prove
model access or service availability; a real successful test run is required.

## Hardware measurements

Jetson Nano 4GB Developer Kit, L4T R32.7.5, Python 3.6.9, MAXN mode. The laptop reaches it
over the micro-USB network link (`jetson-usb`, 192.168.55.1). Servo supply: 5 V 10 A,
measured at 5.0–5.2 V at the PCA9685 terminal (2026-10-11).

Calibration (2026-10-11, `servo_test.py`; * = the script's 1000–2000 µs clamp, not a
physical stop). The servos were not centered before the horns were mounted.

| Ch | Joint | Servo | Min µs | Max µs | HOME µs | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | base | MG996R | 1000* | 2000* | 1500 | ~90° sweep, front at HOME |
| 1 | shoulder | MG996R | 1000* | 2000* | 1450 | 1500 buzzes (backlash at vertical balance) |
| 2 | elbow | MG996R | 1000* | 2000* | 1500 | Window may be offset |
| 3 | wrist_pitch | MG90S | 1180 | 1980 | 1600 | Real stops |
| 4 | wrist_roll | MG90S | 1000* | 2000* | 1500 | |
| 5 | gripper | SG90 | 1130 | 1950 | 1500 | Higher µs opens; 1950 open, 1130 closed |

HOME as a whole passed on 2026-10-11: the arm holds an upright "candle" posture with a
slight hum. On 2026-10-28, read-only checks over SSH showed the following:

- Bus 1 shows `0x40` and `0x70` (PCA9685 and its all-call address). The PCA9685 was at its
  power-on defaults: MODE1 `0x11`, MODE2 `0x04`, prescaler `0x1E`.
- Bus 0 shows `0x48` and `0x49` (two ADS1115, config `0x8583`).
- A USB UVC camera (`0c45:636b`, MJPG up to 1920×1080 at 30 fps) is on `/dev/video0`.
  No step uses it yet.

First real-arm cycles (2026-10-28, `--planner fake --once`, untaught poses, so only the
gripper moves):

- Startup energized all six released channels at HOME without incident.
- Later launches adopted the energized channels; none jumped.
- Four cycles completed: Home once, and three object pickups.
- `execute_ms` was 42 for Home (already at HOME). For the five-primitive pickups it was
  4766–4926 ms, against 4.5 s of ramping (three 1.5 s gripper moves). That is about 0.1 s
  of overhead per real move. Moves that change nothing return at once.

## Nemotron measurements

Model `nvidia/nemotron-3-super-120b-a12b` on Nebius Token Factory (us-central1), called from
the laptop on 2026-10-28.

`pytest -m integration` passed in 7.0 s: one proposal and one motion plan, both valid, with
no fallback.

Live demo, three proposals and two plans, with the simulated arm:

| Operation | Attempt | Latency ms | Prompt / completion tokens (reasoning) | Result |
| --- | --- | --- | --- | --- |
| propose_options | 1 | 3729 | 451 / 373 (266); 432 cached | ok |
| plan_motion (Pick up cup) | 1 | 2981 | 614 / 423 (370) | invalid: `object_label` on GRASP and LIFT |
| plan_motion (Pick up cup) | 2 | 1955 | 779 / 368 (329) | ok after the one repair |
| propose_options | 1 | 1967 | 451 / 405 (301) | ok |
| plan_motion (Home) | 1 | 1283 | 607 / 181 (167) | ok, `[HOME]` |
| propose_options | 1 | 2042 | 451 / 399 (285); 432 cached | ok |

Cycle `plan_ms` was 4940 ms with the repair and 1286 ms without one; `propose_ms` was
2.0–3.7 s. The model listed Home first. Its pickup plan stopped at LIFT, with no handover or
release; that is valid within the vocabulary. The third proposal's description contained
U+2011. Before the console fix above, that character ended the cycle with an encoding error
when output was redirected; the error path homed and disengaged as specified.

Planner benchmark (2026-10-29): `NemotronPlanner` with the defaults (temperature 0.1, JSON
mode, 4096 tokens, reasoning on) against the fixture scene. It ran 10 proposals and 10 plans,
cycling the goals Pick up cup, Pick up bottle, Pick up sports ball, and Home. Latency includes
the repair call.

| Operation | p50 | p95 | Range | Valid on first try | Repaired | Fallback | Prompt tokens (cached) / completion | Cost |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| propose_options | 2.35 s | 3.33 s | 1.54–3.71 s | 10/10 | 0 | 0 | 4510 (3024) / 3704 | $0.0047 |
| plan_motion | 2.53 s | 4.73 s | 1.13–5.05 s | 7/10 | 3 | 0 | 8482 (2960) / 4688 | $0.0068 |

- Every pickup plan had all five primitives (MOVE_ABOVE, GRASP, LIFT, MOVE_TO_POSE handover,
  RELEASE). Home returned `[HOME]`. There were no API errors.
- The run did not record why the three repaired plans were rejected.
- At $0.30/$0.90 per million input/output tokens, without a discount for cached input, one
  proposal plus one plan costs about $0.0011.

Reasoning off (same day, 3 calls per operation), with
`chat_template_kwargs: {"enable_thinking": false}`:

- Proposals took 0.6–1.4 s and all were valid.
- Plans took 0.4–0.8 s, and one of the two was invalid.
- It is not enabled in the default loop.

Nemotron 3.5 Super VL probe (same day, `api.tokenfactory.nebius.com`, which lists the model;
the `us-central1` endpoint does not). One synthetic 320×240 image, JSON mode, reasoning off:

- It answered in 2.2 s.
- The output validated as `Scene`, with object centers within 0.02 of the truth.
- It is not wired into perception.

## Open risks

1. **Deployment Python:** Jetson Nano on JetPack 4.6 ships Python 3.6; this project requires
   3.10+. Chosen for now: the orchestrator and planner stay on the laptop, and the Jetson
   runs only the Python 3.6 arm service (standard library + smbus).
2. **EEG availability:** EEG hardware is expected around week 3. Use fake acquisition and
   recorded files for independent software development.
3. **Nemotron latency:** Measured at 2–4 s per proposal and 1.3–5 s per plan (see Nemotron
   measurements). A repair attempt roughly doubles plan latency. The model's reasoning can
   exhaust the token budget; truncated output is rejected, and the budget and timeout may
   need tuning.
4. **Live service dependency:** Verified on 2026-10-28 with a real key. The demo still
   depends on Token Factory availability, and a plan can need its one repair attempt.
5. **Physical safety:** Speed and envelope checks are not collision avoidance,
   reachability, force limits, calibrated kinematics, or an emergency stop.
   - The only stops are disengage/quit and cutting the servo supply.
   - There is no position feedback: the first pulse to a released servo jumps at full speed,
     and stalls or slips go undetected.
   - After a lost link the arm stays powered in its last posture.
   - The PCA9685 runs from the Jetson's 3.3 V. If the servo supply is cut and restored while
     the chip runs, every servo jumps to its last pulse.
   - A stalled servo is invisible to software: the wrist_pitch MG90S stopped silently once
     while its pulse kept changing.
   - The shoulder hums near vertical balance at HOME (gear backlash).
   - Limits and poses are judged by eye. Four of the limits are the bring-up script's clamp,
     not physical stops, and the channel directions are not yet recorded.
   - This is a supervised tabletop prototype, not an assistive device.
6. **Signal ambiguity:** Planned 10 Hz selection overlaps the alpha range. Future decoding
   must distinguish intent selection from the eyes-closed engagement gesture using recordings.
7. **Object identity:** Label-only references cannot distinguish two objects with the same
   label. Stable object IDs will require a documented interface revision in a later step.
8. **Persistence:** EEG buffers consume memory for the session and are written on orderly
   exit. Long or crashed sessions need incremental storage in a later step. Idle gaps are
   omitted from the NPZ signal; session logs retain wall-clock events.
9. **Input timing:** This skeleton tests keyboard intent only; it does not establish SSVEP
   reliability, flicker timing, usability, or clinical benefit.
